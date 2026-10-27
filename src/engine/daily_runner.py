"""Daily basket cycle: from the closed candles to the orders on Binance Futures.

Once a day, shortly after 00:00 UTC:

1. If today's cycle has already run, it does nothing.
2. Downloads the contract rules and the closed daily candles.
3. Reads the balance and the live positions from the exchange (in ``DRY_RUN``,
   from the paper account, after applying the day's stops and funding to it).
   On the exchange, a position found without its stop gets one back.
4. Asks the orchestrator for the day's orders.
5. Dispatches the closes first and the buys after.
6. Records the cycle, which is not repeated. If a close failed or a held
   symbol had no candles, the day is left open and run again a few minutes
   later (orders already sent are not repeated), until the entry deadline.

The candles are requested before reading the account because the account needs
them: they are used to check the stops of the paper account and to estimate the
risk of a position that shows up without a stop.

The balance passed to the orchestrator is the realized balance of the account,
not the free collateral: position sizes and portfolio limits were measured with
it. Free collateral drops with every open position and, used as the base, would
shrink the buys that follow.

Usage::

    python -m src.engine.daily_runner            # waits for each 00:01:30 UTC
    python -m src.engine.daily_runner --panel    # the same, also serving the basket dashboard
    python -m src.engine.daily_runner --once     # one cycle now, then exits
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import Settings, get_settings
from src.console import configure_logging
from src.data.candle_loader import DailyCandleLoader
from src.engine.daily_reporter import SETTLEMENT_TRADES, CycleSnapshot, DailyReporter
from src.engine.notifier import Notifier, build_notifier, cycle_message, failure_message
from src.engine.orchestrator import CLOSE_LONG, OPEN_LONG, BasketOrchestrator, OrderIntent
from src.engine.post_cycle import LlmAuditor, ping_healthcheck
from src.engine.state_store import DailyStateStore, StateError
from src.execution.errors import BrokerError, BrokerUnavailableError
from src.execution.binance_futures_executor import (
    DONE,
    FILLED,
    DRY_RUN,
    LIVE,
    NO_POSITION,
    PARTIAL_CLOSE,
    PROTECT,
    SIMULATED,
    STOP_FAILED_CLOSED,
    UNPROTECTED,
    BinanceFuturesExecutor,
    FuturesAccount,
    FuturesPosition,
    PaperFuturesAccount,
)
from src.instance_lock import InstanceLock, StateInUseError
from src.risk.portfolio_risk import PortfolioRiskManager
from src.risk.turtle_engine import RiskError
from src.strategies.trend_breakout import StrategyError, TrendBreakoutStrategy

# Status ``run_daily_cycle`` ends with.
COMPLETED = "COMPLETED"
COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
ALREADY_EXECUTED = "ALREADY_EXECUTED"
WRONG_DATE = "WRONG_DATE"
NO_DATA = "NO_DATA"
NO_BALANCE = "NO_BALANCE"
# A close did not go through, or a held symbol had no candles to check its exit: the day is not marked
# done, so it is run again (orders already sent are not repeated) until max_entry_delay_minutes.
INCOMPLETE = "INCOMPLETE"
# Cycles that did not start (or finish) and are worth retrying later.
RETRY = frozenset({NO_DATA, NO_BALANCE, INCOMPLETE})
RETRY_PROBLEMS = {
    NO_DATA: "No symbol has valid daily candles.",
    NO_BALANCE: "The account has no balance.",
    INCOMPLETE: "A sell was not executed or a held position has no candles to check its exit.",
}

# Result of an order that was never sent.
ERROR = "ERROR"
ALREADY_DISPATCHED = "ALREADY_DISPATCHED"
CLOSE_FAILED = "CLOSE_FAILED"
TOO_LATE = "TOO_LATE"
# An entry signal seen only when a pending day was run again: entries are decided once, by the first attempt.
ENTRIES_DECIDED = "ENTRIES_DECIDED"
# Orders that already moved the account: they are not repeated after a restart.
SENT = DONE | {STOP_FAILED_CLOSED, UNPROTECTED}
# Stop of the simulated account hit since the previous cycle; written down with the day's orders.
STOP_HIT = "STOP_HIT"
# Results that are not a failure of the cycle.
RESOLVED = DONE | {NO_POSITION, ALREADY_DISPATCHED}
LATE_WARNING_MINUTES = 5.0


class DailyRunner:
    """Runs the daily basket cycle and remembers which days were already traded."""

    def __init__(
        self,
        executor: BinanceFuturesExecutor,
        loader: DailyCandleLoader,
        store: DailyStateStore,
        symbols: Sequence[str],
        *,
        risk_manager: PortfolioRiskManager | None = None,
        strategy: TrendBreakoutStrategy | None = None,
        reporter: DailyReporter | None = None,
        notifier: Notifier | None = None,
        notify_quiet_days: bool = True,
        healthcheck_url: str | None = None,
        auditor: LlmAuditor | None = None,
        run_delay_seconds: float = 90.0,
        max_entry_delay_minutes: float = 360.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """
        Args:
            executor: account, rules and orders of the exchange.
            loader: closed daily candles.
            store: memory of the cycles already run.
            symbols: the basket.
            risk_manager: portfolio limits; by default, the built-in ones.
            strategy: entry and exit rule; by default, the 20/10 breakout.
            reporter: if given, the report of each cycle is saved when it ends.
            notifier: if given, it sends an alert about what each cycle did and
                about the ones that could not run.
            notify_quiet_days: whether cycles with no buys, sells, stops or
                failures send an alert too.
            healthcheck_url: gets a GET after every COMPLETED cycle (dead-man's switch).
            auditor: if given, answers the report's LLM questions after each cycle.
            run_delay_seconds: seconds after 00:00 UTC at which the cycle is
                due. It lets Binance close the candle and settle funding.
            max_entry_delay_minutes: delay past which the cycle no longer
                buys. The signal is yesterday's close; hours later the price
                may be far from it. Closes always run.
            clock: for tests.
        """
        if not symbols:
            raise ValueError("the basket cannot be empty")
        self._executor = executor
        self._loader = loader
        self._store = store
        self._symbols = list(symbols)
        self._risk_manager = risk_manager if risk_manager is not None else PortfolioRiskManager()
        self._strategy = strategy if strategy is not None else TrendBreakoutStrategy()
        self._reporter = reporter
        self._notifier = notifier
        self._notify_quiet_days = notify_quiet_days
        self._healthcheck_url = healthcheck_url
        self._auditor = auditor
        self._run_delay = timedelta(seconds=run_delay_seconds)
        self._max_entry_delay = max_entry_delay_minutes
        self._clock = clock

    # --- One cycle --------------------------------------------------------

    def run_daily_cycle(self, target_date: str | None = None) -> dict[str, Any]:
        """Run today's cycle (UTC), only once.

        Args:
            target_date: cycle date, ``YYYY-MM-DD``. Only today's is accepted:
                candles and prices are always the current ones, so the cycle
                of another day cannot be traded.

        Returns:
            Summary of the cycle, with ``status`` and what was done.

        Raises:
            BrokerError: if the exchange does not answer when reading the
                candles or the account. The cycle is not recorded and can be retried.
            StateError: if ``target_date`` is not a date or the state cannot
                be saved.
        """
        now = self._now()
        today = now.date().isoformat()
        cycle_date = today if target_date is None else target_date
        if self._store.is_cycle_already_executed(cycle_date):
            logger.info("Cycle {}: already run; not repeated", cycle_date)
            return {"cycle_date": cycle_date, "status": ALREADY_EXECUTED}
        if cycle_date != today:
            logger.error("Cycle {}: not run, today is {} (UTC) and only today's cycle can be traded",
                         cycle_date, today)
            return {"cycle_date": cycle_date, "status": WRONG_DATE}
        earlier_attempt = (self._store.get_cycle(cycle_date) or {}).get("incomplete")
        late = (now - self.scheduled_for(now.date())).total_seconds() / 60.0
        if late > LATE_WARNING_MINUTES:
            logger.warning("Cycle {}: starting {:.0f} minutes late", cycle_date, late)
        logger.info("Cycle {}: starting in mode {}", cycle_date, self._executor.describe_mode())

        # A position open in a symbol that is no longer in the basket is still managed until
        # it is closed: without its candles neither its exit nor its simulated stop would be evaluated.
        orphans = [symbol for symbol in self._executor.held_symbols() if symbol not in self._symbols]
        if orphans:
            logger.warning("Cycle {}: there are positions outside the basket ({}); they are managed until closed, "
                           "without buying them again", cycle_date, ", ".join(orphans))
        symbols = [*self._symbols, *orphans]
        rules = self._executor.load_market_rules(symbols, refresh=True)
        market = self._loader.load(symbols)
        if not market:
            logger.error("Cycle {}: no symbol has valid daily candles; not trading", cycle_date)
            return {"cycle_date": cycle_date, "status": NO_DATA}

        # Stops hit and repairs are written down at once, like orders: if the day has to be run again,
        # its summary still has them.
        for trade in self._executor.settle_paper(market):
            self._store.record_dispatch(cycle_date, trade["symbol"], STOP_HIT,
                                        {**trade, "action": STOP_HIT, "status": SIMULATED})
        distances = self._stop_distances(market)
        for result in self._executor.protect_open_positions(distances, cycle_date):
            self._store.record_dispatch(cycle_date, result["symbol"], result["action"], result)
        account = self._executor.get_account()
        positions = self._executor.get_open_positions(distances)
        if not account.wallet_balance > 0.0:
            logger.error("Cycle {}: the account has no balance ({:.2f} USDT); not trading",
                         cycle_date, account.wallet_balance)
            return {"cycle_date": cycle_date, "status": NO_BALANCE}
        logger.info(
            "Cycle {}: balance {:.2f} USDT | free collateral {:.2f} | unrealized PnL {:+.2f} | {} positions | "
            "{} of {} symbols with candles",
            cycle_date, account.wallet_balance, account.available_balance, account.unrealized_pnl,
            len(positions), len(market), len(symbols),
        )
        # Outside the basket only the exit is evaluated: without a position, the symbol is not an entry candidate.
        held = {position.symbol for position in positions}
        blind = sorted(held - set(market))
        if blind:
            logger.warning("Cycle {}: no candles for {}, which are held: their exit cannot be checked today",
                           cycle_date, ", ".join(blind))
        # A held contract Binance no longer trades (on its way to delisting) keeps its exit checked with the
        # last rules seen; it is never a candidate to enter.
        specs = {symbol: rule.to_spec() for symbol, rule in rules.items()}
        retired = sorted(held - set(rules))
        for symbol in retired:
            known = self._executor.known_rules(symbol)
            if known is not None:
                specs[symbol] = known.to_spec()
        if retired:
            logger.warning("Cycle {}: Binance no longer trades {}, which are held", cycle_date, ", ".join(retired))
        symbols = [symbol for symbol in symbols if symbol in self._symbols or symbol in held]
        market = {symbol: candles for symbol, candles in market.items() if symbol in symbols}

        self._store.mark_cycle_started(cycle_date)
        orchestrator = BasketOrchestrator(self._risk_manager, self._strategy, specs)
        intents = orchestrator.evaluate_daily_cycle(market, positions, account.wallet_balance)
        closes = [intent for intent in intents if intent.action == CLOSE_LONG]
        opens = [intent for intent in intents if intent.action == OPEN_LONG]

        # Closes first: they free the risk the buys were approved with.
        done = self._store.dispatched(cycle_date)
        stopped = [entry for entry in done.values() if entry.get("action") == STOP_HIT]
        protected = [entry for entry in done.values() if entry.get("action") == PROTECT]
        closed = [self._dispatch(cycle_date, intent, done) for intent in closes]
        block = None
        if any(result["status"] not in RESOLVED for result in closed):
            block = CLOSE_FAILED
            logger.warning("Cycle {}: a close was not executed; no positions are opened today", cycle_date)
        elif opens and late > self._max_entry_delay:
            block = TOO_LATE
            logger.warning("Cycle {}: {:.0f} minutes late is over the maximum of {:g}; no positions are opened today",
                           cycle_date, late, self._max_entry_delay)
        if earlier_attempt is not None:
            # Running a pending day again only finishes its exits. Its entries were settled by the first
            # attempt, as if the cycle had run in one go: nothing new is bought and nothing is resent.
            entry_block = earlier_attempt.get("entry_block") or ENTRIES_DECIDED
            opened = [_as_sent(done[key]) if (key := f"{intent.symbol}:{OPEN_LONG}") in done
                      else _skipped(intent, entry_block) for intent in opens]
        else:
            opened = [
                self._dispatch(cycle_date, intent, done) if block is None else _skipped(intent, block)
                for intent in opens
            ]

        unresolved = [result["symbol"] for result in closed if result["status"] not in RESOLVED]
        # Only a symbol that still trades can get its candles back later the same day.
        waiting = [symbol for symbol in blind if symbol in rules]
        if (unresolved or waiting) and late < self._max_entry_delay:
            logger.warning("Cycle {}: left pending ({}); it runs again later without resending what was sent",
                           cycle_date, ", ".join([*unresolved, *waiting]))
            first_block = earlier_attempt.get("entry_block") if earlier_attempt is not None else block
            self._store.mark_cycle_incomplete(cycle_date, first_block)
            return {"cycle_date": cycle_date, "status": INCOMPLETE, "unresolved": unresolved, "without_data": blind}
        stop_check = self._verify_stops(cycle_date, distances)
        # A day run more than once: what earlier attempts sent is part of it even when this attempt no longer
        # produces it (the position it opened or closed has changed since).
        closed, opened = _with_earlier(done, CLOSE_LONG, closed), _with_earlier(done, OPEN_LONG, opened)
        results = [*closed, *opened]
        failed = [result for result in results if result["status"] not in RESOLVED]
        summary = {
            "cycle_date": cycle_date,
            "status": COMPLETED_WITH_ERRORS if failed else COMPLETED,
            "mode": self._executor.mode,
            "late_minutes": round(late, 1),
            "wallet_balance": account.wallet_balance,
            "available_balance": account.available_balance,
            "positions_before": sorted(position.symbol for position in positions),
            "symbols_with_data": len(market),
            "stopped": [trade["symbol"] for trade in stopped],
            "protected": [_brief(result) for result in protected],
            "without_data": blind,
            "untradable": retired,
            "stop_check": stop_check,
            "closes": [_brief(result) for result in closed],
            "opens": [_brief(result) for result in opened],
            "holds": dict(Counter(_family(intent.reason) for intent in intents
                                  if intent.action not in (CLOSE_LONG, OPEN_LONG))),
        }
        self._store.mark_cycle_completed(cycle_date, summary)
        log = logger.warning if failed else logger.info
        log(
            "Cycle {} finished ({}): {} closes, {} buys, {} not executed | not traded: {}",
            cycle_date, summary["status"],
            sum(result["status"] in DONE for result in closed),
            sum(result["status"] in DONE for result in opened),
            len(failed), summary["holds"] or "none",
        )
        # After recording the cycle: a failure writing the report or alerting must not leave the day half done.
        after = self._account_after(cycle_date)
        paths = None
        if after is not None:
            paths = self._report(summary, account.wallet_balance, market, rules, intents, results, symbols, *after)
        self._notify_cycle(summary, stopped, after, self._exit_channels(market, after))
        # Optional, slowest last: a monitoring ping and the LLM review. Neither ever raises.
        if summary["status"] == COMPLETED:
            ping_healthcheck(self._healthcheck_url)
        if self._auditor is not None and paths is not None:
            self._auditor.audit(paths[1], paths[0])
        return summary

    def _account_after(self, cycle_date: str) -> tuple[FuturesAccount, list[FuturesPosition]] | None:
        """The account and its positions after the orders, for the report and the alert; ``None`` if unreadable."""
        if self._reporter is None and self._notifier is None:
            return None
        try:
            return self._executor.get_account_and_positions()
        except BrokerError as error:
            logger.warning("Cycle {}: could not read the account after the orders ({}); the report is not written",
                           cycle_date, error)
            return None

    def _notify_cycle(
        self, summary: dict[str, Any], stopped: list[dict[str, Any]],
        after: tuple[FuturesAccount, list[FuturesPosition]] | None,
        exit_channels: dict[str, float] | None = None,
    ) -> None:
        """Send the alert about what the cycle did. Never raises."""
        if self._notifier is None:
            return
        try:
            text, eventful = cycle_message(summary, stopped, *(after or (None, None)), exit_channels)
            if eventful or self._notify_quiet_days:
                self._notifier.send(text)
        except Exception:
            logger.exception("Cycle {}: alert not sent", summary["cycle_date"])

    def _notify_failure(self, problem: str, retry_seconds: float | None, kind: str | None = None) -> None:
        """Alerts, once a day per kind of failure, that today's cycle could not run. Never raises."""
        if self._notifier is None:
            return
        cycle_date = self._now().date().isoformat()
        kind = kind or ("retry" if retry_seconds else "halt")
        try:
            self._notifier.send(failure_message(self._executor.mode, cycle_date, problem, retry_seconds),
                                once_key=f"{self._executor.mode}:{cycle_date}:{kind}")
        except Exception:
            logger.exception("Cycle {}: failure alert not sent", cycle_date)

    def _report(
        self, summary: dict[str, Any], balance: float, market: dict[str, pd.DataFrame], rules: dict[str, Any],
        intents: list[OrderIntent], results: list[dict[str, Any]], symbols: list[str],
        account: FuturesAccount, positions: list[FuturesPosition],
    ) -> tuple[Path, Path] | None:
        """Writes the cycle report and returns its paths (Markdown, JSON). Never raises: the cycle is already done."""
        if self._reporter is None:
            return None
        try:
            closed_trades = self._executor.closed_trades()
            # The account is the one after the orders: the report shows how it ended up.
            return self._reporter.write(CycleSnapshot(
                cycle_date=summary["cycle_date"], mode=summary["mode"], status=summary["status"],
                late_minutes=summary["late_minutes"], balance=balance, account=account, positions=positions,
                market=market, rules=rules, intents=intents, results=results, symbols=symbols,
                stopped=summary["stopped"], opened_at=self._open_times(positions),
                closed_trades=closed_trades, stop_check=summary.get("stop_check"),
                settlement_market=self._settlement_candles(summary["cycle_date"], closed_trades, market),
            ))
        except BrokerError as error:
            logger.warning("Cycle {}: report not written, {}", summary["cycle_date"], error)
        except Exception:
            logger.exception("Cycle {}: report not written", summary["cycle_date"])
        return None

    def _settlement_candles(
        self, cycle_date: str, closed_trades: list[dict[str, Any]], market: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Read-only candles of the report's closed trades whose coin is outside the cycle's market.

        They are loaded once the cycle is done, so they never reach the orchestrator: they only fill the
        post-exit tables of the report. A coin that fails to load just keeps its blanks. Never raises.
        """
        missing = sorted({str(trade["symbol"]) for trade in closed_trades[-SETTLEMENT_TRADES:]
                          if trade.get("symbol") and trade["symbol"] not in market})
        if not missing:
            return {}
        try:
            return self._loader.load(missing)
        except BrokerError as error:
            logger.warning("Cycle {}: no candles for the closed trades outside the basket ({}), {}",
                           cycle_date, ", ".join(missing), error)
            return {}

    def _exit_channels(
        self, market: dict[str, pd.DataFrame], after: tuple[FuturesAccount, list[FuturesPosition]] | None,
    ) -> dict[str, float]:
        """Exit channel of each position held after the cycle, for the alert."""
        channels = {}
        for position in (after[1] if after else []):
            try:
                channels[position.symbol] = self._strategy.levels(market[position.symbol]).exit_channel
            except (KeyError, StrategyError, RiskError):
                continue
        return channels

    def _verify_stops(self, cycle_date: str, distances: dict[str, float]) -> dict[str, Any]:
        """Every open position must end the cycle with its stop. A failure here is reported, never raised."""
        try:
            check = self._executor.verify_stops(distances, cycle_date)
        except BrokerError as error:
            logger.warning("Cycle {}: the stops could not be checked ({})", cycle_date, error)
            return {"error": str(error)}
        for result in check["repaired"]:
            self._store.record_dispatch(cycle_date, result["symbol"], result["action"], result)
        check["repaired"] = [_brief(result) for result in check["repaired"]]
        if check["unprotected"] or check["uncovered"]:
            logger.error("Cycle {}: positions without a stop {} or with a partial stop {}", cycle_date,
                         check["unprotected"], check["uncovered"])
        else:
            logger.info("Cycle {}: stops checked, {} of {} positions protected", cycle_date,
                        check["protected"], check["positions"])
        return check

    def _open_times(self, positions: list[FuturesPosition]) -> dict[str, int]:
        """When each held position was opened (ms), to measure how far it has gone each way since.

        The simulated account records it. On the exchange it is taken from the cycle that bought it, at the
        cycle's scheduled time; a position bought outside the bot has none.
        """
        times = self._executor.open_times()
        missing = {position.symbol for position in positions} - set(times)
        for cycle in self._store.recent_cycles(400) if missing else []:
            for result in cycle.get("dispatched", {}).values():
                symbol = result.get("symbol")
                if symbol in missing and result.get("action") == OPEN_LONG and result.get("status") in DONE:
                    day = date.fromisoformat(cycle["cycle_date"])
                    times[symbol] = int(self.scheduled_for(day).timestamp() * 1000)
                    missing.discard(symbol)
        return times

    def _dispatch(self, cycle_date: str, intent: OrderIntent, done: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Execute an order and record it. An exchange failure does not stop the others."""
        previous = done.get(f"{intent.symbol}:{intent.action}")
        if previous is not None and previous.get("status") in SENT:
            logger.info("{}: {} was already sent in this cycle ({}); not repeated",
                        intent.symbol, intent.action, previous["status"])
            return _as_sent(previous)
        # The rest of a close that filled in part goes out under its own order id, and adds up with it.
        parts = (previous.get("parts") or [previous]) if previous and previous.get("status") == PARTIAL_CLOSE else []
        cycle_id = f"{cycle_date}-{len(parts) + 1}" if parts else cycle_date
        try:
            result = self._executor.execute_order_intent(intent, cycle_id)
        except BrokerError as error:
            logger.error("{}: {} was not executed: {}", intent.symbol, intent.action, error)
            result = _skipped(intent, ERROR, str(error))
        if parts:
            result = _rest_of_close(parts, result)
        self._store.record_dispatch(cycle_date, intent.symbol, intent.action, result)
        return result

    def _stop_distances(self, market: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Protective distance of each symbol, in case a position shows up without a stop."""
        distances = {}
        for symbol, candles in market.items():
            try:
                distances[symbol] = self._strategy.protective_stop_distance(candles)
            except (StrategyError, RiskError):
                continue
        return distances

    # --- Scheduling -------------------------------------------------------

    @property
    def mode(self) -> str:
        """Mode it is really running in (``DRY_RUN``, ``TESTNET`` or ``LIVE``)."""
        return self._executor.mode

    def scheduled_for(self, day: date) -> datetime:
        """Time (UTC) at which that day's cycle is due."""
        return cycle_time(day, self._run_delay)

    def next_run_at(self, now: datetime | None = None) -> datetime:
        """Next cycle time, after ``now``."""
        return next_cycle_time(now or self._now(), self._run_delay)

    def run_forever(self, stop: threading.Event, retry_seconds: float = 300.0) -> None:
        """Run one cycle each day at its time until ``stop`` is set.

        On startup, if today's cycle is pending and its time has passed, it
        runs right away. A cycle that fails before it starts trading (the
        exchange does not answer, the candles are not there yet) is retried
        every ``retry_seconds``.
        """
        while not stop.is_set():
            now = self._now()
            pending = now >= self.scheduled_for(now.date()) and not self._store.is_cycle_already_executed(
                now.date().isoformat()
            )
            if pending:
                try:
                    status = self.run_daily_cycle()["status"]
                except BrokerError as error:
                    logger.warning("The cycle could not run: {}. Retrying in {:g} s", error, retry_seconds)
                    self._notify_failure(str(error), retry_seconds)
                    status = None
                except (BrokerUnavailableError, StateError) as error:
                    # Waiting does not fix it: the account or the state needs a person.
                    self._notify_failure(str(error), None)
                    raise
                except Exception as error:
                    logger.exception("Unexpected failure in the cycle; retrying in {:g} s", retry_seconds)
                    # Its own once-a-day key: an exchange outage alerted earlier must not hide a bug.
                    self._notify_failure(f"Unexpected failure ({type(error).__name__}: {str(error)[:200]}).",
                                         retry_seconds, kind="unexpected")
                    status = None
                if status in RETRY:
                    self._notify_failure(RETRY_PROBLEMS[status], retry_seconds)
                if status is None or status in RETRY:
                    stop.wait(retry_seconds)
                    continue
            wait = (self.next_run_at() - self._now()).total_seconds()
            logger.info("Next cycle: {} UTC", self.next_run_at().strftime("%Y-%m-%d %H:%M:%S"))
            stop.wait(max(wait, 1.0))

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc)


def _skipped(intent: OrderIntent, status: str, reason: str = "") -> dict[str, Any]:
    return {"symbol": intent.symbol, "action": intent.action, "status": status, "reason": reason or status,
            "payloads": []}


def _as_sent(previous: dict[str, Any]) -> dict[str, Any]:
    """An order a previous attempt already sent. One that went through reads ALREADY_DISPATCHED; one that
    went wrong (bought and closed again, left without stop) keeps its status, so it is not taken for done."""
    if previous.get("status") in DONE:
        return {**previous, "status": ALREADY_DISPATCHED}
    return {**previous, "repeated": True}


def _rest_of_close(parts: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    """A close that filled in part earlier, with what this attempt did about the rest, as one result."""
    fills = [part for part in [*parts, result]
             if isinstance(part.get("quantity"), (int, float)) and isinstance(part.get("price"), (int, float))
             and part["quantity"] > 0]
    quantity = sum(part["quantity"] for part in fills)
    merged = {**result, "parts": [*parts, result], "quantity": quantity,
              "price": sum(part["quantity"] * part["price"] for part in fills) / quantity if quantity else None}
    if result.get("status") == NO_POSITION:
        # Nothing left to sell: the rest went out some other way (its stop) between the two attempts.
        merged.update(status=FILLED, reason="the rest went out at its stop")
    return merged


def _with_earlier(
    done: dict[str, dict[str, Any]], action: str, current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """``current`` preceded by the orders of ``action`` an earlier attempt of the day sent and this one did not."""
    seen = {result["symbol"] for result in current}
    earlier = [result for result in done.values()
               if result.get("action") == action and result.get("status") in SENT | {PARTIAL_CLOSE}
               and result.get("symbol") not in seen]
    return [*earlier, *current]


def _brief(result: dict[str, Any]) -> dict[str, Any]:
    """What the summary keeps of each order; the full detail goes in ``dispatched``."""
    return {key: result[key] for key in ("symbol", "status", "reason", "quantity", "price", "stop_price", "pnl")
            if result.get(key) is not None}


def _family(reason: str) -> str:
    """The reason without its detail (``INVALID_DATA: ...`` -> ``INVALID_DATA``)."""
    return reason.split(":", 1)[0] or "HOLD"


# --- Startup ----------------------------------------------------------------


def cycle_time(day: date, run_delay: timedelta) -> datetime:
    """Time (UTC) at which a day's cycle is due: 00:00 UTC plus the wait margin."""
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + run_delay


def next_cycle_time(now: datetime, run_delay: timedelta) -> datetime:
    """Next cycle time, after ``now``."""
    today = cycle_time(now.date(), run_delay)
    return today if now < today else cycle_time(now.date() + timedelta(days=1), run_delay)


def effective_mode(settings: Settings) -> str:
    """The requested mode, unless ``BOT_DRY_RUN`` is still on: then there is no real money."""
    mode = settings.futures.execution_mode
    if mode == LIVE and settings.dry_run:
        logger.warning("BOT_FUTURES__EXECUTION_MODE=LIVE with BOT_DRY_RUN=true: staying in DRY_RUN")
        return DRY_RUN
    return mode


def state_path(settings: Settings, account: str) -> Path:
    """File with the cycles run on that account (``dry_run``, ``testnet`` or ``live``)."""
    return settings.state_dir / f"runtime_state_{account}.json"


def reports_dir(settings: Settings, account: str) -> Path:
    """Folder of that account's daily reports. It lives next to the state, which is what is kept."""
    return settings.state_dir / "reports" / account


def build_runner(settings: Settings) -> tuple[DailyRunner, InstanceLock]:
    """Build the daily cycle from the settings and take the lock on its state."""
    futures = settings.futures
    executor = BinanceFuturesExecutor(
        mode=effective_mode(settings),
        live_trading_acknowledged=futures.live_trading_acknowledged,
        api_key=futures.api_key.get_secret_value() if futures.api_key else None,
        api_secret=futures.api_secret.get_secret_value() if futures.api_secret else None,
        paper=PaperFuturesAccount(
            settings.state_dir / "paper_futures.json", settings.paper_capital, futures.paper_fee_rate
        ),
        leverage=futures.leverage,
        stop_working_type=futures.stop_working_type,
        max_slippage_pct=futures.max_slippage_pct,
    )
    # One set of files per account: paper trading never mixes with the real thing.
    account = executor.mode.lower()
    lock = InstanceLock(settings.state_dir / f"daily_runner_{account}.lock")
    if not lock.acquire():
        raise StateInUseError(f"Another daily cycle is already running with this state ({lock.path})")
    notifier = build_notifier(settings)
    logger.info("Telegram alerts: {}", "enabled" if notifier is not None else
                "disabled (BOT_TELEGRAM__BOT_TOKEN and BOT_TELEGRAM__CHAT_ID are missing)")
    strategy = TrendBreakoutStrategy(risk_pct=futures.risk_pct, max_leverage=futures.max_position_fraction)
    risk_manager = PortfolioRiskManager(max_portfolio_heat_pct=futures.max_heat_pct)
    runner = DailyRunner(
        executor,
        DailyCandleLoader(executor, futures.candles),
        DailyStateStore(state_path(settings, account)),
        futures.symbol_list,
        risk_manager=risk_manager,
        strategy=strategy,
        reporter=DailyReporter(reports_dir(settings, account), strategy, risk_manager),
        notifier=notifier,
        notify_quiet_days=settings.telegram.quiet_days,
        healthcheck_url=settings.healthcheck_ping_url.get_secret_value() if settings.healthcheck_ping_url else None,
        auditor=_auditor(settings),
        run_delay_seconds=futures.run_delay_seconds,
        max_entry_delay_minutes=futures.max_entry_delay_minutes,
    )
    return runner, lock


def _auditor(settings: Settings) -> LlmAuditor | None:
    """The LLM review of each report, if it was turned on. A missing SDK or key only disables it."""
    audit = settings.audit_llm
    if not audit.enabled:
        return None
    try:
        auditor = LlmAuditor(api_key=audit.api_key.get_secret_value() if audit.api_key else None,
                             model=audit.model, effort=audit.effort)
    except Exception as error:  # noqa: BLE001 - optional feature; the cycle runs without it
        logger.warning("LLM auditor off: it could not be set up ({})", type(error).__name__)
        return None
    logger.info("LLM auditor on ({}, effort {}): every report is sent to Anthropic", audit.model, audit.effort)
    return auditor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily basket cycle on Binance USDT-M Futures")
    parser.add_argument("--once", action="store_true", help="run today's cycle now and exit")
    parser.add_argument("--date", help="cycle date (YYYY-MM-DD); only today's date in UTC is accepted")
    parser.add_argument("--panel", action="store_true",
                        help="also serve the basket dashboard from this process (no effect with --once)")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)
    lock: InstanceLock | None = None
    panel = None
    try:
        runner, lock = build_runner(settings)
        logger.info("Basket: {}", ", ".join(settings.futures.symbol_list))
        if args.once or args.date:
            return 0 if runner.run_daily_cycle(args.date)["status"] in (COMPLETED, ALREADY_EXECUTED) else 1
        if args.panel:
            # Here and not at the top: the dashboard imports this module.
            from src.ui.dashboard import serve_in_background  # noqa: PLC0415

            panel = serve_in_background(settings, runner.mode.lower())
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        runner.run_forever(stop)
    except (BrokerUnavailableError, StateInUseError, StateError) as error:
        logger.error("{}", error)
        return 1
    except BrokerError as error:
        logger.error("The cycle could not run: {}", error)
        return 1
    except Exception:
        logger.exception("Fatal failure: the daily cycle stops")
        return 1
    finally:
        if panel is not None:
            panel.shutdown()
            panel.server_close()
        if lock is not None:
            lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
