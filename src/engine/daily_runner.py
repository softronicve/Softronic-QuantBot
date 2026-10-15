"""Daily basket cycle: from the closed candles to the orders on Binance Futures.

Once a day, shortly after 00:00 UTC:

1. If today's cycle has already run, it does nothing.
2. Downloads the contract rules and the closed daily candles.
3. Reads the balance and the live positions from the exchange (in ``DRY_RUN``,
   from the paper account, after applying the day's stops and funding to it).
4. Asks the orchestrator for the day's orders.
5. Dispatches the closes first and the buys after.
6. Records the cycle, which is not repeated.

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
from src.engine.daily_reporter import CycleSnapshot, DailyReporter
from src.engine.notifier import Notifier, build_notifier, cycle_message, failure_message
from src.engine.orchestrator import CLOSE_LONG, OPEN_LONG, BasketOrchestrator, OrderIntent
from src.engine.state_store import DailyStateStore, StateError
from src.execution.errors import BrokerError, BrokerUnavailableError
from src.execution.binance_futures_executor import (
    DONE,
    DRY_RUN,
    LIVE,
    NO_POSITION,
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
# Cycles that never started and are worth retrying later.
RETRY = frozenset({NO_DATA, NO_BALANCE})
RETRY_PROBLEMS = {
    NO_DATA: "No symbol has valid daily candles.",
    NO_BALANCE: "The account has no balance.",
}

# Result of an order that was never sent.
ERROR = "ERROR"
ALREADY_DISPATCHED = "ALREADY_DISPATCHED"
CLOSE_FAILED = "CLOSE_FAILED"
TOO_LATE = "TOO_LATE"
# Orders that already moved the account: they are not repeated after a restart.
SENT = DONE | {STOP_FAILED_CLOSED, UNPROTECTED}
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
        rules = self._executor.load_market_rules(symbols)
        market = self._loader.load(symbols)
        if not market:
            logger.error("Cycle {}: no symbol has valid daily candles; not trading", cycle_date)
            return {"cycle_date": cycle_date, "status": NO_DATA}

        stopped = self._executor.settle_paper(market)
        account = self._executor.get_account()
        positions = self._executor.get_open_positions(self._stop_distances(market))
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
        symbols = [symbol for symbol in symbols if symbol in self._symbols or symbol in held]
        market = {symbol: candles for symbol, candles in market.items() if symbol in symbols}

        self._store.mark_cycle_started(cycle_date)
        orchestrator = BasketOrchestrator(
            self._risk_manager, self._strategy, {symbol: rule.to_spec() for symbol, rule in rules.items()}
        )
        intents = orchestrator.evaluate_daily_cycle(market, positions, account.wallet_balance)
        closes = [intent for intent in intents if intent.action == CLOSE_LONG]
        opens = [intent for intent in intents if intent.action == OPEN_LONG]

        # Closes first: they free the risk the buys were approved with.
        done = self._store.dispatched(cycle_date)
        closed = [self._dispatch(cycle_date, intent, done) for intent in closes]
        block = None
        if any(result["status"] not in DONE | {NO_POSITION, ALREADY_DISPATCHED} for result in closed):
            block = CLOSE_FAILED
            logger.warning("Cycle {}: a close was not executed; no positions are opened today", cycle_date)
        elif opens and late > self._max_entry_delay:
            block = TOO_LATE
            logger.warning("Cycle {}: {:.0f} minutes late is over the maximum of {:g}; no positions are opened today",
                           cycle_date, late, self._max_entry_delay)
        opened = [
            self._dispatch(cycle_date, intent, done) if block is None else _skipped(intent, block)
            for intent in opens
        ]

        results = [*closed, *opened]
        failed = [result for result in results if result["status"] not in DONE | {NO_POSITION, ALREADY_DISPATCHED}]
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
        if after is not None:
            self._report(summary, account.wallet_balance, market, rules, intents, results, symbols, *after)
        self._notify_cycle(summary, stopped, after)
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
    ) -> None:
        """Send the alert about what the cycle did. Never raises."""
        if self._notifier is None:
            return
        try:
            text, eventful = cycle_message(summary, stopped, *(after or (None, None)))
            if eventful or self._notify_quiet_days:
                self._notifier.send(text)
        except Exception:
            logger.exception("Cycle {}: alert not sent", summary["cycle_date"])

    def _notify_failure(self, problem: str, retry_seconds: float | None) -> None:
        """Send the alert, only once per day, that today's cycle could not run. Never raises."""
        if self._notifier is None:
            return
        cycle_date = self._now().date().isoformat()
        kind = "retry" if retry_seconds else "halt"
        try:
            self._notifier.send(failure_message(self._executor.mode, cycle_date, problem, retry_seconds),
                                once_key=f"{self._executor.mode}:{cycle_date}:{kind}")
        except Exception:
            logger.exception("Cycle {}: failure alert not sent", cycle_date)

    def _report(
        self, summary: dict[str, Any], balance: float, market: dict[str, pd.DataFrame], rules: dict[str, Any],
        intents: list[OrderIntent], results: list[dict[str, Any]], symbols: list[str],
        account: FuturesAccount, positions: list[FuturesPosition],
    ) -> None:
        """Save the cycle report. Never raises: the cycle is already done."""
        if self._reporter is None:
            return
        try:
            # The account is the one after the orders: the report shows how it ended up.
            self._reporter.write(CycleSnapshot(
                cycle_date=summary["cycle_date"], mode=summary["mode"], status=summary["status"],
                late_minutes=summary["late_minutes"], balance=balance, account=account, positions=positions,
                market=market, rules=rules, intents=intents, results=results, symbols=symbols,
                stopped=summary["stopped"],
            ))
        except BrokerError as error:
            logger.warning("Cycle {}: report not written, {}", summary["cycle_date"], error)
        except Exception:
            logger.exception("Cycle {}: report not written", summary["cycle_date"])

    def _dispatch(self, cycle_date: str, intent: OrderIntent, done: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Execute an order and record it. An exchange failure does not stop the others."""
        previous = done.get(f"{intent.symbol}:{intent.action}")
        if previous is not None and previous.get("status") in SENT:
            logger.info("{}: {} was already sent in this cycle ({}); not repeated",
                        intent.symbol, intent.action, previous["status"])
            return {**previous, "status": ALREADY_DISPATCHED}
        try:
            result = self._executor.execute_order_intent(intent, cycle_date)
        except BrokerError as error:
            logger.error("{}: {} was not executed: {}", intent.symbol, intent.action, error)
            result = _skipped(intent, ERROR, str(error))
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
                    self._notify_failure(f"Unexpected failure ({type(error).__name__}).", retry_seconds)
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
        run_delay_seconds=futures.run_delay_seconds,
        max_entry_delay_minutes=futures.max_entry_delay_minutes,
    )
    return runner, lock


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
