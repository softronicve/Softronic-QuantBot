"""Daily basket report: what the cycle saw, what it decided and with which numbers.

At the end of each cycle three files are written to the reports folder:

* ``daily_audit_YYYY-MM-DD.json``: the full detail, for machines. It is also
  what the dashboard reads (``src/ui/dashboard.py``).
* ``daily_audit_YYYY-MM-DD.md``: the same in Markdown, meant to be pasted into
  an LLM to ask it for an audit. It ends with ready-made questions.
* ``market_cache.json``: the latest daily candles and their channels, for charts.

The report is built from the same data the cycle used (candles, balance,
orders and results), not from a second query to the exchange: what counts
is what was actually decided.

The LLM questions carry a warning that should stay: the strategy only has
historical backing, so an audit can point out execution or data faults, but
cannot on its own justify a parameter change.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.atomic_file import write_atomically
from src.engine.orchestrator import CLOSE_LONG, OPEN_LONG, OrderIntent, entry_priority
from src.execution.binance_futures_executor import (
    DONE,
    SLIPPAGE_WARNING_PCT,
    FuturesAccount,
    FuturesPosition,
    MarketRules,
)
from src.risk.portfolio_risk import MAX_HEAT_EXCEEDED, PortfolioRiskManager
from src.risk.turtle_engine import RiskError, wilder_atr
from src.strategies.trend_breakout import Action, StrategyError, TrendBreakoutStrategy

SCHEMA = 1
REPORT_PREFIX = "daily_audit_"
MARKET_CACHE = "market_cache.json"

# State of a symbol on the basket radar.
IN_POSITION = "IN_POSITION"
CANDIDATE = "CANDIDATE"
WAITING = "WAITING"
NO_DATA = "NO_DATA"

# A day whose true range reaches this many times the ATR is flagged as anomalous.
# It is a descriptive threshold for the report: no bot rule depends on it.
VOLATILITY_ALERT = 2.0
# Mean correlation between positions from which the report highlights it.
CORRELATION_ALERT = 0.7
# A "down day" for the tail correlation: the basket's average daily return below this.
TAIL_DAY_RETURN = -0.01
# Fewer down days than this in the window and the tail correlation is not reported: it would be noise.
TAIL_MIN_DAYS = 10
# Breakout strength (in ATR) from which a coin counts as close to breaking out.
PRE_ALERT_STRENGTH = -1.5
DAY_MS = 86_400_000
# Closed trades and heat-limited breakouts followed after the fact.
SETTLEMENT_TRADES = 20
# After an exit the price is also read a fixed number of daily closes later (TradingAgents-style
# settlement), so trades closed on different days compare on the same horizon.
SETTLEMENT_HORIZONS = (5, 10, 20)
# Market turbulence (FinRL): Mahalanobis distance of today's returns from the last year of returns.
TURBULENCE_LOOKBACK = 252
# Edge ratio of the entry signal over the candles in the report, at these horizons in days.
EDGE_HORIZONS = (5, 10, 20)
# Below this many closed trades the trade statistics describe, they do not test anything.
MEANINGFUL_TRADES = 30
HEAT_DISCARD_REPORTS = 30

_DECISIONS = {OPEN_LONG: Action.ENTER_LONG.value, CLOSE_LONG: Action.EXIT_LONG.value}


@dataclass(frozen=True, slots=True)
class CycleSnapshot:
    """What the daily cycle hands to the report: its input data and what it did with it."""

    cycle_date: str
    mode: str
    status: str
    late_minutes: float
    # Balance the orchestrator sized the orders with.
    balance: float
    # The account and the positions as they were left after execution.
    account: FuturesAccount
    positions: Sequence[FuturesPosition]
    market: Mapping[str, pd.DataFrame]
    rules: Mapping[str, MarketRules]
    intents: Sequence[OrderIntent]
    # Result of each order sent or discarded, as the cycle returns them.
    results: Sequence[Mapping[str, Any]]
    symbols: Sequence[str]
    # Positions whose stop was hit since the previous cycle.
    stopped: Sequence[str] = ()
    # When each held position was opened (ms); a symbol missing here gets no excursion figures.
    opened_at: Mapping[str, int] | None = None
    # Closed trades of the account, oldest first, as the paper account keeps them.
    closed_trades: Sequence[Mapping[str, Any]] = ()
    # End-of-cycle check that every open position has its stop (``DailyRunner._verify_stops``).
    stop_check: Mapping[str, Any] | None = None


class DailyReporter:
    """Builds and saves the report of each cycle."""

    def __init__(
        self,
        directory: Path,
        strategy: TrendBreakoutStrategy,
        risk_manager: PortfolioRiskManager,
        *,
        chart_candles: int = 180,
        correlation_days: int = 60,
    ) -> None:
        """
        Args:
            directory: reports folder; it is created when the first one is written.
            strategy: the cycle's rule; the periods and the risk come from it.
            risk_manager: the portfolio limits shown next to the state.
            chart_candles: candles per symbol saved for the chart.
            correlation_days: days of returns used to measure the
                correlation between coins.
        """
        self._directory = directory
        self._strategy = strategy
        self._risk = risk_manager
        self._chart_candles = max(2, chart_candles)
        self._correlation_days = max(2, correlation_days)

    @property
    def directory(self) -> Path:
        return self._directory

    def write(self, cycle: CycleSnapshot) -> tuple[Path, Path]:
        """Save the cycle report as Markdown and JSON, and the candle cache.

        Returns:
            The paths of the Markdown and the JSON.
        """
        report = self.build(cycle)
        markdown_path, json_path = report_paths(self._directory, cycle.cycle_date)
        _write(json_path, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        _write(markdown_path, render_markdown(report))
        _write(self._directory / MARKET_CACHE, json.dumps(self._market_cache(cycle), allow_nan=False))
        logger.info("Cycle {} report saved to {}", cycle.cycle_date, markdown_path)
        return markdown_path, json_path

    # --- Building ---------------------------------------------------------

    def build(self, cycle: CycleSnapshot) -> dict[str, Any]:
        """The report as a dictionary, ready to be saved as JSON."""
        strategy, risk = self._strategy, self._risk
        positions = [_position(position, cycle.balance) for position in cycle.positions]
        held = {position["symbol"] for position in positions}
        intents = {intent.symbol: intent for intent in cycle.intents}
        results = {result["symbol"]: dict(result) for result in cycle.results}
        heat = sum(position["risk_usd"] for position in positions)
        nominal = sum(position["nominal_usd"] for position in positions)
        account = cycle.account
        # Open PnL is the sum over the positions that are listed: the figure the account gives
        # comes from another price read and can differ from that table by cents.
        unrealized = sum(position["unrealized_pnl"] for position in positions)
        equity = account.wallet_balance + unrealized

        basket = [
            self._row(symbol, cycle, intents.get(symbol), results.get(symbol), symbol in held)
            for symbol in cycle.symbols
        ]
        rows = {row["symbol"]: row for row in basket}
        for position in positions:
            position.update(_exit_levels(position, rows.get(position["symbol"])))
            position.update(_excursions(position, cycle.market.get(position["symbol"]),
                                        (cycle.opened_at or {}).get(position["symbol"])))
        observations = {
            "volatility": [row["volatility"] for row in basket if row.get("volatility")],
            "correlation": self._correlation(cycle.market, sorted(held)),
            "execution": _execution(basket, cycle),
            "regime": _regime(basket),
            "turbulence": _turbulence(cycle.market),
            "edge_ratio": _edge_ratio(cycle.market, self._strategy),
            "trade_quality": trade_quality(cycle.closed_trades),
            "benchmarks": self._benchmarks(cycle, equity),
            "heat_discards": self._heat_discards(cycle),
        }
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "cycle_date": cycle.cycle_date,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": cycle.mode,
            "status": cycle.status,
            "late_minutes": round(cycle.late_minutes, 1),
            "parameters": {
                "entry_period": strategy.entry_period,
                "exit_period": strategy.exit_period,
                "atr_period": strategy.atr_period,
                "stop_atr_multiple": strategy.stop_atr_multiple,
                "risk_pct": strategy.risk_pct,
                "max_position_leverage": strategy.max_leverage,
                "max_portfolio_heat_pct": risk.max_portfolio_heat_pct,
                "max_portfolio_leverage": risk.max_portfolio_leverage,
            },
            "account": {
                "sizing_balance": cycle.balance,
                "wallet_balance": account.wallet_balance,
                "unrealized_pnl": unrealized,
                "equity": equity,
                "available_balance": account.available_balance,
                "margin_used": max(equity - account.available_balance, 0.0),
                "heat_usd": heat,
                "heat_pct": _ratio(heat, account.wallet_balance) * 100.0,
                "heat_limit_pct": risk.max_portfolio_heat_pct,
                "nominal_usd": nominal,
                "leverage": _ratio(nominal, account.wallet_balance),
                "leverage_limit": risk.max_portfolio_leverage,
                "open_positions": len(positions),
            },
            "positions": positions,
            "stopped": list(cycle.stopped),
            "closed_trades": _settlement(cycle.closed_trades, cycle.market),
            "stop_check": dict(cycle.stop_check) if cycle.stop_check is not None else None,
            "basket": basket,
            "observations": observations,
        }
        report["llm_prompts"] = _prompts(report)
        return _clean(report)

    def _row(
        self, symbol: str, cycle: CycleSnapshot, intent: OrderIntent | None,
        result: dict[str, Any] | None, held: bool,
    ) -> dict[str, Any]:
        """A symbol's row in the decision matrix, with its sizing."""
        row: dict[str, Any] = {
            "symbol": symbol, "state": IN_POSITION if held else NO_DATA, "decision": Action.HOLD.value,
            "reason": intent.reason if intent else "NO_DATA", "execution": _brief(result),
        }
        if intent is not None:
            row["decision"] = _DECISIONS.get(intent.action, Action.HOLD.value)
        candles, rules = cycle.market.get(symbol), cycle.rules.get(symbol)
        if candles is None:
            return row
        try:
            levels = self._strategy.levels(candles)
        except (StrategyError, RiskError) as error:
            row["reason"] = row["reason"] if intent else f"INVALID_DATA: {error}"
            return row
        breakout = levels.close > levels.entry_channel
        row.update(
            state=IN_POSITION if held else CANDIDATE if breakout else WAITING,
            close=levels.close,
            high_entry=levels.entry_channel,
            low_exit=levels.exit_channel,
            atr=levels.atr,
            breakout_strength=levels.breakout_strength,
            proximity_pct=(levels.close / levels.entry_channel - 1.0) * 100.0,
            volatility=_volatility(symbol, candles, levels.atr),
        )
        if rules is not None:
            row["sizing"] = self._sizing(levels.close, levels.atr, cycle.balance, rules, intent)
        return row

    def _sizing(
        self, close: float, atr: float, balance: float, rules: MarketRules, intent: OrderIntent | None
    ) -> dict[str, Any]:
        """The size an entry in the symbol would have today, step by step.

        Repeats the ``ATRSizer`` math so each step can be shown. A test
        checks that the result is the same as the one from ``ATRSizer``.
        """
        strategy = self._strategy
        stop_distance = atr * strategy.stop_atr_multiple
        risk_budget = balance * strategy.risk_pct / 100.0
        lots_by_risk = _ratio(risk_budget, stop_distance)
        lots_by_cap = _ratio(balance * strategy.max_leverage, close)
        raw = min(lots_by_risk, lots_by_cap)
        limited_by = "position cap" if lots_by_cap < lots_by_risk else "risk"
        if rules.max_qty is not None and raw > float(rules.max_qty):
            raw, limited_by = float(rules.max_qty), "exchange max quantity"
        final = float(rules.round_quantity(raw)) if math.isfinite(raw) and raw > 0.0 else 0.0
        if final < float(rules.min_qty):
            final = 0.0
        nominal = final * close
        stop_price = close - stop_distance
        sizing = {
            "balance": balance,
            "risk_pct": strategy.risk_pct,
            "risk_budget_usd": risk_budget,
            "stop_atr_multiple": strategy.stop_atr_multiple,
            "stop_distance": stop_distance,
            "max_position_leverage": strategy.max_leverage,
            "lots_by_risk": lots_by_risk,
            "lots_by_leverage": lots_by_cap,
            "limited_by": limited_by,
            "lots_raw": raw,
            "step_size": float(rules.step_size),
            "min_qty": float(rules.min_qty),
            "lots_final": final,
            "nominal_usd": nominal,
            "min_notional_usd": float(rules.min_notional),
            "meets_min_notional": final > 0.0 and nominal >= float(rules.min_notional),
            "risk_at_stop_usd": final * stop_distance,
            "risk_at_stop_pct": _ratio(final * stop_distance, balance) * 100.0,
            "stop_price": stop_price,
            "tick_size": float(rules.tick_size),
            "stop_price_rounded": float(rules.round_price(stop_price)) if stop_price > 0.0 else None,
        }
        if intent is not None and intent.action == OPEN_LONG:
            sizing["order_volume"] = intent.volume
            sizing["matches_order"] = math.isclose(intent.volume, final, rel_tol=1e-9, abs_tol=1e-12)
        return sizing

    def _correlation(self, market: Mapping[str, pd.DataFrame], held: list[str]) -> dict[str, Any]:
        """Correlation of recent daily returns, across the basket and among what is open."""
        returns = pd.DataFrame({
            symbol: np.log(candles["close"].astype(float)).diff().iloc[-self._correlation_days :].reset_index(drop=True)
            for symbol, candles in market.items() if len(candles) > self._correlation_days
        })
        summary: dict[str, Any] = {"days": self._correlation_days, "symbols": len(returns.columns),
                                   "basket_mean": None, "held_mean": None, "held_pairs": [],
                                   "tail_threshold_pct": TAIL_DAY_RETURN * 100.0, "tail_days": 0,
                                   "tail_mean": None, "tail_held_mean": None, "tail_down_share_pct": None}
        if len(returns.columns) < 2:
            return summary
        matrix = returns.corr()
        summary["basket_mean"] = _mean_pairwise(matrix, list(matrix.columns))
        inside = [symbol for symbol in held if symbol in matrix.columns]
        # Tail correlation: the same measure, only over the days the basket as a whole fell. Diversification
        # that holds on quiet days and vanishes on the bad ones is the case it is meant to catch.
        down = returns[np.expm1(returns).mean(axis=1) < TAIL_DAY_RETURN]
        summary["tail_days"] = len(down)
        if len(down) >= TAIL_MIN_DAYS:
            # Restricting to down days narrows the range of returns, which by itself pulls a correlation
            # down; the share of coins that fell on those days says the same thing without that bias.
            summary["tail_down_share_pct"] = float((down < 0.0).mean(axis=1).mean() * 100.0)
            tail = down.corr()
            summary["tail_mean"] = _mean_pairwise(tail, list(tail.columns))
            if len(inside) >= 2:
                summary["tail_held_mean"] = _mean_pairwise(tail, inside)
        if len(inside) >= 2:
            summary["held_mean"] = _mean_pairwise(matrix, inside)
            pairs = [
                {"pair": f"{a}–{b}", "correlation": float(matrix.loc[a, b])}
                for i, a in enumerate(inside) for b in inside[i + 1 :]
            ]
            summary["held_pairs"] = sorted(pairs, key=lambda pair: -pair["correlation"])[:5]
        return summary

    def _earlier_reports(self, cycle_date: str) -> list[dict[str, Any]]:
        """Reports of earlier cycles in this folder, oldest first. Unreadable ones are skipped."""
        reports = []
        for path in sorted(self._directory.glob(f"{REPORT_PREFIX}*.json")):
            if path.stem.removeprefix(REPORT_PREFIX) >= cycle_date:
                continue
            try:
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as error:
                logger.warning("Unreadable earlier report, skipped: {} ({})", path.name, error)
        return reports

    def _benchmarks(self, cycle: CycleSnapshot, equity: float) -> dict[str, Any] | None:
        """Return since the first report against holding BTC and against an equal-weight basket, both passive."""
        earlier = self._earlier_reports(cycle.cycle_date)
        if not earlier:
            return None
        first = earlier[0]
        start_equity = (first.get("account") or {}).get("equity")
        # The first cycle traded on the candle that closed the day before it.
        start_open = _day_ms(first["cycle_date"]) - DAY_MS
        changes = {symbol: change for symbol, candles in cycle.market.items()
                   if (change := _change_since(candles, start_open)) is not None}
        return {
            "since": first["cycle_date"],
            "bot_pct": (equity / start_equity - 1.0) * 100.0 if start_equity else None,
            "btc_pct": changes.get("BTCUSDT"),
            "basket_pct": sum(changes.values()) / len(changes) if changes else None,
            "basket_symbols": len(changes),
        }

    def _heat_discards(self, cycle: CycleSnapshot) -> list[dict[str, Any]]:
        """Breakouts of earlier cycles left out by the open-risk limit, and where their price is today."""
        discards = []
        for report in self._earlier_reports(cycle.cycle_date)[-HEAT_DISCARD_REPORTS:]:
            for row in report.get("basket", []):
                if row.get("reason") != MAX_HEAT_EXCEEDED or not row.get("close"):
                    continue
                today = _last_close(cycle.market.get(row["symbol"]))
                discards.append({
                    "cycle_date": report["cycle_date"], "symbol": row["symbol"], "close": row["close"],
                    "breakout_strength": row.get("breakout_strength"), "close_today": today,
                    "change_pct": (today / row["close"] - 1.0) * 100.0 if today else None,
                })
        return discards

    def _market_cache(self, cycle: CycleSnapshot) -> dict[str, Any]:
        """The latest candles of each symbol with both channels, for the dashboard chart."""
        entry, exit_ = self._strategy.entry_period, self._strategy.exit_period
        symbols = {}
        for symbol, candles in cycle.market.items():
            high, low = candles["high"].astype(float), candles["low"].astype(float)
            # Each candle is compared with the channel of the previous ones, not counting itself.
            upper = high.rolling(entry).max().shift(1)
            lower = low.rolling(exit_).min().shift(1)
            tail = slice(-self._chart_candles, None)
            symbols[symbol] = {
                "time": [int(value) // 1000 for value in candles["open_time"].iloc[tail]],
                **{name: [float(value) for value in candles[name].iloc[tail]]
                   for name in ("open", "high", "low", "close")},
                "entry_channel": [None if math.isnan(value) else float(value) for value in upper.iloc[tail]],
                "exit_channel": [None if math.isnan(value) else float(value) for value in lower.iloc[tail]],
            }
        return {"cycle_date": cycle.cycle_date, "mode": cycle.mode, "symbols": symbols}


def report_paths(directory: Path, cycle_date: str) -> tuple[Path, Path]:
    """Paths of the Markdown and the JSON report for a date."""
    stem = f"{REPORT_PREFIX}{cycle_date}"
    return directory / f"{stem}.md", directory / f"{stem}.json"


# --- Report pieces ------------------------------------------------------------


def _position(position: FuturesPosition, balance: float) -> dict[str, Any]:
    contracts = abs(position.quantity)
    nominal = contracts * position.mark_price
    if position.quantity > 0.0 and position.stop_price is not None:
        risk = contracts * max(position.entry_price - position.stop_price, 0.0)
    else:
        # With no known stop (or in a short), the whole notional counts as risk.
        risk = nominal
    return {
        "symbol": position.symbol,
        "quantity": position.quantity,
        "entry_price": position.entry_price,
        "mark_price": position.mark_price,
        "stop_price": position.stop_price,
        "nominal_usd": nominal,
        "risk_usd": risk,
        "risk_pct": _ratio(risk, balance) * 100.0,
        "unrealized_pnl": position.quantity * (position.mark_price - position.entry_price),
    }


def _exit_levels(position: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    """Both exits of a long, and which one would sell it first.

    The stop sells as soon as the price touches it; the channel sells when a
    daily candle closes below it. Whichever is higher is reached first.
    """
    stop, channel = position["stop_price"], (row or {}).get("low_exit")
    mark = position["mark_price"]
    levels = [(price, kind) for price, kind in ((stop, "stop"), (channel, "canal")) if price is not None]
    nearest = max(levels) if levels else None
    risk = position["risk_usd"]
    return {
        "exit_channel": channel,
        "effective_exit": nearest[1] if nearest else None,
        "effective_exit_price": nearest[0] if nearest else None,
        "effective_exit_distance_pct": (nearest[0] / mark - 1.0) * 100.0 if nearest and mark > 0.0 else None,
        # Open result in units of the risk taken at entry (to the initial stop).
        "r_multiple": position["unrealized_pnl"] / risk if stop is not None and risk > 0.0 else None,
    }


def _excursions(position: dict[str, Any], candles: pd.DataFrame | None, opened_at: int | None) -> dict[str, Any]:
    """Maximum favourable and adverse excursion since the entry, from the closed daily candles.

    The candle of the entry day counts whole: the cycle buys a minute after it
    opens. A position bought in this same cycle has no closed candle yet.
    """
    empty = {"mfe_price": None, "mfe_usd": None, "mfe_pct": None, "mfe_r": None,
             "mae_price": None, "mae_usd": None, "mae_pct": None, "mae_r": None}
    if candles is None or opened_at is None or position["quantity"] <= 0.0:
        return empty
    since = candles[candles["open_time"].astype("int64") >= opened_at - opened_at % DAY_MS]
    if since.empty:
        return empty
    entry, quantity, risk = position["entry_price"], position["quantity"], position["risk_usd"]
    high, low = float(since["high"].astype(float).max()), float(since["low"].astype(float).min())
    favourable, adverse = quantity * max(high - entry, 0.0), quantity * min(low - entry, 0.0)
    return {
        "mfe_price": high, "mfe_usd": favourable, "mfe_pct": (high / entry - 1.0) * 100.0,
        "mfe_r": favourable / risk if risk > 0.0 else None,
        "mae_price": low, "mae_usd": adverse, "mae_pct": (low / entry - 1.0) * 100.0,
        "mae_r": adverse / risk if risk > 0.0 else None,
    }


def _settlement(trades: Sequence[Mapping[str, Any]], market: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Closed trades, most recent first, with what their price did after the exit.

    ``protected_usd`` is what the exit saved (positive) or gave up (negative)
    against still holding the same quantity at today's close.
    """
    settled = []
    for trade in list(trades)[-SETTLEMENT_TRADES:][::-1]:
        candles = market.get(str(trade.get("symbol")))
        today = _last_close(candles)
        exit_price, quantity = trade.get("exit_price"), trade.get("quantity")
        settled.append({
            "symbol": trade.get("symbol"), "reason": trade.get("reason"),
            "entry_price": trade.get("entry_price"), "exit_price": exit_price, "quantity": quantity,
            "pnl": trade.get("pnl"), "opened_at": trade.get("opened_at"), "closed_at": trade.get("closed_at"),
            "close_today": today,
            "after_exit_pct": (today / exit_price - 1.0) * 100.0 if today and exit_price else None,
            "protected_usd": quantity * (exit_price - today) if today and exit_price and quantity else None,
            **_trade_excursions(trade, candles),
            "after_exit": _after_exit(trade, candles),
        })
    return settled


def _trade_risk(trade: Mapping[str, Any]) -> float | None:
    """Initial risk of a closed long: quantity × (entry − stop). None if the stop is missing or not below entry."""
    entry, stop, quantity = trade.get("entry_price"), trade.get("stop_price"), trade.get("quantity")
    if not all(isinstance(value, (int, float)) for value in (entry, stop, quantity)) or not entry > stop:
        return None
    return quantity * (entry - stop)


def _trade_excursions(trade: Mapping[str, Any], candles: pd.DataFrame | None) -> dict[str, Any]:
    """Best and worst price a closed trade saw, in R, and how much of its best it gave back.

    The candles from the entry day up to the day before the exit count whole; the exit day only up to
    the exit price, since it is not known whether its high came before or after the fill.
    """
    risk = _trade_risk(trade)
    empty = {"result_r": None, "mfe_r": None, "mae_r": None, "giveback_r": None}
    entry, exit_price, quantity = trade.get("entry_price"), trade.get("exit_price"), trade.get("quantity")
    if risk is None or not isinstance(exit_price, (int, float)):
        return empty
    highs, lows = [exit_price], [exit_price]
    opened, closed = trade.get("opened_at"), trade.get("closed_at")
    if candles is not None and isinstance(opened, int) and isinstance(closed, int):
        times = candles["open_time"].astype("int64")
        inside = candles[(times >= opened - opened % DAY_MS) & (times < closed - closed % DAY_MS)]
        highs += [float(value) for value in inside["high"]]
        lows += [float(value) for value in inside["low"]]
    result = quantity * (exit_price - entry) / risk
    mfe = quantity * max(max(highs) - entry, 0.0) / risk
    return {"result_r": result, "mfe_r": mfe, "mae_r": quantity * min(min(lows) - entry, 0.0) / risk,
            "giveback_r": mfe - result}


def _after_exit(trade: Mapping[str, Any], candles: pd.DataFrame | None) -> dict[str, float | None]:
    """Change from the exit price to the close a fixed number of days after the exit day; None while pending."""
    exit_price, closed = trade.get("exit_price"), trade.get("closed_at")
    changes: dict[str, float | None] = {str(days): None for days in SETTLEMENT_HORIZONS}
    if candles is None or not isinstance(exit_price, (int, float)) or not isinstance(closed, int) or not exit_price:
        return changes
    times = list(candles["open_time"].astype("int64"))
    exit_day = closed - closed % DAY_MS
    if exit_day not in times:
        return changes
    start = times.index(exit_day)
    for days in SETTLEMENT_HORIZONS:
        if start + days < len(candles):
            changes[str(days)] = (float(candles["close"].iloc[start + days]) / exit_price - 1.0) * 100.0
    return changes


def trade_quality(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Expectancy and SQN of the closed trades, in R. SQN needs at least ``MEANINGFUL_TRADES`` to be shown."""
    results = []
    for trade in trades:
        risk = _trade_risk(trade)
        if risk is not None and isinstance(trade.get("pnl"), (int, float)):
            results.append(float(trade["pnl"]) / risk)
    n = len(results)
    mean = sum(results) / n if n else None
    std = float(np.std(results, ddof=1)) if n >= 2 else None
    return {
        "trades": n, "expectancy_r": mean, "best_r": max(results) if results else None,
        "worst_r": min(results) if results else None,
        # sqrt(n) · mean(R) / std(R) (Van Tharp; vectorbt uses ddof=1). Below 30 trades it is noise.
        "sqn": (n ** 0.5) * mean / std if n >= MEANINGFUL_TRADES and std else None,
        "meaningful": n >= MEANINGFUL_TRADES,
    }


def _turbulence(market: Mapping[str, pd.DataFrame]) -> dict[str, Any] | None:
    """FinRL's turbulence index on the basket, informational only.

    Mahalanobis distance of today's daily log returns from the mean of the
    previous ``TURBULENCE_LOOKBACK`` days, with the pseudo-inverse of their
    covariance. Coins without a full window are left out. With p coins the raw
    distance grows with p, so it is reported divided by p and as a percentile
    of its own history over the candles in the report.
    """
    closes = pd.DataFrame({symbol: candles.set_index(candles["open_time"].astype("int64"))["close"].astype(float)
                           for symbol, candles in market.items()}).sort_index()
    returns = np.log(closes).diff().iloc[1:]
    lookback = TURBULENCE_LOOKBACK
    values = []
    for end in range(lookback, len(returns)):
        window = returns.iloc[end - lookback:end].dropna(axis=1)
        today = returns.iloc[end][window.columns]
        window, today = window.loc[:, today.notna()], today.dropna()
        coins = len(window.columns)
        if coins < 2 or coins >= lookback - 2:
            continue
        gap = (today - window.mean()).to_numpy()
        distance = float(gap @ np.linalg.pinv(window.cov().to_numpy()) @ gap)
        values.append((distance, coins))
    if not values:
        return None
    distance, coins = values[-1]
    history = [value for value, _ in values]
    return {
        "value": distance / coins, "raw": distance, "coins": coins, "lookback": lookback,
        "history_days": len(history),
        "percentile": sum(value <= distance for value in history) / len(history) * 100.0,
    }


def _edge_ratio(market: Mapping[str, pd.DataFrame], strategy: TrendBreakoutStrategy) -> dict[str, Any] | None:
    """Edge ratio of the entry signal over every breakout in the report's candles, against any day.

    For each day the close broke the high of the previous ``entry_period``
    candles: the best and worst move over the next n days, each divided by the
    ATR of that day; E = mean(favourable) / mean(adverse). The same over every
    day gives the baseline, close to 1 for an entry with no edge. It measures
    the signal on history, not the bot's fills.
    """
    signal = {days: ([], []) for days in EDGE_HORIZONS}
    every = {days: ([], []) for days in EDGE_HORIZONS}
    breakouts = 0
    for candles in market.values():
        high, low, close = (candles[name].astype(float).to_numpy() for name in ("high", "low", "close"))
        if len(close) <= strategy.atr_period + max(EDGE_HORIZONS):
            continue
        try:
            atr = wilder_atr(high, low, close, strategy.atr_period)
        except RiskError:
            continue
        channel = pd.Series(high).rolling(strategy.entry_period).max().shift(1).to_numpy()
        for t in range(strategy.atr_period, len(close)):
            if not atr[t] > 0.0:
                continue
            is_signal = close[t] > channel[t]
            breakouts += bool(is_signal) and t + min(EDGE_HORIZONS) < len(close)
            for days in EDGE_HORIZONS:
                if t + days >= len(close):
                    continue
                favourable = (high[t + 1:t + days + 1].max() - close[t]) / atr[t]
                adverse = (close[t] - low[t + 1:t + days + 1].min()) / atr[t]
                for target in ((signal, every) if is_signal else (every,)):
                    target[days][0].append(max(favourable, 0.0))
                    target[days][1].append(max(adverse, 0.0))
    if not breakouts:
        return None

    def ratio(pairs: tuple[list[float], list[float]]) -> float | None:
        up, down = pairs
        return float(np.mean(up) / np.mean(down)) if up and np.mean(down) > 0.0 else None

    return {"breakouts": breakouts,
            "horizons": [{"days": days, "signal": ratio(signal[days]), "any_day": ratio(every[days]),
                          "signals": len(signal[days][0])} for days in EDGE_HORIZONS]}


def _regime(basket: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of the basket is breaking out, close to it, or below its exit channel today."""
    measured = [row for row in basket if row.get("breakout_strength") is not None]
    return {
        "symbols": len(measured),
        "breaking_out": sum(row["breakout_strength"] > 0.0 for row in measured),
        "pre_alert": sum(PRE_ALERT_STRENGTH < row["breakout_strength"] <= 0.0 for row in measured),
        "pre_alert_from": PRE_ALERT_STRENGTH,
        "below_exit_channel": sum(row["close"] < row["low_exit"] for row in measured),
    }


def _last_close(candles: pd.DataFrame | None) -> float | None:
    return float(candles["close"].iloc[-1]) if candles is not None and len(candles) else None


def _change_since(candles: pd.DataFrame, open_ms: int) -> float | None:
    """Change in % from the close of the candle that opened at ``open_ms`` to the last close."""
    start = candles[candles["open_time"].astype("int64") == open_ms]
    if start.empty:
        return None
    return (float(candles["close"].iloc[-1]) / float(start["close"].iloc[0]) - 1.0) * 100.0


def _day_ms(cycle_date: str) -> int:
    return int(datetime.fromisoformat(cycle_date).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _volatility(symbol: str, candles: pd.DataFrame, atr: float) -> dict[str, Any] | None:
    if len(candles) < 2 or not atr > 0.0:
        return None
    last, previous = candles.iloc[-1], float(candles["close"].iloc[-2])
    true_range = max(float(last["high"]) - float(last["low"]), abs(float(last["high"]) - previous),
                     abs(float(last["low"]) - previous))
    ratio = true_range / atr
    return {
        "symbol": symbol,
        "return_pct": (float(last["close"]) / previous - 1.0) * 100.0,
        "true_range": true_range,
        "range_in_atr": ratio,
        "atr_pct_of_price": atr / float(last["close"]) * 100.0,
        "anomaly": ratio >= VOLATILITY_ALERT,
    }


def _execution(basket: list[dict[str, Any]], cycle: CycleSnapshot) -> list[dict[str, Any]]:
    """What happened to each order: executed (and how far from the close that gave the signal) or discarded."""
    rows = {row["symbol"]: row for row in basket}
    notes = []
    for result in cycle.results:
        note = {"symbol": result["symbol"], "action": result.get("action"), "status": result.get("status"),
                "reason": result.get("reason"), "executed": result.get("status") in DONE}
        row = rows.get(result["symbol"]) or {}
        price, close, atr = result.get("price"), row.get("close"), row.get("atr")
        if note["executed"] and price and close:
            note["price"] = price
            note["signal_close"] = close
            note["slippage_pct"] = (price / close - 1.0) * 100.0
            if result.get("action") == OPEN_LONG:
                note["slippage_atr"] = (price - close) / atr if atr else None
                note["slippage_warning"] = abs(note["slippage_pct"]) > SLIPPAGE_WARNING_PCT
                if result.get("collar_exceeded"):
                    note["collar_exceeded"] = True
        notes.append(note)
    return notes


def _brief(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {key: result[key] for key in ("action", "status", "reason", "quantity", "price", "stop_price", "pnl")
            if result.get(key) is not None}


def _mean_pairwise(matrix: pd.DataFrame, symbols: list[str]) -> float | None:
    values = [float(matrix.loc[a, b]) for i, a in enumerate(symbols) for b in symbols[i + 1 :]]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else None


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _clean(value: Any) -> Any:
    """A copy that fits strict JSON: no NaN or infinities, no numpy types."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _write(path: Path, text: str) -> None:
    """Atomic write: a reader sees the old file or the new one, never a half-written one."""
    # The .md and the .json of a day share a stem: the temporary name keeps the whole name.
    write_atomically(path, text, path.with_name(path.name + ".tmp"))


# --- Questions for the LLM ---------------------------------------------------

_GUARDRAIL = (
    "Context you must respect: this is a channel breakout strategy (entry {entry_period} days, exit "
    "{exit_period}, ATR of {atr_period}, stop at {stop_atr_multiple:g} ATR), long only, on daily candles. "
    "Its backing is historical and not yet confirmed going forward, and hundreds of variants have already been "
    "tested on the same data. Do not propose changing parameters based on a single day: point out data, "
    "calculation or execution errors, and keep what you see in these numbers apart from what you assume."
)


def _prompts(report: dict[str, Any]) -> list[dict[str, str]]:
    """Ready-made questions, with the day's data inside, to paste into an LLM."""
    parameters, account = report["parameters"], report["account"]
    guardrail = _GUARDRAIL.format(**parameters)
    header = (
        f"Cycle {report['cycle_date']} in {report['mode']} mode. Balance {account['wallet_balance']:.2f} USDT, "
        f"{account['open_positions']} positions, open risk {account['heat_pct']:.2f} % of a max of "
        f"{account['heat_limit_pct']:g} %, notional {account['leverage']:.2f}x of a max of "
        f"{account['leverage_limit']:g}x."
    )

    volatility = report["observations"]["volatility"]
    anomalies = [item for item in volatility if item["anomaly"]]
    lines = [
        f"- {item['symbol']}: return {item['return_pct']:+.2f} %, range of {item['range_in_atr']:.2f} ATR, "
        f"ATR = {item['atr_pct_of_price']:.2f} % of the price"
        for item in sorted(volatility, key=lambda item: -item["range_in_atr"])
    ]
    found = (
        f"{len(anomalies)} symbol{'' if len(anomalies) == 1 else 's'} had a range of {VOLATILITY_ALERT:g} ATR or more: "
        + ", ".join(item["symbol"] for item in anomalies) + "."
        if anomalies else f"No symbol had a range of {VOLATILITY_ALERT:g} ATR or more."
    )
    prompts = [{
        "title": "Volatility anomalies",
        "prompt": "\n".join([
            header, guardrail, "", "Last daily candle of each symbol:", *lines, "", found, "",
            "1. Does any move look like a data error (inconsistent candle, gap, off-market price) "
            "rather than a real move? Say what you would check.",
            "2. For the open positions, is the 2 ATR stop still far from the noise of a day like this one?",
            "3. Is there any symbol whose ATR is so large relative to the price that the resulting size "
            "falls below the exchange minimums?",
        ]),
    }]

    correlation = report["observations"]["correlation"]
    held = [position["symbol"] for position in report["positions"]]
    pairs = [f"- {pair['pair']}: {pair['correlation']:+.2f}" for pair in correlation["held_pairs"]]
    facts = [
        f"Mean correlation of the basket's daily returns ({correlation['symbols']} symbols, "
        f"{correlation['days']} days): {_text(correlation['basket_mean'])}.",
        f"Open positions: {', '.join(held) if held else 'none'}.",
        f"Mean correlation between them: {_text(correlation['held_mean'])}.",
        _tail_text(correlation),
        *([f"Market turbulence (FinRL): percentile {turbulence['percentile']:.0f} of the last "
           f"{turbulence['history_days']} days."] if (turbulence := report["observations"].get("turbulence")) else []),
    ]
    if correlation["held_mean"] is not None and correlation["held_mean"] >= CORRELATION_ALERT:
        facts.append(f"Above {CORRELATION_ALERT:g}: the positions move almost as one.")
    prompts.append({
        "title": "Correlation and concentration",
        "prompt": "\n".join([
            header, guardrail, "", *facts, *(["Most correlated pairs among the open ones:", *pairs] if pairs else []),
            "",
            "1. If every position fell to its stop at the same time, the loss would be the open risk. "
            "Is that number consistent with the measured correlation?",
            "2. The portfolio control does not limit the number of correlated coins in crypto, only total risk and "
            "notional. Given these data, what concrete scenario would worry you and how would you detect it?",
            "3. Compare the correlation on down days with the one over all days: how much diversification is lost "
            "when it is needed most, and what other figure would you ask for to judge concentration?",
        ]),
    })

    execution = report["observations"]["execution"]
    if execution:
        orders = [
            f"- {item['symbol']} {item['action']}: {item['status']}"
            + (f", at {item['price']:g} against a close of {item['signal_close']:g} "
               f"({item['slippage_pct']:+.3f} %)" if "slippage_pct" in item else "")
            + (f", reason: {item['reason']}" if item.get("reason") and not item["executed"] else "")
            for item in execution
        ]
    else:
        orders = ["- No order was sent or discarded today."]
    rejected = [row for row in report["basket"] if row["state"] == CANDIDATE and row["decision"] == Action.HOLD.value]
    prompts.append({
        "title": "Execution quality",
        "prompt": "\n".join([
            header, guardrail, "",
            f"The cycle ended as {report['status']}, {report['late_minutes']:g} minutes late.",
            "Orders of the day:", *orders,
            "Breakouts not traded: "
            + (", ".join(f"{row['symbol']} ({row['reason']})" for row in rejected) if rejected else "none") + ".",
            *([f"Stops hit since the previous cycle: {', '.join(report['stopped'])}."] if report["stopped"] else []),
            "",
            "1. Is the difference between the fill price and the close that gave the signal reasonable for a market "
            "order at 00:01:30 UTC? Above what difference would you look into it?",
            "2. Of the discarded orders, does any point to a fault (exchange minimums, margin, data) rather than "
            "to a risk limit that worked as it should?",
            "3. What automatic check would you add to the cycle to catch a problem like today's earlier?",
        ]),
    })
    return prompts


# --- Markdown -----------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    """The report in Markdown, built from the dictionary saved as JSON."""
    parameters, account = report["parameters"], report["account"]
    atr_label = f"ATR({parameters['atr_period']})"
    lines = [
        f"# Daily basket audit: {report['cycle_date']}",
        "",
        f"Mode **{report['mode']}** · cycle **{report['status']}** · generated {report['generated_at']} · "
        f"delay {report['late_minutes']:g} min",
        "",
    ]
    if report["mode"] == "DRY_RUN":
        lines += ["> Paper trading: real prices and a paper account. No order was sent to the exchange.", ""]

    lines += [
        "## 1. Executive summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Balance (realized balance) | {_usd(account['wallet_balance'])} |",
        f"| Equity (balance + open PnL) | {_usd(account['equity'])} |",
        f"| Unrealized PnL | {_usd(account['unrealized_pnl'], signed=True)} |",
        f"| Margin used | {_usd(account['margin_used'])} |",
        f"| Free collateral | {_usd(account['available_balance'])} |",
        f"| Portfolio heat (risk down to the stops) | {account['heat_pct']:.2f} % of {account['heat_limit_pct']:g} % "
        f"max ({_usd(account['heat_usd'])}) |",
        f"| Total notional leverage | {account['leverage']:.3f}x of {account['leverage_limit']:g}x max "
        f"({_usd(account['nominal_usd'])}) |",
        f"| Open positions | {account['open_positions']} |",
        f"| Stops checked / reconciled | {stop_check_text(report.get('stop_check'))} |",
        "",
    ]
    if report["positions"]:
        lines += [
            "### Open positions",
            "",
            "| Symbol | Quantity | Entry | Mark | Stop | Notional | Risk | Risk % | Open PnL |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {p['symbol']} | {p['quantity']:g} | {_price(p['entry_price'])} | {_price(p['mark_price'])} | "
            f"{_price(p['stop_price'])} | {_usd(p['nominal_usd'])} | {_usd(p['risk_usd'])} | {p['risk_pct']:.2f} % | "
            f"{_usd(p['unrealized_pnl'], signed=True)} |"
            for p in report["positions"]
        ]
        lines.append("")
        lines += _exit_lines(report["positions"], parameters)
    if report["stopped"]:
        lines += [f"Stops hit since the previous cycle: {', '.join(report['stopped'])}.", ""]
    lines += _settlement_lines(report.get("closed_trades", []))

    lines += [
        "## 2. Basket decision matrix",
        "",
        f"Entry: close above the high of the previous {parameters['entry_period']} candles. "
        f"Exit: close below the low of the previous {parameters['exit_period']}. "
        f"Breakout strength = (close − {parameters['entry_period']}-candle high) / {atr_label}.",
        "",
        f"| Symbol | Close | High {parameters['entry_period']} | Low {parameters['exit_period']} | {atr_label} | "
        "Strength | State | Decision | Reason | Execution |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in report["basket"]:
        execution = row.get("execution") or {}
        done = execution.get("status", "—")
        if "close" not in row:
            lines.append(f"| {row['symbol']} | — | — | — | — | — | {row['state']} | {row['decision']} | "
                         f"{row['reason']} | {done} |")
            continue
        lines.append(
            f"| {row['symbol']} | {_price(row['close'])} | {_price(row['high_entry'])} | {_price(row['low_exit'])} | "
            f"{_price(row['atr'])} | {row['breakout_strength']:+.2f} | {row['state']} | {row['decision']} | "
            f"{row['reason']} | {done} |"
        )
    lines.append("")

    lines += ["## 3. Order math, step by step", ""]
    # In the order the cycle processed them: exits first, which free up risk, then the breakouts
    # from strongest to weakest, which is how the risk is shared among them.
    proposed = sorted(
        (row for row in report["basket"]
         if row.get("sizing") and (row["decision"] == Action.ENTER_LONG.value or row["state"] == CANDIDATE)),
        key=lambda row: entry_priority(row["breakout_strength"], _ratio(row["atr"], row["close"]), row["symbol"]),
    )
    closes = [row for row in report["basket"] if row["decision"] == Action.EXIT_LONG.value]
    if not proposed and not closes:
        lines += ["No breakout and no exit today: no order was proposed.", ""]
    elif proposed:
        lines += ["Listed in the order the cycle processed them: closes first and then entries, "
                  "from the strongest breakout to the weakest.", ""]
    for row in closes:
        execution = row.get("execution") or {}
        lines += [
            f"### {row['symbol']}: position closed ({row['reason']})",
            "",
            f"- The close {_price(row.get('close'))} ended below the {parameters['exit_period']}-candle low "
            f"({_price(row.get('low_exit'))}).",
            f"- Market order with `reduceOnly` for the whole position. Result: {execution.get('status', '—')}"
            + (f", {execution['quantity']:g} units at {_price(execution.get('price'))}" if execution.get("quantity") else "")
            + (f", PnL {_usd(execution['pnl'], signed=True)}" if execution.get("pnl") is not None else "") + ".",
            "",
        ]
    for rank, row in enumerate(proposed, start=1):
        lines += _sizing_lines(row, parameters, rank)

    lines += ["## 4. Observations of the day", ""]
    volatility = sorted(report["observations"]["volatility"], key=lambda item: -item["range_in_atr"])
    if volatility:
        lines += [
            f"A range of {VOLATILITY_ALERT:g} ATR or more is flagged as anomalous. It is a report threshold, "
            "not a bot rule.",
            "",
            "| Symbol | Return of the day | Range in ATR | ATR / price | Anomalous |",
            "|---|---:|---:|---:|---|",
            *[f"| {item['symbol']} | {item['return_pct']:+.2f} % | {item['range_in_atr']:.2f} | "
              f"{item['atr_pct_of_price']:.2f} % | {'yes' if item['anomaly'] else 'no'} |" for item in volatility],
            "",
        ]
    correlation = report["observations"]["correlation"]
    lines += [
        f"- Mean basket correlation ({correlation['symbols']} symbols, {correlation['days']} days): "
        f"{_text(correlation['basket_mean'])}.",
        f"- Mean correlation between the open positions: {_text(correlation['held_mean'])}.",
        f"- {_tail_text(correlation)}",
    ]
    for item in report["observations"]["execution"]:
        detail = (f"at {_price(item['price'])} against a close of {_price(item['signal_close'])} "
                  f"({item['slippage_pct']:+.3f} %)" if "slippage_pct" in item else (item.get("reason") or ""))
        if item.get("slippage_atr") is not None:
            detail += f", {item['slippage_atr']:+.3f} ATR"
        if item.get("slippage_warning"):
            detail += f" · WARNING: slippage above {SLIPPAGE_WARNING_PCT:g} %"
        if item.get("collar_exceeded"):
            detail += " · on TESTNET or LIVE this buy would not have been sent (over the slippage collar)"
        lines.append(f"- {item['symbol']} {item['action']}: {item['status']} {detail}".rstrip())
    lines.append("")
    lines += _regime_lines(report["observations"], parameters)

    lines += [
        "## 5. Questions for the LLM",
        "",
        "Each block is a complete prompt: it can be pasted as is into any LLM chat. It carries the day's "
        "data and the context the model needs so it does not draw more conclusions than the data allow.",
        "",
    ]
    for prompt in report["llm_prompts"]:
        lines += [f"### {prompt['title']}", "", "```text", prompt["prompt"], "```", ""]
    if report.get("llm_audit"):
        lines += _audit_lines(report["llm_audit"])
    return "\n".join(lines)


def _audit_lines(audit: dict[str, Any]) -> list[str]:
    """Section 6: the answers of the optional LLM auditor (``src/engine/post_cycle.py``)."""
    lines = [
        "## 6. Automatic review by the LLM auditor",
        "",
        f"Answers from {audit.get('model', 'a model')} to the blocks of section 5, written when the cycle finished. "
        "It is a model's opinion on these numbers, not a check: verify it before acting on it.",
        "",
    ]
    for answer in audit.get("answers", []):
        text = ("The model declined to answer this block." if answer.get("refused")
                else answer.get("answer") or "No answer.")
        if answer.get("truncated"):
            text += "\n\n(Answer cut off by length.)"
        lines += [f"### {answer.get('title', 'Block')}", "", text, ""]
    return lines


def stop_check_text(check: Mapping[str, Any] | None) -> str:
    """The end-of-cycle stop check in one line, for the report and the alert."""
    if check is None:
        return "not checked"
    if check.get("error"):
        return f"could not be checked ({check['error']})"
    problems = [*(f"{symbol} WITHOUT A STOP" for symbol in check["unprotected"]),
                *(f"{symbol} with a partial stop" for symbol in check["uncovered"])]
    repaired = [f"{item['symbol']} {item['status']}" for item in check.get("repaired", [])]
    text = ("OK" if not problems else "ATTENTION: " + ", ".join(problems)) + (
        f" ({check['protected']} of {check['positions']} positions protected)")
    return text + (f"; repaired: {', '.join(repaired)}" if repaired else "")


def _exit_lines(positions: list[dict[str, Any]], parameters: dict[str, Any]) -> list[str]:
    """Where each open position will be sold, and how far it has gone each way since the entry."""
    lines = [
        "#### Automatic sell levels and excursion since the entry",
        "",
        f"The stop sells as soon as the price touches it; the channel (low of {parameters['exit_period']} candles) "
        "sells when a daily candle closes below it, and rises as old lows leave it. The higher one rules. R = result "
        "in units of the initial risk (entry − stop). MFE/MAE: best and worst price since the entry, on closed daily "
        "candles.",
        "",
        f"| Symbol | Initial stop | Low {parameters['exit_period']} | Sells first | Distance | Open R | "
        "MFE | MAE |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for p in positions:
        effective = (f"{p['effective_exit']} at {_price(p['effective_exit_price'])}"
                     if p.get("effective_exit") else "—")
        lines.append(
            f"| {p['symbol']} | {_price(p['stop_price'])} | {_price(p.get('exit_channel'))} | {effective} | "
            f"{_pct(p.get('effective_exit_distance_pct'))} | {_r(p.get('r_multiple'))} | "
            f"{_excursion(p, 'mfe')} | {_excursion(p, 'mae')} |"
        )
    return [*lines, ""]


def _settlement_lines(trades: list[dict[str, Any]]) -> list[str]:
    """Closed trades and what the price did after each exit (deferred settlement)."""
    if not trades:
        return []
    lines = [
        "### Closed positions and post-exit audit",
        "",
        "Each closed position against today's close. \"Capital protected\" is what the exit saved (+) or gave up "
        "(−) against still holding the same quantity: a hindsight estimate, not a rule of the bot.",
        "",
        "| Symbol | Exit reason | Entry price | Exit price | Realized PnL | Close today | "
        "Since the exit | Capital protected (est.) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {t['symbol']} | {t.get('reason') or '—'} | {_price(t.get('entry_price'))} | {_price(t.get('exit_price'))} | "
        f"{_usd(t.get('pnl'), signed=True)} | {_price(t.get('close_today'))} | {_pct(t.get('after_exit_pct'))} | "
        f"{_usd(t.get('protected_usd'), signed=True)} |"
        for t in trades
    ]
    horizons = " | ".join(f"+{days} d" for days in SETTLEMENT_HORIZONS)
    lines += [
        "",
        "How each trade went: result, best and worst point in R, how much of its best it gave back, and where the "
        f"price was {', '.join(str(days) for days in SETTLEMENT_HORIZONS)} daily closes after the exit "
        "(\"pending\" until they have passed).",
        "",
        f"| Symbol | Result | MFE | MAE | Given back | {horizons} |",
        "|---|---:|---:|---:|---:|" + "---:|" * len(SETTLEMENT_HORIZONS),
    ]
    lines += [
        f"| {t['symbol']} | {_r(t.get('result_r'))} | {_r(t.get('mfe_r'))} | {_r(t.get('mae_r'))} | "
        f"{_r(t.get('giveback_r'))} | "
        + " | ".join(_pct((t.get("after_exit") or {}).get(str(days))) if (t.get("after_exit") or {}).get(str(days))
                     is not None else "pending" for days in SETTLEMENT_HORIZONS) + " |"
        for t in trades
    ]
    protected = [t["protected_usd"] for t in trades if t.get("protected_usd") is not None]
    if protected:
        lines += ["", f"Taken together, the exits saved {_usd(sum(protected), signed=True)} against holding the "
                      "positions until today." if sum(protected) >= 0 else
                      f"Taken together, the exits gave up {_usd(-sum(protected))} against holding the positions "
                      "until today."]
    return [*lines, ""]


def _regime_lines(observations: dict[str, Any], parameters: dict[str, Any]) -> list[str]:
    """Breadth of the basket, the record against passive benchmarks, and breakouts the heat limit left out."""
    regime, benchmarks = observations.get("regime"), observations.get("benchmarks")
    lines = ["### Regime and basket breadth", ""]
    if regime:
        lines += [
            f"- Breaking out (strength > 0): {regime['breaking_out']} of {regime['symbols']}.",
            f"- On pre-alert (strength between {regime['pre_alert_from']:g} and 0): {regime['pre_alert']}.",
            f"- Closed below their {parameters['exit_period']}-candle low: {regime['below_exit_channel']}.",
        ]
    if benchmarks:
        lines += [
            f"- Since the first report ({benchmarks['since']}): bot {_pct(benchmarks['bot_pct'])}, BTC buy and hold "
            f"{_pct(benchmarks['btc_pct'])}, equal-weight basket of {benchmarks['basket_symbols']} coins "
            f"{_pct(benchmarks['basket_pct'])}. Both benchmarks are fully invested; the bot only in what it holds.",
        ]
    turbulence = observations.get("turbulence")
    if turbulence:
        lines.append(
            f"- Turbulence (FinRL, Mahalanobis over {turbulence['coins']} coins and {turbulence['lookback']} days): "
            f"{turbulence['value']:.2f} per coin, percentile {turbulence['percentile']:.0f} of the last "
            f"{turbulence['history_days']} days. Above 90 the market is moving unusually; it is for information, "
            "the bot does not use it.")
    lines += _quality_lines(observations)
    discards = observations.get("heat_discards") or []
    lines += ["", "Breakouts left out by the open-risk limit in earlier cycles: "
              + ("none." if not discards else "")]
    if discards:
        lines += [
            "",
            "| Cycle | Symbol | Strength | Close then | Close today | Change |",
            "|---|---|---:|---:|---:|---:|",
            *[f"| {d['cycle_date']} | {d['symbol']} | {_text(d.get('breakout_strength'))} | {_price(d['close'])} | "
              f"{_price(d.get('close_today'))} | {_pct(d.get('change_pct'))} |" for d in discards],
        ]
    return [*lines, ""]


def _quality_lines(observations: dict[str, Any]) -> list[str]:
    """Edge ratio of the signal on history, and the closed trades in R with how little they can say yet."""
    lines = []
    edge = observations.get("edge_ratio")
    if edge:
        lines += [
            "",
            f"Signal quality on history ({edge['breakouts']} breakouts in the report's candles): edge ratio = mean "
            "best move / mean worst move, both in ATR, after the breakout. Close to 1 would be chance; it describes "
            "the signal, not the bot's orders.",
            "",
            "| Horizon | After a breakout | Any day | Breakouts |",
            "|---|---:|---:|---:|",
            *[f"| {h['days']} days | {_ratio_text(h['signal'])} | {_ratio_text(h['any_day'])} | {h['signals']} |"
              for h in edge["horizons"]],
        ]
    quality = observations.get("trade_quality") or {}
    if quality.get("trades"):
        sqn = f"{quality['sqn']:.2f}" if quality.get("sqn") is not None else f"n/a (n = {quality['trades']} < {MEANINGFUL_TRADES})"
        lines += [
            "",
            f"Closed trades: {quality['trades']}; expectancy {_r(quality['expectancy_r'])} per trade "
            f"(best {_r(quality['best_r'])}, worst {_r(quality['worst_r'])}); SQN {sqn}."
            + ("" if quality.get("meaningful") else
               f" With fewer than {MEANINGFUL_TRADES} trades these figures describe what happened; they are not "
               "evidence about the strategy."),
        ]
    return lines


def _ratio_text(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _tail_text(correlation: dict[str, Any]) -> str:
    days = correlation.get("tail_days", 0)
    if correlation.get("tail_mean") is None:
        return (f"Correlation on down days: not measured ({days} days with the basket below "
                f"{correlation.get('tail_threshold_pct', TAIL_DAY_RETURN * 100.0):g} %, {TAIL_MIN_DAYS} needed).")
    held = (f"; among the open ones {_text(correlation['tail_held_mean'])}"
            if correlation.get("tail_held_mean") is not None else "")
    return (f"Mean correlation on down days ({days} days with the basket below "
            f"{correlation['tail_threshold_pct']:g} %): {_text(correlation['tail_mean'])}{held}. On those days "
            f"{correlation['tail_down_share_pct']:.0f} % of the coins fell on average. Careful when comparing it with "
            "the correlation over all days: measuring it only on down days narrows the range of returns, which by "
            "itself pulls it down; the share of coins that fall has no such bias.")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f} %"


def _r(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}R"


def _excursion(position: dict[str, Any], kind: str) -> str:
    if position.get(f"{kind}_price") is None:
        return "—"
    return (f"{_price(position[f'{kind}_price'])} ({_pct(position[f'{kind}_pct'])}, "
            f"{_r(position.get(f'{kind}_r'))})")


def _sizing_lines(row: dict[str, Any], parameters: dict[str, Any], rank: int) -> list[str]:
    sizing, execution = row["sizing"], row.get("execution") or {}
    proposed = row["decision"] == Action.ENTER_LONG.value
    title = "proposed entry" if proposed else f"breakout not traded ({row['reason']})"
    lines = [
        f"### {rank}. {row['symbol']}: {title} (strength {row['breakout_strength']:+.2f} ATR)",
        "",
        f"1. Risk per trade = balance × {sizing['risk_pct']:g} % = {_usd(sizing['balance'])} × "
        f"{sizing['risk_pct'] / 100:g} = **{_usd(sizing['risk_budget_usd'])}**",
        f"2. Stop distance = {sizing['stop_atr_multiple']:g} × ATR({parameters['atr_period']}) = "
        f"{sizing['stop_atr_multiple']:g} × {_price(row['atr'])} = **{_price(sizing['stop_distance'])}**",
        f"3. Units by risk = risk / distance = {_usd(sizing['risk_budget_usd'])} / "
        f"{_price(sizing['stop_distance'])} = {sizing['lots_by_risk']:.6g}",
        f"4. Units by position cap = balance × {sizing['max_position_leverage']:g} / close = "
        f"{_usd(sizing['balance'] * sizing['max_position_leverage'])} / {_price(row['close'])} = "
        f"{sizing['lots_by_leverage']:.6g}",
        f"5. Take the smaller: {sizing['lots_raw']:.6g} (limited by {sizing['limited_by']})",
        f"6. Round down to the step {sizing['step_size']:g}: **{sizing['lots_final']:g} units**"
        + ("" if sizing["lots_final"] > 0.0 else f" (below the minimum of {sizing['min_qty']:g})"),
        f"7. Notional = {sizing['lots_final']:g} × {_price(row['close'])} = {_usd(sizing['nominal_usd'])}; "
        f"exchange minimum {_usd(sizing['min_notional_usd'])}: "
        f"{'met' if sizing['meets_min_notional'] else 'NOT met'}",
        f"8. Stop = close − distance = {_price(row['close'])} − {_price(sizing['stop_distance'])} = "
        f"{_price(sizing['stop_price'])}; at tick {sizing['tick_size']:g}: **{_price(sizing['stop_price_rounded'])}**",
        f"9. Loss if the stop is hit = {sizing['lots_final']:g} × {_price(sizing['stop_distance'])} = "
        f"{_usd(sizing['risk_at_stop_usd'])} ({sizing['risk_at_stop_pct']:.2f} % of the balance)",
    ]
    if proposed:
        match = "matches" if sizing.get("matches_order") else "does NOT match"
        lines.append(f"10. Orchestrator order: {sizing.get('order_volume', 0.0):g} units ({match} step 6). "
                     f"Result: {execution.get('status', '—')}"
                     + (f", {execution['quantity']:g} at {_price(execution.get('price'))}" if execution.get("quantity") else "")
                     + (f". Reason: {execution['reason']}" if execution.get("reason") else "") + ".")
    else:
        lines.append(f"10. Order not proposed: {row['reason']}.")
    return [*lines, ""]


def _usd(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+,.2f} USDT" if signed else f"{value:,.2f} USDT"


def _price(value: float | None) -> str:
    """A price with the decimals its size calls for: BTC does not read like DOGE."""
    if value is None:
        return "—"
    magnitude = abs(value)
    decimals = 2 if magnitude >= 1_000 else 4 if magnitude >= 1 else 6
    return f"{value:,.{decimals}f}"


def _text(value: float | None) -> str:
    return "not enough data" if value is None else f"{value:+.2f}"
