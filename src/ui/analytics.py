"""Portfolio analytics for the basket dashboard.

Three readings of what the bot has already done, taken from its own files:

* the closed trades, summarized the way a risk desk would do it
  (wins, average win and loss, expectancy, result in multiples
  of the initial risk);
* the result of each coin, closed and open;
* execution quality: how far from the close that gave the signal each
  order was executed, and what was paid in fees and funding.

They are sums over past facts. With few trades they say nothing about the
strategy: each figure comes with the number of trades it is drawn from.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from statistics import median
from typing import Any

DAY_MS = 86_400_000
OPEN_ACTION = "OPEN_LONG"


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _clean(trades: Iterable[Any]) -> list[dict[str, Any]]:
    """The trades with the minimum needed to count them; the rest is ignored without breaking the dashboard."""
    return [dict(trade) for trade in trades
            if isinstance(trade, Mapping) and isinstance(trade.get("symbol"), str) and _number(trade.get("pnl")) is not None]


def initial_risk(trade: Mapping[str, Any]) -> float | None:
    """What the trade risked when it was opened: from the entry to the stop, times the units."""
    quantity, entry, stop = (_number(trade.get(key)) for key in ("quantity", "entry_price", "stop_price"))
    if quantity is None or entry is None or stop is None:
        return None
    risk = quantity * (entry - stop)
    return risk if risk > 0.0 else None


def trade_statistics(trades: Iterable[Any]) -> dict[str, Any]:
    """Summary of the closed trades. What cannot be computed is ``None``."""
    closed = _clean(trades)
    results = [float(trade["pnl"]) for trade in closed]
    wins = [value for value in results if value > 0.0]
    losses = [value for value in results if value < 0.0]
    multiples = [float(trade["pnl"]) / risk for trade in closed if (risk := initial_risk(trade)) is not None]
    held = [(trade["closed_at"] - trade["opened_at"]) / DAY_MS for trade in closed
            if _number(trade.get("closed_at")) is not None and _number(trade.get("opened_at")) is not None]
    reasons: dict[str, dict[str, float]] = {}
    for trade in closed:
        bucket = reasons.setdefault(str(trade.get("reason") or "OTHER"), {"count": 0, "pnl_usd": 0.0})
        bucket["count"] += 1
        bucket["pnl_usd"] += float(trade["pnl"])
    gross_profit, gross_loss = sum(wins), -sum(losses)
    return {
        "count": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": len(wins) / len(closed) * 100.0 if closed else None,
        "net_usd": sum(results),
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        # With no losses there is no ratio; it is left empty instead of infinite.
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else None,
        "average_win_usd": gross_profit / len(wins) if wins else None,
        "average_loss_usd": -gross_loss / len(losses) if losses else None,
        "payoff_ratio": (gross_profit / len(wins)) / (gross_loss / len(losses)) if wins and losses else None,
        "expectancy_usd": sum(results) / len(closed) if closed else None,
        "expectancy_r": sum(multiples) / len(multiples) if multiples else None,
        "best_usd": max(results) if results else None,
        "worst_usd": min(results) if results else None,
        "average_days_held": sum(held) / len(held) if held else None,
        "fees_usd": sum(_number(trade.get("fees")) or 0.0 for trade in closed),
        "funding_usd": sum(_number(trade.get("funding")) or 0.0 for trade in closed),
        "by_reason": dict(sorted(reasons.items())),
    }


def by_symbol(trades: Iterable[Any], positions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Result of each coin: what is closed, what it has open and the total, from largest to smallest."""
    rows: dict[str, dict[str, Any]] = {}

    def row(symbol: str) -> dict[str, Any]:
        return rows.setdefault(symbol, {"symbol": symbol, "trades": 0, "wins": 0, "realized_usd": 0.0,
                                        "open_usd": 0.0, "is_open": False})

    for trade in _clean(trades):
        entry = row(trade["symbol"])
        entry["trades"] += 1
        entry["wins"] += float(trade["pnl"]) > 0.0
        entry["realized_usd"] += float(trade["pnl"])
    for position in positions:
        entry = row(str(position["symbol"]))
        entry["is_open"] = True
        entry["open_usd"] += _number(position.get("pnl_usd")) or 0.0
    for entry in rows.values():
        entry["total_usd"] = entry["realized_usd"] + entry["open_usd"]
    return sorted(rows.values(), key=lambda entry: (-entry["total_usd"], entry["symbol"]))


def executions_of(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The executed orders of a daily report, with their slippage against the close that gave the signal."""
    notes = ((report.get("observations") or {}).get("execution") or []) if isinstance(report, Mapping) else []
    return [
        {"date": report.get("cycle_date"), "symbol": note["symbol"], "action": note.get("action"),
         "price": note["price"], "signal_close": note["signal_close"], "slippage_pct": float(note["slippage_pct"])}
        for note in notes
        if isinstance(note, Mapping) and note.get("executed") and _number(note.get("slippage_pct")) is not None
        and _number(note.get("price")) is not None and isinstance(note.get("symbol"), str)
    ]


def execution_quality(executions: Sequence[Mapping[str, Any]], recent: int = 20) -> dict[str, Any]:
    """How much worse (or better) than the signal close the orders were executed.

    The cost is the slippage seen from the portfolio: buying above the
    close costs, selling below it does too. Positive is against.
    """
    costs = [note["slippage_pct"] if note.get("action") == OPEN_ACTION else -note["slippage_pct"] for note in executions]
    entries = [note["slippage_pct"] for note in executions if note.get("action") == OPEN_ACTION]
    exits = [-note["slippage_pct"] for note in executions if note.get("action") != OPEN_ACTION]

    def summary(values: list[float]) -> dict[str, Any]:
        return {"count": len(values), "mean_pct": sum(values) / len(values) if values else None,
                "median_pct": median(values) if values else None, "worst_pct": max(values) if values else None}

    return {
        "all": summary(costs),
        "entries": summary(entries),
        "exits": summary(exits),
        "recent": [dict(note, cost_pct=cost) for note, cost in list(zip(executions, costs))[-recent:]][::-1],
    }
