"""Tests for the portfolio analytics of the dashboard (``src/ui/analytics.py``)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.ui.analytics import by_symbol, execution_quality, executions_of, initial_risk, trade_statistics

DAY_MS = 86_400_000


def _trade(symbol: str, pnl: float, *, reason: str = "STOP", entry: float = 100.0, stop: float = 90.0,
           quantity: float = 10.0, days: float = 2.0, fees: float = 1.0, funding: float = 0.5) -> dict[str, Any]:
    return {"symbol": symbol, "quantity": quantity, "entry_price": entry, "exit_price": entry + pnl / quantity,
            "stop_price": stop, "opened_at": 0, "closed_at": int(days * DAY_MS), "reason": reason, "fees": fees,
            "funding": funding, "pnl": pnl}


TRADES = [
    _trade("AAAUSDT", 300.0, reason="EXIT_CHANNEL_BROKEN", days=10.0),
    _trade("AAAUSDT", -100.0),
    _trade("BBBUSDT", -50.0, days=1.0),
    _trade("CCCUSDT", 150.0, reason="EXIT_CHANNEL_BROKEN", days=7.0),
]


def test_statistics_of_closed_trades() -> None:
    stats = trade_statistics(TRADES)
    json.dumps(stats, allow_nan=False)
    assert (stats["count"], stats["wins"], stats["losses"]) == (4, 2, 2)
    assert stats["win_rate_pct"] == 50.0 and stats["net_usd"] == 300.0
    assert (stats["gross_profit_usd"], stats["gross_loss_usd"]) == (450.0, 150.0)
    assert stats["profit_factor"] == 3.0
    assert (stats["average_win_usd"], stats["average_loss_usd"], stats["payoff_ratio"]) == (225.0, -75.0, 3.0)
    assert stats["expectancy_usd"] == 75.0
    # Each trade risked 10 × (100 − 90) = 100: the results are +3, −1, −0.5 and +1.5 times that risk.
    assert stats["expectancy_r"] == pytest.approx(0.75)
    assert (stats["best_usd"], stats["worst_usd"]) == (300.0, -100.0)
    assert stats["average_days_held"] == pytest.approx(5.0)
    assert (stats["fees_usd"], stats["funding_usd"]) == (4.0, 2.0)
    assert stats["by_reason"] == {"EXIT_CHANNEL_BROKEN": {"count": 2, "pnl_usd": 450.0},
                                  "STOP": {"count": 2, "pnl_usd": -150.0}}


def test_statistics_without_trades_are_empty_not_wrong() -> None:
    stats = trade_statistics([])
    json.dumps(stats, allow_nan=False)
    assert stats["count"] == 0 and stats["net_usd"] == 0.0 and stats["by_reason"] == {}
    for key in ("win_rate_pct", "profit_factor", "average_win_usd", "average_loss_usd", "payoff_ratio",
                "expectancy_usd", "expectancy_r", "best_usd", "worst_usd", "average_days_held"):
        assert stats[key] is None, key


def test_only_winners_leave_the_ratios_empty_instead_of_infinite() -> None:
    stats = trade_statistics([_trade("AAAUSDT", 50.0), _trade("BBBUSDT", 25.0)])
    json.dumps(stats, allow_nan=False)
    assert stats["win_rate_pct"] == 100.0 and stats["profit_factor"] is None and stats["payoff_ratio"] is None
    assert stats["average_loss_usd"] is None and stats["average_win_usd"] == 37.5


@pytest.mark.parametrize("junk", [None, "text", 7, {"symbol": "AAAUSDT"}, {"pnl": 5.0}, {"symbol": 3, "pnl": 5.0},
                                  {"symbol": "AAAUSDT", "pnl": "a lot"}, {"symbol": "AAAUSDT", "pnl": True}])
def test_malformed_trades_are_ignored(junk: Any) -> None:
    assert trade_statistics([junk, _trade("AAAUSDT", 10.0)])["count"] == 1
    assert [row["symbol"] for row in by_symbol([junk, _trade("AAAUSDT", 10.0)], [])] == ["AAAUSDT"]


def test_the_initial_risk_needs_a_stop_below_the_entry() -> None:
    assert initial_risk(_trade("AAAUSDT", 0.0)) == 100.0
    assert initial_risk(_trade("AAAUSDT", 0.0, stop=100.0)) is None
    assert initial_risk({"quantity": 1.0, "entry_price": 100.0}) is None
    # With no known risk the trade counts in USDT, but not in multiples of the risk.
    stats = trade_statistics([_trade("AAAUSDT", 40.0, stop=120.0)])
    assert stats["expectancy_usd"] == 40.0 and stats["expectancy_r"] is None


def test_result_by_coin_adds_what_is_closed_and_what_is_open() -> None:
    rows = by_symbol(TRADES, [{"symbol": "BBBUSDT", "pnl_usd": 80.0}, {"symbol": "DDDUSDT", "pnl_usd": -20.0}])
    assert [row["symbol"] for row in rows] == ["AAAUSDT", "CCCUSDT", "BBBUSDT", "DDDUSDT"]  # from largest to smallest total
    aaa, ccc, bbb, ddd = rows
    assert (aaa["trades"], aaa["wins"], aaa["realized_usd"], aaa["open_usd"], aaa["total_usd"], aaa["is_open"]) == (
        2, 1, 200.0, 0.0, 200.0, False)
    assert (bbb["realized_usd"], bbb["open_usd"], bbb["total_usd"], bbb["is_open"]) == (-50.0, 80.0, 30.0, True)
    assert (ddd["trades"], ddd["total_usd"], ddd["is_open"]) == (0, -20.0, True)
    assert ccc["total_usd"] == 150.0 and by_symbol([], []) == []


def _report(day: str, *notes: dict[str, Any]) -> dict[str, Any]:
    return {"cycle_date": day, "observations": {"execution": list(notes)}}


def _note(symbol: str, action: str, slippage: float | None, executed: bool = True) -> dict[str, Any]:
    note: dict[str, Any] = {"symbol": symbol, "action": action, "status": "SIMULATED", "executed": executed}
    if slippage is not None:
        note |= {"price": 100.0 * (1 + slippage / 100), "signal_close": 100.0, "slippage_pct": slippage}
    return note


def test_executions_are_read_from_a_daily_report() -> None:
    report = _report("2026-10-12", _note("AAAUSDT", "OPEN_LONG", 0.4), _note("BBBUSDT", "CLOSE_LONG", -0.2),
                     _note("CCCUSDT", "OPEN_LONG", None, executed=False), _note("DDDUSDT", "OPEN_LONG", None))
    notes = executions_of(report)
    assert [(note["date"], note["symbol"], note["action"], note["slippage_pct"]) for note in notes] == [
        ("2026-10-12", "AAAUSDT", "OPEN_LONG", 0.4), ("2026-10-12", "BBBUSDT", "CLOSE_LONG", -0.2)]
    for empty in ({}, {"observations": None}, {"observations": {"execution": None}}, {"observations": {"execution": ["x"]}}):
        assert executions_of(empty) == []


def test_execution_quality_counts_the_cost_from_the_side_of_the_portfolio() -> None:
    notes = executions_of(_report("2026-10-12", _note("AAAUSDT", "OPEN_LONG", 0.4), _note("BBBUSDT", "OPEN_LONG", 0.2),
                                  _note("CCCUSDT", "CLOSE_LONG", -0.3), _note("DDDUSDT", "CLOSE_LONG", 0.1)))
    quality = execution_quality(notes)
    json.dumps(quality, allow_nan=False)
    # Buying above the close costs; selling below it, too. Selling above it is in favor.
    assert quality["entries"] == {"count": 2, "mean_pct": pytest.approx(0.3), "median_pct": pytest.approx(0.3),
                                  "worst_pct": 0.4}
    assert quality["exits"] == {"count": 2, "mean_pct": pytest.approx(0.1), "median_pct": pytest.approx(0.1),
                                "worst_pct": 0.3}
    assert quality["all"]["count"] == 4 and quality["all"]["mean_pct"] == pytest.approx(0.2)
    assert [(note["symbol"], note["cost_pct"]) for note in quality["recent"]] == [
        ("DDDUSDT", -0.1), ("CCCUSDT", 0.3), ("BBBUSDT", 0.2), ("AAAUSDT", 0.4)]  # the most recent one first
    assert len(execution_quality(notes, recent=2)["recent"]) == 2


def test_execution_quality_without_orders() -> None:
    quality = execution_quality([])
    assert quality["all"] == {"count": 0, "mean_pct": None, "median_pct": None, "worst_pct": None}
    assert quality["recent"] == []
