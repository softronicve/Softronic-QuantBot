"""Tests for the daily basket report (``DailyReporter``)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src.data.candle_loader import DailyCandleLoader
from src.engine.daily_reporter import (
    CANDIDATE,
    IN_POSITION,
    MARKET_CACHE,
    NO_DATA,
    WAITING,
    CycleSnapshot,
    DailyReporter,
    render_markdown,
    report_paths,
)
from src.engine.daily_runner import COMPLETED, DailyRunner
from src.engine.orchestrator import CLOSE_LONG, HOLD, MARKET, OPEN_LONG, OrderIntent
from src.engine.state_store import DailyStateStore
from src.data.candle_loader import CANDLE_COLUMNS
from src.execution.binance_futures_executor import (
    DRY_RUN,
    TESTNET,
    BinanceFuturesExecutor,
    FuturesAccount,
    FuturesPosition,
    MarketRules,
    PaperFuturesAccount,
)
from src.risk.portfolio_risk import PortfolioRiskManager
from src.risk.position_sizer import ATRSizer
from src.strategies.trend_breakout import TrendBreakoutStrategy
from tests.fake_futures import DAY_MS, NOW_MS, FakeFuturesClient, daily_klines, symbol_info

DAY = "2026-10-09"
BREAKS, QUIET, FALLS = 104.0, 100.0, 96.0
STRATEGY = TrendBreakoutStrategy()
RISK = PortfolioRiskManager()


def _frame(last: float, n: int = 150) -> pd.DataFrame:
    rows = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), int(k[6]))
            for k in daily_klines(last, n=n, forming=False)]
    return pd.DataFrame(rows, columns=CANDLE_COLUMNS)


def _rules(symbol: str, step: str = "0.001", tick: str = "0.01", min_notional: str = "5") -> MarketRules:
    return MarketRules(symbol, Decimal(step), Decimal(step), None, Decimal(tick), Decimal(min_notional))


def _snapshot(**overrides: Any) -> CycleSnapshot:
    """A four-coin cycle: one held, one that enters, one left out by risk and one that stays quiet."""
    lasts = {"AAAUSDT": 105.0, "BBBUSDT": BREAKS, "CCCUSDT": BREAKS, "DDDUSDT": QUIET}
    fields: dict[str, Any] = {
        "cycle_date": DAY, "mode": DRY_RUN, "status": COMPLETED, "late_minutes": 0.0, "balance": 10_000.0,
        "account": FuturesAccount(wallet_balance=9_990.0, available_balance=7_950.0, unrealized_pnl=40.0),
        "positions": [
            FuturesPosition("AAAUSDT", 10.0, 101.0, 105.0, 95.0),
            FuturesPosition("BBBUSDT", 9.615, 104.1, 104.1, 98.19),
        ],
        "market": {symbol: _frame(last) for symbol, last in lasts.items()},
        "rules": {symbol: _rules(symbol) for symbol in lasts},
        "intents": [
            OrderIntent("BBBUSDT", OPEN_LONG, 9.615, MARKET, 98.198, "BREAKOUT"),
            OrderIntent("AAAUSDT", HOLD, 0.0, MARKET, None, "TREND_INTACT"),
            OrderIntent("CCCUSDT", HOLD, 0.0, MARKET, None, "MAX_HEAT_EXCEEDED"),
            OrderIntent("DDDUSDT", HOLD, 0.0, MARKET, None, "NO_BREAKOUT"),
        ],
        "results": [{"symbol": "BBBUSDT", "action": OPEN_LONG, "status": "SIMULATED", "quantity": 9.615,
                     "price": 104.1, "stop_price": 98.19, "payloads": []}],
        "symbols": list(lasts),
    }
    fields.update(overrides)
    return CycleSnapshot(**fields)


def _reporter(tmp_path: Path, **options: Any) -> DailyReporter:
    return DailyReporter(tmp_path / "reports", STRATEGY, RISK, **options)


def _row(report: dict[str, Any], symbol: str) -> dict[str, Any]:
    return next(row for row in report["basket"] if row["symbol"] == symbol)


# --- Files -------------------------------------------------------------------


def test_write_creates_markdown_json_and_market_cache(tmp_path: Path) -> None:
    markdown_path, json_path = _reporter(tmp_path).write(_snapshot())

    assert markdown_path == tmp_path / "reports" / f"daily_audit_{DAY}.md"
    assert json_path == tmp_path / "reports" / f"daily_audit_{DAY}.json"
    assert (markdown_path, json_path) == report_paths(tmp_path / "reports", DAY)
    assert sorted(path.name for path in (tmp_path / "reports").iterdir()) == [
        f"daily_audit_{DAY}.json", f"daily_audit_{DAY}.md", MARKET_CACHE,
    ]
    assert markdown_path.read_text(encoding="utf-8").startswith(f"# Daily basket audit: {DAY}")


def test_json_is_strict_and_rebuilds_the_same_markdown(tmp_path: Path) -> None:
    markdown_path, json_path = _reporter(tmp_path).write(_snapshot())

    def refuse(constant: str) -> None:
        raise AssertionError(f"the JSON contains {constant}")

    report = json.loads(json_path.read_text(encoding="utf-8"), parse_constant=refuse)
    assert report["schema"] == 1 and report["cycle_date"] == DAY
    assert render_markdown(report) == markdown_path.read_text(encoding="utf-8")


def test_writing_again_replaces_the_report_of_that_day(tmp_path: Path) -> None:
    reporter = _reporter(tmp_path)
    reporter.write(_snapshot())
    _, json_path = reporter.write(_snapshot(status="COMPLETED_WITH_ERRORS"))
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "COMPLETED_WITH_ERRORS"
    assert len(list((tmp_path / "reports").glob("daily_audit_*"))) == 2
    assert not list((tmp_path / "reports").glob("*.tmp"))


# --- Executive summary -------------------------------------------------------


def test_open_result_of_the_summary_is_the_sum_of_the_positions_listed(tmp_path: Path) -> None:
    # The account brings an open PnL read at other prices (35): the positions table wins (40).
    stale = FuturesAccount(wallet_balance=9_990.0, available_balance=7_950.0, unrealized_pnl=35.0)
    report = _reporter(tmp_path).build(_snapshot(account=stale))
    listed = sum(position["unrealized_pnl"] for position in report["positions"])
    assert listed == pytest.approx(40.0) and report["account"]["unrealized_pnl"] == pytest.approx(listed)
    assert report["account"]["equity"] == pytest.approx(9_990.0 + listed)
    assert "| Unrealized PnL | +40.00 USDT |" in render_markdown(report)


def test_executive_summary_adds_up(tmp_path: Path) -> None:
    account = _reporter(tmp_path).build(_snapshot())["account"]
    heat = 10.0 * (101.0 - 95.0) + 9.615 * (104.1 - 98.19)
    nominal = 10.0 * 105.0 + 9.615 * 104.1

    assert account["wallet_balance"] == 9_990.0 and account["unrealized_pnl"] == 40.0
    assert account["equity"] == pytest.approx(10_030.0)
    assert account["margin_used"] == pytest.approx(10_030.0 - 7_950.0)
    assert account["heat_usd"] == pytest.approx(heat)
    assert account["heat_pct"] == pytest.approx(heat / 9_990.0 * 100.0)
    assert account["heat_limit_pct"] == 4.0
    assert account["nominal_usd"] == pytest.approx(nominal)
    assert account["leverage"] == pytest.approx(nominal / 9_990.0) and account["leverage_limit"] == 1.0
    assert account["open_positions"] == 2 and account["sizing_balance"] == 10_000.0


def test_positions_carry_their_risk_and_result(tmp_path: Path) -> None:
    first, second = _reporter(tmp_path).build(_snapshot())["positions"]
    assert first == {
        "symbol": "AAAUSDT", "quantity": 10.0, "entry_price": 101.0, "mark_price": 105.0, "stop_price": 95.0,
        "nominal_usd": 1_050.0, "risk_usd": 60.0, "risk_pct": pytest.approx(0.6), "unrealized_pnl": 40.0,
    }
    assert second["unrealized_pnl"] == pytest.approx(0.0) and second["risk_usd"] == pytest.approx(9.615 * 5.91)


@pytest.mark.parametrize(
    ("position", "risk"),
    [
        (FuturesPosition("AAAUSDT", 10.0, 101.0, 105.0, 102.0), 0.0),  # stop above the entry
        (FuturesPosition("AAAUSDT", 10.0, 101.0, 105.0, None), 1_050.0),  # no stop: the whole notional
        (FuturesPosition("AAAUSDT", -10.0, 101.0, 105.0, None), 1_050.0),  # short
    ],
)
def test_risk_of_unusual_positions(tmp_path: Path, position: FuturesPosition, risk: float) -> None:
    report = _reporter(tmp_path).build(_snapshot(positions=[position]))
    assert report["positions"][0]["risk_usd"] == pytest.approx(risk)
    assert report["account"]["heat_usd"] == pytest.approx(risk)


def test_parameters_are_the_ones_the_cycle_uses(tmp_path: Path) -> None:
    strategy = TrendBreakoutStrategy(entry_period=30, exit_period=15, atr_period=50, risk_pct=0.5, max_leverage=0.2)
    reporter = DailyReporter(tmp_path, strategy, PortfolioRiskManager(max_portfolio_heat_pct=6.0))
    report = reporter.build(_snapshot())
    assert report["parameters"] == {
        "entry_period": 30, "exit_period": 15, "atr_period": 50, "stop_atr_multiple": 2.0, "risk_pct": 0.5,
        "max_position_leverage": 0.2, "max_portfolio_heat_pct": 6.0, "max_portfolio_leverage": 1.0,
    }
    markdown = render_markdown(report)
    assert "| Symbol | Close | High 30 | Low 15 | ATR(50) |" in markdown and "6 % max" in markdown


# --- Decision matrix ---------------------------------------------------------


def test_decision_matrix_has_one_row_per_symbol_with_the_strategy_numbers(tmp_path: Path) -> None:
    cycle = _snapshot()
    report = _reporter(tmp_path).build(cycle)

    assert [row["symbol"] for row in report["basket"]] == ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
    for row in report["basket"]:
        levels = STRATEGY.levels(cycle.market[row["symbol"]])
        assert row["close"] == levels.close and row["atr"] == levels.atr
        assert row["high_entry"] == levels.entry_channel and row["low_exit"] == levels.exit_channel
        assert row["breakout_strength"] == pytest.approx((levels.close - levels.entry_channel) / levels.atr)
        assert row["proximity_pct"] == pytest.approx((levels.close / levels.entry_channel - 1.0) * 100.0)


def test_decisions_states_and_reasons(tmp_path: Path) -> None:
    report = _reporter(tmp_path).build(_snapshot())
    rows = {row["symbol"]: (row["state"], row["decision"], row["reason"]) for row in report["basket"]}
    assert rows == {
        "AAAUSDT": (IN_POSITION, "HOLD", "TREND_INTACT"),
        "BBBUSDT": (IN_POSITION, "ENTER_LONG", "BREAKOUT"),
        "CCCUSDT": (CANDIDATE, "HOLD", "MAX_HEAT_EXCEEDED"),
        "DDDUSDT": (WAITING, "HOLD", "NO_BREAKOUT"),
    }
    assert _row(report, "BBBUSDT")["execution"] == {
        "action": OPEN_LONG, "status": "SIMULATED", "quantity": 9.615, "price": 104.1, "stop_price": 98.19,
    }
    assert _row(report, "DDDUSDT")["execution"] is None


def test_an_exit_is_reported_as_exit_long(tmp_path: Path) -> None:
    cycle = _snapshot(
        positions=[],
        market={"AAAUSDT": _frame(FALLS)}, rules={"AAAUSDT": _rules("AAAUSDT")}, symbols=["AAAUSDT"],
        intents=[OrderIntent("AAAUSDT", CLOSE_LONG, 0.0, MARKET, None, "EXIT_CHANNEL_BROKEN")],
        results=[{"symbol": "AAAUSDT", "action": CLOSE_LONG, "status": "SIMULATED", "quantity": 10.0,
                  "price": 96.1, "pnl": -49.5}],
    )
    report = _reporter(tmp_path).build(cycle)
    row = _row(report, "AAAUSDT")
    assert (row["state"], row["decision"], row["reason"]) == (WAITING, "EXIT_LONG", "EXIT_CHANNEL_BROKEN")
    markdown = render_markdown(report)
    assert "### AAAUSDT: position closed (EXIT_CHANNEL_BROKEN)" in markdown
    assert "`reduceOnly`" in markdown and "PnL -49.50 USDT" in markdown


def test_symbols_without_usable_candles_do_not_break_the_report(tmp_path: Path) -> None:
    cycle = _snapshot()
    market = dict(cycle.market)
    del market["DDDUSDT"]
    market["CCCUSDT"] = _frame(BREAKS, n=40)  # fewer candles than the ATR needs
    intents = [intent for intent in cycle.intents if intent.symbol not in ("CCCUSDT", "DDDUSDT")]
    report = _reporter(tmp_path).build(_snapshot(market=market, intents=intents))

    missing, short = _row(report, "DDDUSDT"), _row(report, "CCCUSDT")
    assert (missing["state"], missing["reason"]) == (NO_DATA, "NO_DATA") and "close" not in missing
    assert short["state"] == NO_DATA and short["reason"].startswith("INVALID_DATA: 101 candles required")
    markdown = render_markdown(report)
    assert "| DDDUSDT | — | — | — | — | — | NO_DATA | HOLD | NO_DATA | — |" in markdown


def test_a_held_symbol_without_candles_is_still_shown_as_held(tmp_path: Path) -> None:
    cycle = _snapshot()
    market = {symbol: candles for symbol, candles in cycle.market.items() if symbol != "AAAUSDT"}
    intents = [OrderIntent("AAAUSDT", HOLD, 0.0, MARKET, None, "NO_DATA"), *cycle.intents[2:]]
    assert _row(_reporter(tmp_path).build(_snapshot(market=market, intents=intents)), "AAAUSDT")["state"] == IN_POSITION


# --- Order math --------------------------------------------------------------


@pytest.mark.parametrize("balance", [300.0, 10_000.0, 250_000.0])
@pytest.mark.parametrize("last", [BREAKS, 250.0, 0.9])
@pytest.mark.parametrize("step", ["0.001", "0.1", "1", "10"])
@pytest.mark.parametrize("max_leverage", [0.1, 1.0])
def test_sizing_shown_is_exactly_what_atr_sizer_computes(
    tmp_path: Path, balance: float, last: float, step: str, max_leverage: float
) -> None:
    strategy = TrendBreakoutStrategy(max_leverage=max_leverage)
    candles = _frame(BREAKS)
    candles[["open", "high", "low", "close"]] *= last / BREAKS
    rules = _rules("AAAUSDT", step=step)
    cycle = _snapshot(balance=balance, market={"AAAUSDT": candles}, rules={"AAAUSDT": rules},
                      symbols=["AAAUSDT"], intents=[], results=[], positions=[])
    sizing = DailyReporter(tmp_path, strategy, RISK).build(cycle)["basket"][0]["sizing"]

    levels = strategy.levels(candles)
    expected = ATRSizer().size_position(
        account_balance=balance, risk_pct=strategy.risk_pct, atr_val=levels.atr, multiplier=2.0, contract_size=1.0,
        step_size=float(step), asset_price=levels.close, max_leverage=max_leverage, min_volume=float(step),
    )
    assert sizing["lots_final"] == pytest.approx(expected.lots, abs=1e-12)
    assert sizing["nominal_usd"] == pytest.approx(expected.nominal_value)
    assert sizing["risk_at_stop_usd"] == pytest.approx(expected.risk_amount)
    assert sizing["stop_distance"] == pytest.approx(expected.stop_distance)
    if expected.lots > 0.0:
        assert sizing["limited_by"] == ("risk" if expected.limited_by == "risk" else "position cap")


def test_sizing_steps_are_consistent(tmp_path: Path) -> None:
    report = _reporter(tmp_path).build(_snapshot())
    row = _row(report, "BBBUSDT")
    s = row["sizing"]

    assert s["risk_budget_usd"] == pytest.approx(100.0) and s["balance"] == 10_000.0
    assert s["stop_distance"] == pytest.approx(2.0 * row["atr"])
    assert s["lots_by_risk"] == pytest.approx(100.0 / s["stop_distance"])
    assert s["lots_by_leverage"] == pytest.approx(1_000.0 / 104.0)
    assert s["lots_raw"] == pytest.approx(min(s["lots_by_risk"], s["lots_by_leverage"]))
    assert s["limited_by"] == "position cap" and s["lots_final"] == pytest.approx(9.615)
    assert s["lots_final"] <= s["lots_raw"] < s["lots_final"] + s["step_size"]
    assert s["nominal_usd"] == pytest.approx(9.615 * 104.0) and s["meets_min_notional"] is True
    assert s["stop_price"] == pytest.approx(104.0 - s["stop_distance"])
    assert s["stop_price_rounded"] == pytest.approx(98.19) and s["stop_price_rounded"] <= s["stop_price"]
    assert s["risk_at_stop_pct"] == pytest.approx(9.615 * s["stop_distance"] / 100.0)
    assert (s["order_volume"], s["matches_order"]) == (9.615, True)
    assert "order_volume" not in _row(report, "CCCUSDT")["sizing"]


def test_an_order_that_differs_from_the_shown_sizing_is_flagged(tmp_path: Path) -> None:
    cycle = _snapshot()
    intents = [OrderIntent("BBBUSDT", OPEN_LONG, 9.0, MARKET, 98.198, "BREAKOUT"), *cycle.intents[1:]]
    report = _reporter(tmp_path).build(_snapshot(intents=intents))
    assert _row(report, "BBBUSDT")["sizing"]["matches_order"] is False
    assert "9 units (does NOT match step 6)" in render_markdown(report)


def test_sizing_below_the_exchange_minimums_is_flagged(tmp_path: Path) -> None:
    rules = {"BBBUSDT": _rules("BBBUSDT", min_notional="50"), "CCCUSDT": _rules("CCCUSDT", step="1")}
    cycle = _snapshot(balance=300.0, rules={**_snapshot().rules, **rules})
    report = _reporter(tmp_path).build(cycle)

    small = _row(report, "BBBUSDT")["sizing"]  # 30 USDT of notional against a minimum of 50
    assert small["lots_final"] == pytest.approx(0.288) and small["meets_min_notional"] is False
    zero = _row(report, "CCCUSDT")["sizing"]  # 0.288 units do not reach a step of 1
    assert zero["lots_final"] == 0.0 and zero["meets_min_notional"] is False and zero["nominal_usd"] == 0.0
    markdown = render_markdown(report)
    assert "exchange minimum 50.00 USDT: NOT met" in markdown and "(below the minimum of 1)" in markdown


def test_the_orders_section_follows_the_order_in_which_the_cycle_processed_them(tmp_path: Path) -> None:
    # The basket goes from the weakest breakout to the strongest, and there is an exit: the report puts
    # the exit first and then the entries from strongest to weakest, which is how the risk is shared.
    runner, client, paper = build_cycle(tmp_path, {"WEAKUSDT": 103.5, "MIDUSDT": 105.0, "STRONGUSDT": 108.0,
                                                   "EXITUSDT": 96.0})
    paper.open("EXITUSDT", 5.0, 100.0, 80.0, client.now_ms - 86_400_000)
    runner.run_daily_cycle()
    markdown = report_paths(tmp_path / "reports" / "dry_run", DAY)[0].read_text(encoding="utf-8")
    section = markdown[markdown.index("## 3."):markdown.index("## 4.")]
    headings = [line for line in section.splitlines() if line.startswith("### ")]
    assert [heading.split(":")[0] for heading in headings] == [
        "### EXITUSDT", "### 1. STRONGUSDT", "### 2. MIDUSDT", "### 3. WEAKUSDT"]
    strengths = [float(heading.rsplit("strength ", 1)[1].split()[0]) for heading in headings[1:]]
    assert strengths == sorted(strengths, reverse=True) and strengths[-1] > 0.0
    assert "closes first and then entries" in section


def test_markdown_walks_through_the_sizing_of_proposed_and_rejected_breakouts(tmp_path: Path) -> None:
    markdown = render_markdown(_reporter(tmp_path).build(_snapshot()))
    # Same strength (+0.35 ATR for both): the symbol breaks the tie, as in the risk allocation.
    assert "### 1. BBBUSDT: proposed entry (strength +0.35 ATR)" in markdown
    assert "### 2. CCCUSDT: breakout not traded (MAX_HEAT_EXCEEDED) (strength +0.35 ATR)" in markdown
    assert "### AAAUSDT" not in markdown and "### DDDUSDT" not in markdown
    for step in (
        "1. Risk per trade = balance × 1 % = 10,000.00 USDT × 0.01 = **100.00 USDT**",
        "2. Stop distance = 2 × ATR(100)",
        "5. Take the smaller:",
        "6. Round down to the step 0.001: **9.615 units**",
        "exchange minimum 5.00 USDT: met",
        "at tick 0.01: **98.1900**",
        "10. Orchestrator order: 9.615 units (matches step 6). Result: SIMULATED, 9.615 at 104.1000.",
        "10. Order not proposed: MAX_HEAT_EXCEEDED.",
    ):
        assert step in markdown


def test_a_day_without_orders_says_so(tmp_path: Path) -> None:
    cycle = _snapshot(
        positions=[], market={"DDDUSDT": _frame(QUIET)}, rules={"DDDUSDT": _rules("DDDUSDT")}, symbols=["DDDUSDT"],
        intents=[OrderIntent("DDDUSDT", HOLD, 0.0, MARKET, None, "NO_BREAKOUT")], results=[],
    )
    markdown = render_markdown(_reporter(tmp_path).build(cycle))
    assert "No breakout and no exit today: no order was proposed." in markdown
    assert "- No order was sent or discarded today." in markdown
    assert "Open positions: none." in markdown and "### Open positions" not in markdown


# --- Observations ------------------------------------------------------------


def test_a_day_with_a_range_of_two_atr_is_flagged_as_anomalous(tmp_path: Path) -> None:
    wild = _frame(QUIET)
    wild.loc[wild.index[-1], ["high", "low", "close"]] = [112.0, 99.0, 111.0]
    market = {"DDDUSDT": wild, "EEEUSDT": _frame(QUIET)}
    cycle = _snapshot(market=market, symbols=list(market), positions=[], intents=[], results=[],
                      rules={symbol: _rules(symbol) for symbol in market})
    report = _reporter(tmp_path).build(cycle)

    by_symbol = {item["symbol"]: item for item in report["observations"]["volatility"]}
    previous = float(wild["close"].iloc[-2])
    assert by_symbol["DDDUSDT"]["anomaly"] is True
    assert by_symbol["DDDUSDT"]["true_range"] == pytest.approx(13.0)
    assert by_symbol["DDDUSDT"]["return_pct"] == pytest.approx((111.0 / previous - 1.0) * 100.0)
    assert by_symbol["DDDUSDT"]["range_in_atr"] == pytest.approx(13.0 / _row(report, "DDDUSDT")["atr"])
    assert by_symbol["EEEUSDT"]["anomaly"] is False and by_symbol["EEEUSDT"]["range_in_atr"] < 2.0
    prompt = report["llm_prompts"][0]["prompt"]
    assert "1 symbol had a range of 2 ATR or more: DDDUSDT." in prompt
    assert "| DDDUSDT |" in render_markdown(report) and "| yes |" in render_markdown(report)


def test_true_range_counts_the_gap_from_the_previous_close(tmp_path: Path) -> None:
    gapped = _frame(QUIET)
    previous = float(gapped["close"].iloc[-2])
    gapped.loc[gapped.index[-1], ["open", "high", "low", "close"]] = [120.0, 121.0, 119.0, 120.0]
    cycle = _snapshot(market={"DDDUSDT": gapped}, symbols=["DDDUSDT"], positions=[], intents=[], results=[])
    volatility = _reporter(tmp_path).build(cycle)["observations"]["volatility"][0]
    assert volatility["true_range"] == pytest.approx(121.0 - previous)


def test_correlation_of_what_is_held(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    common = rng.normal(0.0, 0.02, 150)

    def series(own: float, seed: int) -> pd.DataFrame:
        returns = common + np.random.default_rng(seed).normal(0.0, own, 150)
        frame = _frame(QUIET)
        frame["close"] = 100.0 * np.exp(np.cumsum(returns))
        frame["open"] = frame["close"].shift(1).fillna(100.0)
        frame["high"] = frame[["open", "close"]].max(axis=1) * 1.01
        frame["low"] = frame[["open", "close"]].min(axis=1) * 0.99
        return frame

    market = {"AAAUSDT": series(0.001, 1), "BBBUSDT": series(0.001, 2), "CCCUSDT": series(0.08, 3)}
    cycle = _snapshot(market=market, symbols=list(market), intents=[], results=[],
                      rules={symbol: _rules(symbol) for symbol in market})
    correlation = _reporter(tmp_path, correlation_days=60).build(cycle)["observations"]["correlation"]

    closes = pd.DataFrame({symbol: np.log(frame["close"]).diff().iloc[-60:].reset_index(drop=True)
                           for symbol, frame in market.items()})
    assert correlation["days"] == 60 and correlation["symbols"] == 3
    assert correlation["held_mean"] == pytest.approx(closes["AAAUSDT"].corr(closes["BBBUSDT"])) and correlation["held_mean"] > 0.95
    assert correlation["basket_mean"] < correlation["held_mean"]
    assert correlation["held_pairs"] == [{"pair": "AAAUSDT–BBBUSDT", "correlation": pytest.approx(correlation["held_mean"])}]


def test_correlation_needs_two_symbols(tmp_path: Path) -> None:
    cycle = _snapshot(market={"AAAUSDT": _frame(105.0)}, symbols=["AAAUSDT"], intents=[], results=[],
                      positions=[FuturesPosition("AAAUSDT", 10.0, 101.0, 105.0, 95.0)])
    report = _reporter(tmp_path).build(cycle)
    assert report["observations"]["correlation"]["basket_mean"] is None
    assert report["observations"]["correlation"]["held_mean"] is None
    assert "Mean correlation between them: not enough data." in report["llm_prompts"][1]["prompt"]


def test_execution_notes_measure_the_distance_to_the_signal_close(tmp_path: Path) -> None:
    cycle = _snapshot(results=[
        {"symbol": "BBBUSDT", "action": OPEN_LONG, "status": "SIMULATED", "quantity": 9.615, "price": 104.52},
        {"symbol": "CCCUSDT", "action": OPEN_LONG, "status": "REJECTED", "reason": "BELOW_MIN_NOTIONAL"},
    ])
    filled, rejected = _reporter(tmp_path).build(cycle)["observations"]["execution"]
    assert filled["executed"] is True and filled["signal_close"] == 104.0
    assert filled["slippage_pct"] == pytest.approx(0.5)
    assert rejected == {"symbol": "CCCUSDT", "action": OPEN_LONG, "status": "REJECTED",
                        "reason": "BELOW_MIN_NOTIONAL", "executed": False}


# --- Questions for the LLM ---------------------------------------------------


def test_llm_prompts_are_ready_to_paste(tmp_path: Path) -> None:
    report = _reporter(tmp_path).build(_snapshot(stopped=["EEEUSDT"], late_minutes=12.0))
    assert [prompt["title"] for prompt in report["llm_prompts"]] == [
        "Volatility anomalies", "Correlation and concentration", "Execution quality",
    ]
    for prompt in report["llm_prompts"]:
        text = prompt["prompt"]
        assert text.startswith(f"Cycle {DAY} in DRY_RUN mode. Balance 9990.00 USDT, 2 positions")
        # The warning that keeps a one-day audit from ending in a parameter change.
        assert "Its backing is historical and not yet confirmed going forward" in text
        assert "Do not propose changing parameters based on a single day" in text
        assert "entry 20 days, exit 10, ATR of 100, stop at 2 ATR" in text
        assert text.rstrip().endswith("?") and "1. " in text and "3. " in text
    execution = report["llm_prompts"][2]["prompt"]
    assert "12 minutes late" in execution
    assert "- BBBUSDT OPEN_LONG: SIMULATED, at 104.1 against a close of 104 (+0.096 %)" in execution
    assert "Breakouts not traded: CCCUSDT (MAX_HEAT_EXCEEDED)." in execution
    assert "Stops hit since the previous cycle: EEEUSDT." in execution
    assert "Open positions: AAAUSDT, BBBUSDT." in report["llm_prompts"][1]["prompt"]


def test_markdown_has_the_five_sections_and_fenced_prompts(tmp_path: Path) -> None:
    report = _reporter(tmp_path).build(_snapshot(stopped=["EEEUSDT"]))
    markdown = render_markdown(report)

    headings = [line for line in markdown.splitlines() if line.startswith("## ")]
    assert headings == [
        "## 1. Executive summary", "## 2. Basket decision matrix",
        "## 3. Order math, step by step", "## 4. Observations of the day", "## 5. Questions for the LLM",
    ]
    assert markdown.count("```text") == 3 and markdown.count("```") == 6
    for prompt in report["llm_prompts"]:
        assert f"### {prompt['title']}\n\n```text\n{prompt['prompt']}\n```" in markdown
    assert "| Portfolio heat (risk down to the stops) | 1.17 % of 4 % max (116.82 USDT) |" in markdown
    assert "| Total notional leverage | 0.205x of 1x max (2,050.92 USDT) |" in markdown
    assert "| Unrealized PnL | +40.00 USDT |" in markdown
    assert "| AAAUSDT | 10 | 101.0000 | 105.0000 | 95.0000 | 1,050.00 USDT | 60.00 USDT | 0.60 % | +40.00 USDT |" in markdown
    assert "| Symbol | Close | High 20 | Low 10 | ATR(100) | Strength | State | Decision | Reason | Execution |" in markdown
    assert "Stops hit since the previous cycle: EEEUSDT." in markdown


def test_the_paper_trading_notice_only_appears_in_dry_run(tmp_path: Path) -> None:
    reporter = _reporter(tmp_path)
    assert "> Paper trading: real prices and a paper account." in render_markdown(reporter.build(_snapshot()))
    testnet = render_markdown(reporter.build(_snapshot(mode=TESTNET)))
    assert "Paper trading" not in testnet and "Mode **TESTNET**" in testnet


@pytest.mark.parametrize(("value", "shown"), [(83_281.9, "83,281.90"), (104.0, "104.0000"), (0.08156418, "0.081564")])
def test_prices_keep_the_decimals_their_size_needs(tmp_path: Path, value: float, shown: str) -> None:
    candles = _frame(BREAKS)
    candles[["open", "high", "low", "close"]] *= value / BREAKS
    cycle = _snapshot(market={"AAAUSDT": candles}, symbols=["AAAUSDT"], positions=[], intents=[], results=[])
    assert f"| AAAUSDT | {shown} |" in render_markdown(_reporter(tmp_path).build(cycle))


# --- Candle cache ------------------------------------------------------------


def test_market_cache_holds_the_last_candles_and_their_channels(tmp_path: Path) -> None:
    cycle = _snapshot()
    _reporter(tmp_path, chart_candles=60).write(cycle)
    cache = json.loads((tmp_path / "reports" / MARKET_CACHE).read_text(encoding="utf-8"))

    assert cache["cycle_date"] == DAY and cache["mode"] == DRY_RUN and list(cache["symbols"]) == list(cycle.market)
    candles, drawn = cycle.market["BBBUSDT"], cache["symbols"]["BBBUSDT"]
    assert {len(values) for values in drawn.values()} == {60}
    assert drawn["time"] == [int(value) // 1000 for value in candles["open_time"].iloc[-60:]]
    assert drawn["close"] == [float(value) for value in candles["close"].iloc[-60:]]
    # Each candle's channel comes only from the previous ones: on the last one it matches the strategy's.
    levels = STRATEGY.levels(candles)
    assert drawn["entry_channel"][-1] == pytest.approx(levels.entry_channel)
    assert drawn["exit_channel"][-1] == pytest.approx(levels.exit_channel)
    assert drawn["entry_channel"][-2] == pytest.approx(float(candles["high"].iloc[-22:-2].max()))


def test_market_cache_leaves_gaps_where_the_channel_does_not_exist_yet(tmp_path: Path) -> None:
    cycle = _snapshot()
    _reporter(tmp_path, chart_candles=500).write(cycle)
    drawn = json.loads((tmp_path / "reports" / MARKET_CACHE).read_text(encoding="utf-8"))["symbols"]["BBBUSDT"]
    assert len(drawn["time"]) == 150
    assert drawn["entry_channel"][:20] == [None] * 20 and drawn["entry_channel"][20] is not None
    assert drawn["exit_channel"][:10] == [None] * 10 and drawn["exit_channel"][10] is not None


# --- With the daily cycle -----------------------------------------------------


def build_cycle(tmp_path: Path, lasts: dict[str, float], reporter: DailyReporter | None = None) -> tuple[DailyRunner, FakeFuturesClient, PaperFuturesAccount]:
    """A DRY_RUN daily cycle on the fake exchange, with its report."""
    client = FakeFuturesClient({symbol: symbol_info(symbol, step="0.001", min_qty="0.001", tick="0.01") for symbol in lasts})
    set_prices(client, lasts)
    paper = PaperFuturesAccount(tmp_path / "paper_futures.json", 10_000.0, 0.0005)
    executor = BinanceFuturesExecutor(mode=DRY_RUN, client=client, paper=paper, sleep=lambda _: None,
                                      clock=lambda: client.now_ms / 1000)
    strategy, risk = TrendBreakoutStrategy(), PortfolioRiskManager()
    runner = DailyRunner(
        executor, DailyCandleLoader(executor, candles=150), DailyStateStore(tmp_path / "runtime_state_dry_run.json"),
        list(lasts), strategy=strategy, risk_manager=risk, clock=lambda: client.now_ms / 1000,
        reporter=reporter or DailyReporter(tmp_path / "reports" / "dry_run", strategy, risk),
    )
    return runner, client, paper


def set_prices(client: FakeFuturesClient, lasts: dict[str, float]) -> None:
    for symbol, last in lasts.items():
        client.klines[symbol] = daily_klines(last, now_ms=client.now_ms)
        client.marks[symbol] = last


def test_the_daily_cycle_writes_its_report(tmp_path: Path) -> None:
    runner, _, paper = build_cycle(tmp_path, {"AAAUSDT": BREAKS, "BBBUSDT": QUIET})
    assert runner.run_daily_cycle()["status"] == COMPLETED

    markdown_path, json_path = report_paths(tmp_path / "reports" / "dry_run", DAY)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["status"] == COMPLETED and report["mode"] == DRY_RUN
    # The report shows the account as it was left after buying.
    assert [position["symbol"] for position in report["positions"]] == ["AAAUSDT"]
    assert report["account"]["open_positions"] == 1 and report["account"]["heat_pct"] > 0.5
    assert report["account"]["wallet_balance"] == pytest.approx(paper.balance)
    row = _row(report, "AAAUSDT")
    assert (row["decision"], row["execution"]["status"], row["sizing"]["matches_order"]) == ("ENTER_LONG", "SIMULATED", True)
    assert row["sizing"]["balance"] == 10_000.0
    assert "### 1. AAAUSDT: proposed entry (strength +" in markdown_path.read_text(encoding="utf-8")


def test_the_report_of_a_cycle_adds_up_while_prices_move(tmp_path: Path) -> None:
    runner, client, _ = build_cycle(tmp_path, {"AAAUSDT": BREAKS, "BBBUSDT": BREAKS})
    read = client.futures_mark_price

    def drifting(**parameters: Any) -> dict[str, Any]:  # each read gives a slightly lower price
        answer = read(**parameters)
        client.marks[parameters["symbol"]] -= 0.05
        return answer

    client.futures_mark_price = drifting  # type: ignore[method-assign]
    runner.run_daily_cycle()

    report = json.loads(report_paths(tmp_path / "reports" / "dry_run", DAY)[1].read_text(encoding="utf-8"))
    account, positions = report["account"], report["positions"]
    assert len(positions) == 2 and account["unrealized_pnl"] != 0.0
    assert account["unrealized_pnl"] == pytest.approx(sum(p["unrealized_pnl"] for p in positions), abs=1e-9)
    assert account["equity"] == pytest.approx(account["wallet_balance"] + account["unrealized_pnl"], abs=1e-9)
    # With leverage 1 the margin used is the notional of the positions.
    assert account["margin_used"] == pytest.approx(account["nominal_usd"], abs=1e-6)


def test_each_day_gets_its_own_report(tmp_path: Path) -> None:
    runner, client, _ = build_cycle(tmp_path, {"AAAUSDT": BREAKS, "BBBUSDT": QUIET})
    runner.run_daily_cycle()
    client.now_ms += DAY_MS
    set_prices(client, {"AAAUSDT": 90.0, "BBBUSDT": BREAKS})
    runner.run_daily_cycle()

    reports = tmp_path / "reports" / "dry_run"
    assert sorted(path.name for path in reports.glob("daily_audit_*.md")) == [
        "daily_audit_2026-10-09.md", "daily_audit_2026-10-10.md",
    ]
    second = json.loads(report_paths(reports, "2026-10-10")[1].read_text(encoding="utf-8"))
    assert second["stopped"] == ["AAAUSDT"] and [p["symbol"] for p in second["positions"]] == ["BBBUSDT"]
    assert json.loads((reports / MARKET_CACHE).read_text(encoding="utf-8"))["cycle_date"] == "2026-10-10"


def test_a_cycle_that_already_ran_does_not_rewrite_its_report(tmp_path: Path) -> None:
    runner, _, _ = build_cycle(tmp_path, {"AAAUSDT": BREAKS})
    runner.run_daily_cycle()
    json_path = report_paths(tmp_path / "reports" / "dry_run", DAY)[1]
    written = json_path.stat().st_mtime_ns
    assert runner.run_daily_cycle()["status"] == "ALREADY_EXECUTED"
    assert json_path.stat().st_mtime_ns == written


def test_a_failing_report_never_breaks_the_cycle(tmp_path: Path) -> None:
    class Broken(DailyReporter):
        def write(self, cycle: CycleSnapshot) -> tuple[Path, Path]:
            raise OSError("disk full")

    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}")
    try:
        runner, _, paper = build_cycle(tmp_path, {"AAAUSDT": BREAKS}, Broken(tmp_path / "x", STRATEGY, RISK))
        summary = runner.run_daily_cycle()
    finally:
        logger.remove(sink)
    assert summary["status"] == COMPLETED and "AAAUSDT" in paper.positions
    assert DailyStateStore(tmp_path / "runtime_state_dry_run.json").is_cycle_already_executed(DAY)
    assert any(line.startswith(f"ERROR|Cycle {DAY}: report not written") for line in lines)


def test_the_report_is_written_after_the_cycle_is_recorded(tmp_path: Path) -> None:
    store = DailyStateStore(tmp_path / "runtime_state_dry_run.json")
    seen: list[bool] = []

    class Watching(DailyReporter):
        def write(self, cycle: CycleSnapshot) -> tuple[Path, Path]:
            seen.append(DailyStateStore(tmp_path / "runtime_state_dry_run.json").is_cycle_already_executed(DAY))
            return super().write(cycle)

    runner, _, _ = build_cycle(tmp_path, {"AAAUSDT": BREAKS}, Watching(tmp_path / "r", STRATEGY, RISK))
    runner.run_daily_cycle()
    assert seen == [True] and not store.is_cycle_already_executed(DAY)  # ``store`` was loaded before the cycle


def test_timestamps_of_the_fake_clock_are_the_expected_day() -> None:
    assert pd.to_datetime(NOW_MS, unit="ms").strftime("%Y-%m-%d %H:%M:%S") == f"{DAY} 00:01:30"
