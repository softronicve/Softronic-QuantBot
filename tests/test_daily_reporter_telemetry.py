"""Tests of the post-trade figures of the daily report: exits, excursions, settlement, tail correlation, regime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.daily_reporter import TAIL_DAY_RETURN, render_markdown, report_paths
from src.execution.binance_futures_executor import FuturesPosition
from tests.fake_futures import DAY_MS
from tests.test_daily_reporter import DAY, QUIET, STRATEGY, _frame, _reporter, _rules, _snapshot


def _position(report: dict, symbol: str) -> dict:
    return next(position for position in report["positions"] if position["symbol"] == symbol)


def test_each_position_shows_both_exits_and_which_one_sells_first(tmp_path: Path) -> None:
    report = _reporter(tmp_path).build(_snapshot())
    held = _position(report, "AAAUSDT")
    channel = STRATEGY.levels(_frame(105.0)).exit_channel
    assert held["exit_channel"] == pytest.approx(channel)
    nearest = max((95.0, "stop"), (channel, "canal"))
    assert (held["effective_exit"], held["effective_exit_price"]) == (nearest[1], pytest.approx(nearest[0]))
    assert held["effective_exit_distance_pct"] == pytest.approx((nearest[0] / 105.0 - 1.0) * 100.0)
    # 10 units, entry 101, stop 95: 60 USDT at risk; 40 USDT open profit is +0.67R.
    assert held["r_multiple"] == pytest.approx(40.0 / 60.0)


def test_a_position_without_a_stop_has_no_r_multiple(tmp_path: Path) -> None:
    cycle = _snapshot(positions=[FuturesPosition("AAAUSDT", 10.0, 101.0, 105.0, None)])
    held = _position(_reporter(tmp_path).build(cycle), "AAAUSDT")
    assert held["r_multiple"] is None and held["effective_exit"] == "canal"


def test_excursions_since_the_entry_come_from_the_closed_daily_candles(tmp_path: Path) -> None:
    frame = _frame(105.0)
    entry_open = int(frame["open_time"].iloc[-3])
    cycle = _snapshot(opened_at={"AAAUSDT": entry_open + 90_000})  # bought a minute and a half into the candle
    held = _position(_reporter(tmp_path).build(cycle), "AAAUSDT")

    since = frame.iloc[-3:]
    high, low = float(since["high"].max()), float(since["low"].min())
    assert (held["mfe_price"], held["mae_price"]) == (pytest.approx(high), pytest.approx(low))
    assert held["mfe_usd"] == pytest.approx(10.0 * max(high - 101.0, 0.0))
    assert held["mae_usd"] == pytest.approx(10.0 * min(low - 101.0, 0.0))
    assert held["mfe_r"] == pytest.approx(held["mfe_usd"] / 60.0)
    assert held["mae_pct"] == pytest.approx((low / 101.0 - 1.0) * 100.0)


def test_no_excursions_without_an_entry_time_or_before_the_first_closed_candle(tmp_path: Path) -> None:
    last_open = int(_frame(105.0)["open_time"].iloc[-1])
    for opened_at in (None, {"AAAUSDT": last_open + DAY_MS + 90_000}):
        held = _position(_reporter(tmp_path).build(_snapshot(opened_at=opened_at)), "AAAUSDT")
        assert held["mfe_price"] is None and held["mae_r"] is None


def test_closed_trades_are_followed_after_the_exit(tmp_path: Path) -> None:
    trades = [
        {"symbol": "DDDUSDT", "reason": "STOP", "quantity": 2.0, "entry_price": 120.0, "exit_price": 110.0,
         "pnl": -20.0, "opened_at": 1, "closed_at": 2},
        {"symbol": "ZZZUSDT", "reason": "EXIT_CHANNEL_BROKEN", "quantity": 1.0, "entry_price": 5.0,
         "exit_price": 6.0, "pnl": 1.0, "opened_at": 3, "closed_at": 4},
    ]
    report = _reporter(tmp_path).build(_snapshot(closed_trades=trades))
    latest, earlier = report["closed_trades"]  # most recent first
    assert latest["symbol"] == "ZZZUSDT" and latest["close_today"] is None and latest["protected_usd"] is None
    today = float(_frame(QUIET)["close"].iloc[-1])
    assert earlier["close_today"] == pytest.approx(today)
    assert earlier["after_exit_pct"] == pytest.approx((today / 110.0 - 1.0) * 100.0)
    # The price kept falling after the stop: holding 2 units would have lost 2 × (110 − today) more.
    assert earlier["protected_usd"] == pytest.approx(2.0 * (110.0 - today))

    markdown = render_markdown(report)
    assert "### Closed positions and post-exit audit" in markdown
    assert "| DDDUSDT | STOP | 120.0000 | 110.0000 | -20.00 USDT |" in markdown


def _market_with_down_days(days: int = 80) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(rng.normal(0.0, 0.02, (days, 3)), columns=["AAAUSDT", "BBBUSDT", "CCCUSDT"])
    returns.iloc[-60::6] -= 0.05  # ten clear down days inside the window
    market = {}
    for symbol in returns:
        frame = _frame(QUIET, n=days + 1)
        close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns[symbol].to_numpy())]))
        frame["close"] = close
        frame["open"] = frame["close"].shift(1).fillna(100.0)
        frame["high"] = frame[["open", "close"]].max(axis=1) * 1.01
        frame["low"] = frame[["open", "close"]].min(axis=1) * 0.99
        market[symbol] = frame
    return market, returns


def test_tail_correlation_is_measured_only_on_the_days_the_basket_fell(tmp_path: Path) -> None:
    market, _ = _market_with_down_days()
    cycle = _snapshot(market=market, symbols=list(market), intents=[], results=[],
                      rules={symbol: _rules(symbol) for symbol in market})
    correlation = _reporter(tmp_path, correlation_days=60).build(cycle)["observations"]["correlation"]

    window = pd.DataFrame({symbol: np.log(frame["close"]).diff().iloc[-60:].reset_index(drop=True)
                           for symbol, frame in market.items()})
    down = window[np.expm1(window).mean(axis=1) < TAIL_DAY_RETURN]
    matrix = down.corr()
    expected = np.mean([matrix.loc["AAAUSDT", "BBBUSDT"], matrix.loc["AAAUSDT", "CCCUSDT"],
                        matrix.loc["BBBUSDT", "CCCUSDT"]])
    assert correlation["tail_days"] == len(down) >= 10
    assert correlation["tail_mean"] == pytest.approx(expected)
    assert correlation["tail_down_share_pct"] == pytest.approx((down < 0).mean(axis=1).mean() * 100.0)
    assert "Mean correlation on down days" in render_markdown(_reporter(tmp_path).build(cycle))


def test_tail_correlation_is_not_reported_on_too_few_down_days(tmp_path: Path) -> None:
    market, _ = _market_with_down_days()
    for frame in market.values():  # calm markets: no day falls 1 %
        frame["close"] = 100.0 + np.random.default_rng(3).normal(0.0, 0.01, len(frame))
    cycle = _snapshot(market=market, symbols=list(market), intents=[], results=[],
                      rules={symbol: _rules(symbol) for symbol in market})
    report = _reporter(tmp_path).build(cycle)
    correlation = report["observations"]["correlation"]
    assert correlation["tail_mean"] is None and correlation["tail_days"] < 5
    assert "Correlation on down days: not measured" in report["llm_prompts"][1]["prompt"]


def test_regime_counts_breakouts_pre_alerts_and_closes_below_the_exit(tmp_path: Path) -> None:
    regime = _reporter(tmp_path).build(_snapshot())["observations"]["regime"]
    # BBB and CCC close above their 20-day high; AAA (105) as well; DDD sits inside its range.
    assert regime["symbols"] == 4 and regime["breaking_out"] == 3
    assert regime["pre_alert"] + regime["breaking_out"] <= 4 and regime["below_exit_channel"] == 0


def _earlier_report(tmp_path: Path, cycle_date: str, **fields: object) -> None:
    _, path = report_paths(tmp_path / "reports", cycle_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cycle_date": cycle_date, "account": {"equity": 10_000.0}, "basket": [], **fields}))


def test_benchmarks_compare_the_record_with_passive_btc_and_an_equal_weight_basket(tmp_path: Path) -> None:
    btc = _frame(QUIET)
    market = {"BTCUSDT": btc, "AAAUSDT": _frame(105.0)}
    start = int(btc["open_time"].iloc[-11])
    start_date = pd.Timestamp(start + DAY_MS, unit="ms").strftime("%Y-%m-%d")
    _earlier_report(tmp_path, start_date)
    _earlier_report(tmp_path, "2099-01-01")  # a later report is not "earlier" and is ignored
    cycle = _snapshot(market=market, symbols=list(market), intents=[], results=[],
                      rules={symbol: _rules(symbol) for symbol in market})
    benchmarks = _reporter(tmp_path).build(cycle)["observations"]["benchmarks"]

    def change(frame: pd.DataFrame) -> float:
        return (float(frame["close"].iloc[-1]) / float(frame["close"].iloc[-11]) - 1.0) * 100.0

    assert benchmarks["since"] == start_date
    assert benchmarks["btc_pct"] == pytest.approx(change(btc))
    assert benchmarks["basket_pct"] == pytest.approx((change(btc) + change(market["AAAUSDT"])) / 2)
    # The report's equity is the balance plus the open result of the positions it lists.
    equity = 9_990.0 + sum(p.quantity * (p.mark_price - p.entry_price) for p in cycle.positions)
    assert benchmarks["bot_pct"] == pytest.approx((equity / 10_000.0 - 1.0) * 100.0)


def test_no_benchmarks_on_the_first_report(tmp_path: Path) -> None:
    assert _reporter(tmp_path).build(_snapshot())["observations"]["benchmarks"] is None


def test_breakouts_left_out_by_the_heat_limit_are_followed(tmp_path: Path) -> None:
    _earlier_report(tmp_path, "2026-10-05", basket=[
        {"symbol": "CCCUSDT", "reason": "MAX_HEAT_EXCEEDED", "close": 100.0, "breakout_strength": 0.4},
        {"symbol": "DDDUSDT", "reason": "NO_BREAKOUT", "close": 90.0},
    ])
    report = _reporter(tmp_path).build(_snapshot())
    (discard,) = report["observations"]["heat_discards"]
    today = float(_frame(104.0)["close"].iloc[-1])
    assert discard == {"cycle_date": "2026-10-05", "symbol": "CCCUSDT", "close": 100.0, "breakout_strength": 0.4,
                       "close_today": pytest.approx(today), "change_pct": pytest.approx((today / 100.0 - 1) * 100)}
    assert "Breakouts left out by the open-risk limit" in render_markdown(report)


def test_an_unreadable_earlier_report_is_skipped(tmp_path: Path) -> None:
    _, path = report_paths(tmp_path / "reports", "2026-10-05")
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    report = _reporter(tmp_path).build(_snapshot())
    assert report["observations"]["heat_discards"] == [] and report["observations"]["benchmarks"] is None


def test_the_report_with_the_new_figures_is_still_strict_json(tmp_path: Path) -> None:
    frame = _frame(105.0)
    cycle = _snapshot(opened_at={"AAAUSDT": int(frame["open_time"].iloc[-3])},
                      closed_trades=[{"symbol": "DDDUSDT", "reason": "STOP", "quantity": 2.0, "entry_price": 120.0,
                                      "exit_price": 110.0, "pnl": -20.0}])
    markdown_path, json_path = _reporter(tmp_path).write(cycle)
    report = json.loads(json_path.read_text(), parse_constant=lambda name: pytest.fail(name))
    assert render_markdown(report) == markdown_path.read_text()
    assert DAY in markdown_path.read_text() and "MFE" in markdown_path.read_text()
