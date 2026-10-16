"""Tests for the basket dashboard (``src/ui/dashboard.py``): it reads from disk and serves the page."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from config.settings import Settings
from src.engine.daily_reporter import MARKET_CACHE, report_paths
from src.engine.state_store import DailyStateStore
from src.ui import context, dashboard, quotes
from src.ui.dashboard import BasketDashboardServer, BasketPanel, ReportNotFoundError
from src.ui.quotes import LiveBySymbol, LiveCandles, LiveFeed, LivePrices, LiveQuotes, Quote
from tests.fake_futures import DAY_MS, NOW_MS
from tests.test_daily_reporter import build_cycle, set_prices

DAY, NEXT_DAY = "2026-10-09", "2026-10-10"
BREAKS, QUIET = 104.0, 100.0


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test goes out to the internet: the current price, if needed, is injected."""

    def refuse(*_: Any, **__: Any) -> Any:
        raise OSError("no network in the tests")

    monkeypatch.setattr(quotes, "fetch_quote", refuse)
    monkeypatch.setattr(quotes, "fetch_prices", refuse)
    monkeypatch.setattr(quotes, "fetch_stats", refuse)
    monkeypatch.setattr(quotes, "fetch_funding", refuse)
    monkeypatch.setattr(quotes, "fetch_candles", refuse)
    monkeypatch.setattr(quotes, "fetch_derivatives", refuse)
    monkeypatch.setattr(quotes, "fetch_book", refuse)
    monkeypatch.setattr(context, "http_get", refuse)


def _quoted(price: float, low: float | None = None) -> LiveQuotes:
    """A fixed current price for any symbol, with the low of today's candle."""
    today = 4_000_000_000_000  # a candle later than any check of the stop
    quote = Quote(price, ((today, today + DAY_MS - 1, price if low is None else low),))
    return LiveQuotes("https://example.invalid", fetch=lambda symbol: quote, wall_clock=lambda: 1_791_800_000.0)


def _panel(root: Path, now: datetime | None = None, **options: Any) -> BasketPanel:
    if now is not None:
        options["clock"] = lambda: now
    return BasketPanel(root / "runtime_state_dry_run.json", root / "reports" / "dry_run", root / "paper_futures.json",
                       **options)


def _utc(day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, second, tzinfo=timezone.utc)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


@pytest.fixture
def traded(tmp_path: Path) -> Path:
    """A state folder after two cycles: a buy, its stop the next day and another buy."""
    runner, client, _ = build_cycle(tmp_path, {"AAAUSDT": BREAKS, "BBBUSDT": QUIET})
    runner.run_daily_cycle()
    client.now_ms += DAY_MS
    set_prices(client, {"AAAUSDT": 90.0, "BBBUSDT": BREAKS})
    runner.run_daily_cycle()
    return tmp_path


def _files(root: Path) -> dict[str, tuple[int, int]]:
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file()}


# --- Reading from disk -------------------------------------------------------


def test_an_empty_state_directory_gives_an_empty_panel(tmp_path: Path) -> None:
    snapshot = _panel(tmp_path).snapshot()
    assert snapshot["report"] is None and snapshot["market"] == {}
    assert snapshot["history"] == [] and snapshot["trades"] == []
    assert snapshot["meta"]["notices"] == [] and snapshot["meta"]["report_dates"] == []
    assert list(tmp_path.iterdir()) == []


def test_snapshot_shows_the_latest_cycle(traded: Path) -> None:
    snapshot = _panel(traded).snapshot()
    json.dumps(snapshot, allow_nan=False)

    report = snapshot["report"]
    assert report["cycle_date"] == NEXT_DAY and report["mode"] == "DRY_RUN"
    assert report == json.loads(report_paths(traded / "reports" / "dry_run", NEXT_DAY)[1].read_text(encoding="utf-8"))
    assert snapshot["meta"]["report_dates"] == [NEXT_DAY, DAY] and snapshot["meta"]["notices"] == []
    assert sorted(snapshot["market"]) == ["AAAUSDT", "BBBUSDT"]
    assert {len(values) for values in snapshot["market"]["AAAUSDT"].values()} == {150}


def test_history_lists_the_cycles_newest_first(traded: Path) -> None:
    newest, oldest = _panel(traded).snapshot()["history"]
    assert (newest["cycle_date"], oldest["cycle_date"]) == (NEXT_DAY, DAY)
    assert newest["status"] == "COMPLETED" and newest["stopped"] == ["AAAUSDT"]
    assert [order["symbol"] for order in newest["opens"]] == ["BBBUSDT"]
    assert [order["symbol"] for order in oldest["opens"]] == ["AAAUSDT"] and oldest["closes"] == []
    assert oldest["holds"] == {"NO_BREAKOUT": 1} and oldest["late_minutes"] == 0.0
    assert newest["completed_at"] >= oldest["completed_at"]


def test_closed_trades_come_from_the_paper_account(traded: Path) -> None:
    (trade,) = _panel(traded).snapshot()["trades"]
    assert (trade["symbol"], trade["reason"]) == ("AAAUSDT", "STOP") and trade["pnl"] < 0.0
    without_paper = BasketPanel(traded / "runtime_state_dry_run.json", traded / "reports" / "dry_run", None)
    assert without_paper.snapshot()["trades"] == []


def test_the_panel_never_writes_to_the_state_directory(traded: Path) -> None:
    before = _files(traded)
    panel = _panel(traded)
    panel.snapshot()
    panel.report_file("md")
    panel.report_file("json", DAY)
    assert _files(traded) == before


def test_only_well_named_reports_count(traded: Path) -> None:
    reports = traded / "reports" / "dry_run"
    for name in ("daily_audit_2099-99-99x.json", "daily_audit_draft.json", "other_2026-10-11.json", "daily_audit_2026-10-11.md"):
        (reports / name).write_text("{}", encoding="utf-8")
    assert _panel(traded).report_dates() == [NEXT_DAY, DAY]


def test_an_unreadable_report_is_a_notice_not_a_crash(traded: Path) -> None:
    report_paths(traded / "reports" / "dry_run", NEXT_DAY)[1].write_text("{ broken", encoding="utf-8")
    snapshot = _panel(traded).snapshot()
    assert snapshot["report"] is None and len(snapshot["history"]) == 2
    assert any(notice.startswith(f"Could not read daily_audit_{NEXT_DAY}.json") for notice in snapshot["meta"]["notices"])


def test_an_unreadable_state_or_paper_file_is_a_notice(traded: Path) -> None:
    (traded / "runtime_state_dry_run.json").write_text("{ broken", encoding="utf-8")
    (traded / "paper_futures.json").write_text("[1, 2]", encoding="utf-8")
    snapshot = _panel(traded).snapshot()
    assert snapshot["history"] == [] and snapshot["trades"] == [] and snapshot["report"] is not None
    assert any("cannot be read" in notice for notice in snapshot["meta"]["notices"])


def test_a_chart_cache_from_another_cycle_is_pointed_out(traded: Path) -> None:
    cache = traded / "reports" / "dry_run" / MARKET_CACHE
    stale = json.loads(cache.read_text(encoding="utf-8")) | {"cycle_date": DAY}
    cache.write_text(json.dumps(stale), encoding="utf-8")
    assert _panel(traded).snapshot()["meta"]["notices"] == ["The chart candles are from a different cycle than the report."]


# --- Open positions -----------------------------------------------------------


def test_an_open_position_shows_how_it_is_doing_now(traded: Path) -> None:
    snapshot = _panel(traded, quotes=_quoted(110.0)).snapshot()
    json.dumps(snapshot, allow_nan=False)
    (position,) = snapshot["positions"]
    reported = snapshot["report"]["positions"][0]
    quantity, entry, stop = reported["quantity"], reported["entry_price"], reported["stop_price"]

    assert position["symbol"] == "BBBUSDT" and position["entry_price"] == entry and position["quantity"] == quantity
    assert position["price"] == 110.0 and position["price_is_live"] is True
    assert position["cost_usd"] == pytest.approx(quantity * entry)
    assert position["value_usd"] == pytest.approx(quantity * 110.0)
    assert position["pnl_usd"] == pytest.approx(quantity * (110.0 - entry)) and position["pnl_usd"] > 0.0
    assert position["pnl_pct"] == pytest.approx((110.0 / entry - 1.0) * 100.0)
    # Where it sells: the stop that was set when buying and what it would cost.
    assert position["stop_price"] == stop and position["stop_distance_pct"] == pytest.approx((stop / 110.0 - 1.0) * 100.0)
    assert position["result_at_stop_usd"] == pytest.approx(quantity * (stop - entry)) and position["result_at_stop_usd"] < 0.0
    assert position["status"] == dashboard.HOLDING
    # From the paper account: when it was bought and what was already paid in fees.
    assert position["opened_at"] == NOW_MS + DAY_MS and position["costs_usd"] == pytest.approx(quantity * entry * 0.0005)

    live = snapshot["live"]
    assert live == {"at": 1_791_800_000_000, "live": True, "open_pnl_usd": position["pnl_usd"],
                    "equity": snapshot["report"]["account"]["wallet_balance"] + position["pnl_usd"]}


def test_the_exit_channel_shown_is_the_one_the_next_cycle_will_use(traded: Path) -> None:
    snapshot = _panel(traded, quotes=_quoted(110.0)).snapshot()
    (position,) = snapshot["positions"]
    lows = snapshot["market"]["BBBUSDT"]["low"]
    period = snapshot["report"]["parameters"]["exit_period"]
    # The low of the last closed candles, yesterday's included: today's close will be compared with that.
    assert position["exit_channel"] == min(lows[-period:]) and period == 10
    assert position["exit_distance_pct"] == pytest.approx((position["exit_channel"] / 110.0 - 1.0) * 100.0)


def test_a_stop_touched_during_the_day_is_flagged_even_if_the_price_recovered(traded: Path) -> None:
    stop = _panel(traded).snapshot()["report"]["positions"][0]["stop_price"]
    (touched,) = _panel(traded, quotes=_quoted(105.0, low=stop - 0.5)).snapshot()["positions"]
    assert touched["status"] == dashboard.STOP_TOUCHED and touched["low_since_check"] == stop - 0.5
    # The result is already that of the stop, even if the price has gone back up or keeps falling.
    assert touched["price"] == 105.0 and touched["valued_at_stop"] is True
    assert touched["pnl_usd"] == pytest.approx(touched["result_at_stop_usd"]) and touched["pnl_usd"] < 0.0
    snapshot = _panel(traded, quotes=_quoted(stop - 10.0)).snapshot()
    (below,) = snapshot["positions"]
    assert below["status"] == dashboard.STOP_TOUCHED and below["pnl_usd"] == pytest.approx(below["result_at_stop_usd"])
    assert snapshot["live"]["open_pnl_usd"] == pytest.approx(below["result_at_stop_usd"])
    (fine,) = _panel(traded, quotes=_quoted(105.0, low=stop + 0.5)).snapshot()["positions"]
    assert fine["status"] == dashboard.HOLDING and fine["valued_at_stop"] is False


def test_lows_already_checked_by_the_cycle_do_not_count_again(traded: Path) -> None:
    stop = _panel(traded).snapshot()["report"]["positions"][0]["stop_price"]
    old = Quote(105.0, ((0, DAY_MS - 1, stop - 5.0),))  # a candle earlier than the last check of the stop
    panel = _panel(traded, quotes=LiveQuotes("https://example.invalid", fetch=lambda symbol: old))
    (position,) = panel.snapshot()["positions"]
    assert position["status"] == dashboard.HOLDING and position["low_since_check"] is None


def test_without_a_live_price_the_position_shows_the_price_of_the_last_cycle(traded: Path) -> None:
    for panel in (_panel(traded), _panel(traded, quotes=LiveQuotes("https://example.invalid"))):
        snapshot = panel.snapshot()
        (position,) = snapshot["positions"]
        reported = snapshot["report"]["positions"][0]
        assert position["price"] == reported["mark_price"] and position["price_is_live"] is False
        assert position["pnl_usd"] == pytest.approx(reported["unrealized_pnl"])
        assert snapshot["live"]["live"] is False and snapshot["live"]["at"] is None
        assert snapshot["meta"]["notices"] == []


def test_no_positions_no_price_request(tmp_path: Path) -> None:
    runner, _, _ = build_cycle(tmp_path, {"AAAUSDT": QUIET, "BBBUSDT": QUIET})
    runner.run_daily_cycle()
    asked: list[str] = []
    panel = _panel(tmp_path, quotes=LiveQuotes("https://example.invalid", fetch=asked.append))  # type: ignore[arg-type]
    snapshot = panel.snapshot()
    assert snapshot["positions"] == [] and asked == []
    assert snapshot["live"] == {"at": None, "live": False, "open_pnl_usd": 0.0, "equity": None}
    assert _panel(tmp_path / "empty").snapshot()["positions"] == []


def test_a_price_below_the_exit_channel_warns_of_the_sale_at_the_next_cycle() -> None:
    held = {"symbol": "AAAUSDT", "quantity": 10.0, "entry_price": 100.0, "mark_price": 100.0, "stop_price": 80.0}
    below = dashboard._position_now(held, Quote(94.0, ((0, DAY_MS - 1, 93.0),)), 95.0, {})
    assert below["status"] == dashboard.BELOW_EXIT_CHANNEL and below["pnl_usd"] == pytest.approx(-60.0)
    assert below["opened_at"] is None and below["costs_usd"] is None  # outside DRY_RUN there is no paper account
    above = dashboard._position_now(held, Quote(96.0, ((0, DAY_MS - 1, 93.0),)), 95.0, {})
    assert above["status"] == dashboard.HOLDING

    # No stop on the exchange and no channel: the result is shown and nothing else.
    bare = dashboard._position_now({**held, "stop_price": None}, None, None, {})
    assert (bare["stop_price"], bare["stop_distance_pct"], bare["result_at_stop_usd"]) == (None, None, None)
    assert (bare["exit_channel"], bare["exit_distance_pct"], bare["status"]) == (None, None, dashboard.HOLDING)


# --- Radar with the current prices ---------------------------------------------


def _priced(**prices: float) -> LivePrices:
    return LivePrices("https://example.invalid", fetch=lambda: dict(prices), wall_clock=lambda: 1_791_800_000.0)


def test_the_radar_places_the_price_of_now_against_the_levels_of_the_next_cycle(traded: Path) -> None:
    snapshot = _panel(traded, prices=_priced(AAAUSDT=95.0, BBBUSDT=110.0, ZZZUSDT=1.0)).snapshot()
    json.dumps(snapshot, allow_nan=False)
    radar, market = snapshot["radar"], snapshot["market"]
    assert radar["at"] == 1_791_800_000_000 and sorted(radar["rows"]) == ["AAAUSDT", "BBBUSDT"]  # only the basket

    flat = radar["rows"]["AAAUSDT"]  # no position: it is compared with the high of the last 20 closed candles
    assert flat["price"] == 95.0 and flat["entry_channel"] == max(market["AAAUSDT"]["high"][-20:])
    assert flat["change_pct"] == pytest.approx((95.0 / market["AAAUSDT"]["close"][-1] - 1.0) * 100.0)
    assert flat["proximity_pct"] == pytest.approx((95.0 / flat["entry_channel"] - 1.0) * 100.0) and flat["proximity_pct"] < 0.0
    assert flat["breaking"] is False and flat["exiting"] is False

    held = radar["rows"]["BBBUSDT"]  # with a position: even above its high, it is not a candidate
    assert held["price"] > held["entry_channel"] and held["breaking"] is False and held["exiting"] is False
    assert held["exit_channel"] == min(market["BBBUSDT"]["low"][-10:])


def test_the_radar_carries_the_market_figures_of_each_coin(traded: Path) -> None:
    stats = LiveFeed(lambda: {"AAAUSDT": {"change_24h_pct": -3.5, "volume_24h_usd": 2_000_000.0}}, what="x", ttl_seconds=30.0)
    funding = LiveFeed(lambda: {"AAAUSDT": 0.01, "BBBUSDT": -0.002}, what="x", ttl_seconds=60.0)
    rows = _panel(traded, prices=_priced(AAAUSDT=95.0, BBBUSDT=110.0), stats=stats, funding=funding).snapshot()["radar"]["rows"]
    assert (rows["AAAUSDT"]["change_24h_pct"], rows["AAAUSDT"]["volume_24h_usd"], rows["AAAUSDT"]["funding_pct"]) == (
        -3.5, 2_000_000.0, 0.01)
    # What does not arrive stays empty, without dropping the row.
    assert (rows["BBBUSDT"]["change_24h_pct"], rows["BBBUSDT"]["volume_24h_usd"], rows["BBBUSDT"]["funding_pct"]) == (
        None, None, -0.002)
    bare = _panel(traded, prices=_priced(AAAUSDT=95.0)).snapshot()["radar"]["rows"]["AAAUSDT"]
    assert (bare["change_24h_pct"], bare["volume_24h_usd"], bare["funding_pct"]) == (None, None, None)


def test_without_prices_the_market_figures_are_not_even_asked(traded: Path) -> None:
    asked: list[int] = []
    feed = LiveFeed(lambda: asked.append(1) or {}, what="x", ttl_seconds=30.0)
    _panel(traded, stats=feed, funding=feed).snapshot()
    assert asked == []


def test_other_markets_listed_on_the_exchange_come_with_the_prices(traded: Path) -> None:
    stats = LiveFeed(lambda: {"XAUUSDT": {"change_24h_pct": -0.7, "volume_24h_usd": 1.0}}, what="x", ttl_seconds=30.0)
    panel = _panel(traded, prices=_priced(AAAUSDT=95.0, XAUUSDT=4_176.1, SPYUSDT=779.4), stats=stats)
    assert panel.snapshot()["radar"]["cross"] == [
        {"symbol": "SPYUSDT", "name": "S&P 500 (SPY)", "price": 779.4, "change_24h_pct": None},
        {"symbol": "XAUUSDT", "name": "Gold", "price": 4_176.1, "change_24h_pct": -0.7}]


def test_the_radar_flags_what_would_happen_if_the_day_closed_now(traded: Path) -> None:
    rows = _panel(traded, prices=_priced(AAAUSDT=120.0, BBBUSDT=50.0)).snapshot()["radar"]["rows"]
    assert rows["AAAUSDT"]["breaking"] is True and rows["AAAUSDT"]["proximity_pct"] > 0.0
    assert rows["BBBUSDT"]["exiting"] is True and rows["BBBUSDT"]["breaking"] is False


def test_without_prices_the_radar_only_has_the_last_cycle(traded: Path, tmp_path: Path) -> None:
    for panel in (_panel(traded), _panel(traded, prices=LivePrices("https://example.invalid")),
                  _panel(traded, prices=_priced(AAAUSDT=0.0))):
        assert panel.snapshot()["radar"] == {"at": None, "rows": {}, "cross": []}
    asked: list[int] = []
    empty = _panel(tmp_path / "empty", prices=LivePrices("https://example.invalid", fetch=lambda: asked.append(1) or {}))
    assert empty.snapshot()["radar"] == {"at": None, "rows": {}, "cross": []} and asked == []  # without a report nothing is requested


# --- Analytics -----------------------------------------------------------------


def test_the_snapshot_carries_the_analytics_of_the_account(traded: Path) -> None:
    snapshot = _panel(traded, quotes=_quoted(110.0)).snapshot()
    json.dumps(snapshot, allow_nan=False)
    analytics = snapshot["analytics"]
    # Two cycles: AAA was bought and its stop was hit; BBB was bought and is still open.
    trades = analytics["trades"]
    assert (trades["count"], trades["wins"], trades["losses"]) == (1, 0, 1) and trades["by_reason"]["STOP"]["count"] == 1
    assert trades["net_usd"] == pytest.approx(snapshot["trades"][0]["pnl"]) and trades["expectancy_r"] < 0.0
    by_coin = {row["symbol"]: row for row in analytics["symbols"]}
    assert by_coin["AAAUSDT"]["realized_usd"] == pytest.approx(trades["net_usd"]) and not by_coin["AAAUSDT"]["is_open"]
    assert by_coin["BBBUSDT"]["is_open"] and by_coin["BBBUSDT"]["open_usd"] == pytest.approx(snapshot["positions"][0]["pnl_usd"])
    execution = analytics["execution"]
    assert execution["entries"]["count"] == 2 and execution["exits"]["count"] == 0
    assert [note["symbol"] for note in execution["recent"]] == ["BBBUSDT", "AAAUSDT"]
    assert "analytics" not in dashboard.LIVE_KEYS  # it only changes with the cycle: it goes in the full read


def test_an_empty_panel_has_empty_analytics(tmp_path: Path) -> None:
    analytics = _panel(tmp_path).snapshot()["analytics"]
    assert analytics["trades"]["count"] == 0 and analytics["symbols"] == []
    assert analytics["execution"]["all"]["count"] == 0
    assert _panel(tmp_path).coins() == []


def test_the_panel_knows_the_coins_of_the_basket_and_the_public_stream(traded: Path) -> None:
    panel = _panel(traded, stream="wss://fstream.binance.com")
    assert panel.snapshot()["meta"]["stream"] == "wss://fstream.binance.com" and panel.coins() == ["AAA", "BBB"]
    assert _panel(traded).snapshot()["meta"]["stream"] is None


# --- Account performance ------------------------------------------------------


def _cycle_point(day: int, equity: float, exposure: float, btc: float | None) -> dict[str, Any]:
    return {"date": f"2026-10-{day:02d}", "time": int(_utc(day).timestamp()), "equity": equity, "exposure": exposure,
            "btc": btc}


def test_the_curve_follows_the_account_and_its_fall_from_the_peak() -> None:
    cycles = [_cycle_point(9, 10_000.0, 0.0, 80_000.0), _cycle_point(10, 10_200.0, 0.5, 80_000.0),
              _cycle_point(11, 9_690.0, 0.5, 88_000.0), _cycle_point(12, 9_894.0, 0.25, 79_200.0)]
    performance = dashboard._performance(cycles, None, None, int(_utc(12, 15).timestamp()))
    points, stats = performance["points"], performance["stats"]

    assert [point["equity"] for point in points] == [10_000.0, 10_200.0, 9_690.0, 9_894.0]
    assert [point["drawdown_pct"] for point in points] == pytest.approx([0.0, 0.0, -5.0, -3.0])
    # The benchmark: with no exposure it does not move; with half invested, half of what BTC does.
    assert [point["benchmark"] for point in points] == pytest.approx([10_000.0, 10_000.0, 10_500.0, 9_975.0])
    assert stats["since"] == "2026-10-09" and stats["cycles"] == 4 and stats["is_live"] is False
    assert stats["return_pct"] == pytest.approx(-1.06) and stats["benchmark_return_pct"] == pytest.approx(-0.25)
    assert stats["drawdown_pct"] == pytest.approx(-3.0) and stats["max_drawdown_pct"] == pytest.approx(-5.0)
    assert stats["mean_exposure_pct"] == pytest.approx(31.25)
    json.dumps(performance, allow_nan=False)


def test_the_curve_ends_with_the_account_as_it_is_now() -> None:
    cycles = [_cycle_point(11, 10_000.0, 0.0, 80_000.0), _cycle_point(12, 9_998.0, 0.25, 84_000.0)]
    now = int(_utc(12, 15, 49).timestamp())
    performance = dashboard._performance(cycles, 9_848.0, 82_320.0, now)
    last = performance["points"][-1]
    assert len(performance["points"]) == 3 and (last["date"], last["time"], last["equity"]) == (None, now, 9_848.0)
    assert last["benchmark"] == pytest.approx(10_000.0 * (1.0 + 0.25 * -0.02))  # BTC −2 % with a quarter invested
    assert last["drawdown_pct"] == pytest.approx(-1.52)
    stats = performance["stats"]
    assert stats["is_live"] is True and stats["equity"] == 9_848.0 and stats["cycles"] == 2
    assert stats["mean_exposure_pct"] == pytest.approx(12.5)  # the mean is over the cycles, without the current point


def test_a_day_without_a_btc_price_does_not_move_the_reference() -> None:
    cycles = [_cycle_point(9, 10_000.0, 1.0, 100.0), _cycle_point(10, 10_000.0, 1.0, None),
              _cycle_point(11, 10_000.0, 1.0, 110.0)]
    benchmark = [point["benchmark"] for point in dashboard._performance(cycles, None, None, 0)["points"]]
    assert benchmark == pytest.approx([10_000.0, 10_000.0, 11_000.0])
    assert dashboard._performance([], None, None, 0) == {"points": [], "stats": None}
    assert dashboard._performance([], 9_000.0, 100.0, 5) == {"points": [], "stats": None}


@pytest.mark.parametrize("report", [None, [], {}, {"account": {}}, {"account": {"equity": 0.0, "nominal_usd": 0.0}},
                                    {"account": {"equity": "a lot", "nominal_usd": 1.0}}])
def test_a_report_without_a_usable_account_adds_no_point(report: Any) -> None:
    assert dashboard._equity_point("2026-10-09", report) is None


def test_the_panel_draws_one_point_per_daily_report(traded: Path) -> None:
    snapshot = _panel(traded).snapshot()
    points = snapshot["performance"]["points"]
    assert [point["date"] for point in points] == [DAY, NEXT_DAY]
    assert points[0]["time"] == int(_utc(9).timestamp())
    assert points[-1]["equity"] == snapshot["report"]["account"]["equity"]
    assert points[-1]["exposure_pct"] == pytest.approx(
        snapshot["report"]["account"]["nominal_usd"] / snapshot["report"]["account"]["equity"] * 100.0)
    assert snapshot["performance"]["stats"]["is_live"] is False

    live = _panel(traded, _utc(10, 12), quotes=_quoted(110.0)).snapshot()
    assert len(live["performance"]["points"]) == 3 and live["performance"]["stats"]["is_live"] is True
    assert live["performance"]["points"][-1]["equity"] == pytest.approx(live["live"]["equity"])
    assert _panel(traded.parent / "empty").snapshot()["performance"] == {"points": [], "stats": None}


def test_old_reports_are_read_only_once(traded: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _panel(traded)
    read: list[str] = []
    original = BasketPanel._read

    def counting(path: Path, notices: list[str]) -> Any:
        read.append(path.name)
        return original(path, notices)

    monkeypatch.setattr(BasketPanel, "_read", staticmethod(counting))
    panel.snapshot()
    panel.snapshot()
    old = f"daily_audit_{DAY}.json"
    assert read.count(old) == 1
    # If a report is rewritten, it is read again.
    path = traded / "reports" / "dry_run" / old
    report = json.loads(path.read_text(encoding="utf-8"))
    report["account"]["equity"] = 12_345.0
    path.write_text(json.dumps(report), encoding="utf-8")
    assert panel.snapshot()["performance"]["points"][0]["equity"] == 12_345.0 and read.count(old) == 2


# --- Next cycle --------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "expected", "pending"),
    [
        # Before today's time: today's is due.
        (_utc(10, 0, 0, 0), _utc(10, 0, 1, 30), False),
        (_utc(10, 0, 1, 29), _utc(10, 0, 1, 30), False),
        # Today's (day 10) is already recorded as completed: tomorrow's is due.
        (_utc(10, 0, 1, 30), _utc(11, 0, 1, 30), False),
        (_utc(10, 20, 0, 0), _utc(11, 0, 1, 30), False),
        # The time of day 11 has passed and its cycle is not recorded: it is still the next one, pending.
        (_utc(11, 0, 1, 30), _utc(11, 0, 1, 30), True),
        (_utc(11, 19, 0, 0), _utc(11, 0, 1, 30), True),
    ],
)
def test_next_cycle_is_announced_with_its_time(traded: Path, now: datetime, expected: datetime, pending: bool) -> None:
    meta = _panel(traded, now).snapshot()["meta"]
    assert meta["next_cycle"] == {"at": _ms(expected), "pending": pending}
    assert meta["generated_at"] == _ms(now)


def test_next_cycle_without_any_state(tmp_path: Path) -> None:
    assert _panel(tmp_path, _utc(9, 0, 0, 0)).snapshot()["meta"]["next_cycle"] == {
        "at": _ms(_utc(9, 0, 1, 30)), "pending": False}
    assert _panel(tmp_path, _utc(9, 6, 0, 0)).snapshot()["meta"]["next_cycle"] == {
        "at": _ms(_utc(9, 0, 1, 30)), "pending": True}


def test_a_cycle_that_started_but_did_not_finish_is_still_pending(tmp_path: Path) -> None:
    DailyStateStore(tmp_path / "runtime_state_dry_run.json").mark_cycle_started("2026-10-09")
    assert _panel(tmp_path, _utc(9, 0, 5, 0)).snapshot()["meta"]["next_cycle"] == {
        "at": _ms(_utc(9, 0, 1, 30)), "pending": True}


def test_next_cycle_follows_the_configured_delay(traded: Path) -> None:
    meta = _panel(traded, _utc(10, 0, 5, 0), run_delay_seconds=600.0).snapshot()["meta"]
    assert meta["next_cycle"] == {"at": _ms(_utc(10, 0, 10, 0)), "pending": False}


def test_an_unreadable_state_does_not_break_the_schedule(traded: Path) -> None:
    (traded / "runtime_state_dry_run.json").write_text("{", encoding="utf-8")
    meta = _panel(traded, _utc(10, 20, 0, 0)).snapshot()["meta"]
    assert meta["next_cycle"] == {"at": _ms(_utc(10, 0, 1, 30)), "pending": True}
    assert len(meta["notices"]) == 1


# --- Report download ---------------------------------------------------------


def test_report_file_returns_the_file_on_disk(traded: Path) -> None:
    panel = _panel(traded)
    markdown_path, json_path = report_paths(traded / "reports" / "dry_run", NEXT_DAY)
    assert panel.report_file("md") == (markdown_path.read_bytes(), f"daily_audit_{NEXT_DAY}.md")
    assert panel.report_file("json") == (json_path.read_bytes(), f"daily_audit_{NEXT_DAY}.json")
    body, name = panel.report_file("md", DAY)
    assert name == f"daily_audit_{DAY}.md" and body.decode("utf-8").startswith(f"# Daily basket audit: {DAY}")


@pytest.mark.parametrize(
    "requested", ["2026-10-11", "", "yesterday", "../../paper_futures", "../../../.env", "2026-10-09/../x", "2026-10-09\x00"]
)
def test_only_existing_report_dates_can_be_downloaded(traded: Path, requested: str) -> None:
    with pytest.raises(ReportNotFoundError):
        _panel(traded).report_file("md", requested)


def test_no_report_yet(tmp_path: Path) -> None:
    with pytest.raises(ReportNotFoundError, match="there is no report yet"):
        _panel(tmp_path).report_file("md")


# --- Server ------------------------------------------------------------------


class _CountingPanel(BasketPanel):
    def __init__(self, root: Path) -> None:
        super().__init__(root / "runtime_state_dry_run.json", root / "reports" / "dry_run", root / "paper_futures.json")
        self.calls = 0
        self.error: Exception | None = None

    def snapshot(self) -> dict[str, Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return super().snapshot()


@pytest.fixture
def served(traded: Path) -> Iterator[tuple[str, _CountingPanel]]:
    panel = _CountingPanel(traded)
    server = BasketDashboardServer(("127.0.0.1", 0), panel, cache_seconds=60.0)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", panel
    finally:
        server.shutdown()
        server.server_close()


def _get(url: str) -> tuple[int, Any, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def test_serves_the_page_and_its_assets(served: tuple[str, _CountingPanel]) -> None:
    base, _ = served
    status, headers, body = _get(base + "/")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    for anchor in (b'id="portfolio"', b'id="radar"', b'id="chart"', b'id="math"', b'id="history"',
                   b'id="clock-utc"', b'id="clock-local"', b'id="next-utc"', b'id="next-local"', b'id="next-in"',
                   b'id="positions"', b'id="stop-note"', "What each figure means".encode(),
                   b'id="equity-chart"', b'id="drawdown-chart"', b'id="curve-stats"',
                   b'id="live"', b'id="radar-preview"', b'id="th-cycle"',
                   b'id="timeframes"', b'id="list-search"', b'id="list-head"', b'id="breadth"',
                   b'id="context"', b'id="trade-stats"', b'id="corr-matrix"', b'id="book"', b'id="tape"',
                   b'id="derivatives"', b'id="news"', b'id="calendar"', b'id="command"', b'id="help"',
                   b'id="book-pair"', b'id="tape-pair"', b'id="derivatives-pair"',
                   "Download today's audit report".encode()):
        assert anchor in body
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store" and headers["X-Content-Type-Options"] == "nosniff"

    for asset, expected in (("app.js", "text/javascript"), ("styles.css", "text/css"),
                            ("analytics.js", "text/javascript"), ("context.js", "text/javascript"),
                            ("book.js", "text/javascript"), ("shell.js", "text/javascript"),
                            ("lightweight-charts.js", "text/javascript")):
        status, headers, body = _get(f"{base}/static/{asset}")
        assert status == 200 and headers["Content-Type"].startswith(expected) and len(body) > 1_000


def test_the_page_loads_nothing_from_other_sites(served: tuple[str, _CountingPanel]) -> None:
    base, _ = served
    page = _get(base + "/")[2]
    scripts = re.findall(rb'<script src="(/static/[a-z-]+\.js)">', page)
    assert len(scripts) == 6  # the charting library and the five files of the dashboard
    for path in ("/", "/static/styles.css", *(script.decode() for script in scripts)):
        body = _get(base + path)[2]
        if path.endswith("lightweight-charts.js"):
            continue  # the library carries in its text the URLs of its license and of its attribution notice
        # No URL written out: what the page shows from outside is given to it by the server.
        assert b"http://" not in body and b"wss://" not in body, path
        assert b"https://" not in body.replace(b'startsWith("https://")', b""), path


@pytest.mark.parametrize(
    "path", ["/static/../dashboard.py", "/static/..%2Fdashboard.py", "/dashboard.py", "/static/", "/api", "/.env",
             "/static/index.htm", "/api/report.pdf"]
)
def test_serves_nothing_outside_its_table(served: tuple[str, _CountingPanel], path: str) -> None:
    assert _get(served[0] + path)[0] == 404


def test_snapshot_endpoint_is_json_and_cached(served: tuple[str, _CountingPanel]) -> None:
    base, panel = served
    status, headers, body = _get(base + "/api/snapshot")
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert json.loads(body)["report"]["cycle_date"] == NEXT_DAY
    assert _get(base + "/api/snapshot")[2] == body and panel.calls == 1


def test_live_endpoint_carries_only_what_moves_between_cycles(served: tuple[str, _CountingPanel]) -> None:
    base, panel = served
    full = json.loads(_get(base + "/api/snapshot")[2])
    status, headers, body = _get(base + "/api/live")
    live = json.loads(body)
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert sorted(live) == ["live", "meta", "performance", "positions", "radar"]
    assert all(live[key] == full[key] for key in live)
    assert len(body) < len(json.dumps(full)) / 5  # without the candles or the report
    assert panel.calls == 1  # both answers come from the same snapshot


def _serving(**parts: Any) -> Iterator[str]:
    """A dashboard server with no data, with the parts given; it is used as ``for base in _serving(...)``."""
    server = BasketDashboardServer(("127.0.0.1", 0), BasketPanel(Path("nothing"), Path("nothing")), **parts)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _candles_server(feed: Any) -> Iterator[str]:
    return _serving(candles=None if feed is None else LiveCandles("https://example.invalid", fetch=feed))


@pytest.mark.parametrize("path", ["/api/derivatives", "/api/book"])
def test_per_contract_endpoints_validate_cache_and_hide_failures(path: str) -> None:
    asked: list[str] = []

    def feed(symbol: str) -> dict[str, Any]:
        asked.append(symbol)
        if symbol == "BADUSDT":
            raise OSError("https://secret.example/path")
        return {"symbol_seen": symbol}

    live = LiveBySymbol(feed, ttl_seconds=60.0)
    for base in _serving(derivatives=live, book=live):
        assert json.loads(_get(f"{base}{path}?symbol=NEARUSDT")[2]) == {"symbol_seen": "NEARUSDT"}
        _get(f"{base}{path}?symbol=NEARUSDT")
        assert asked == ["NEARUSDT"]  # the repeated one comes from the cache
        for bad in ("nearusdt", "NEARUSD", "NEAR%2F..%2FUSDT", ""):
            assert _get(f"{base}{path}?symbol={bad}")[0] == 400
        status, _, body = _get(f"{base}{path}?symbol=BADUSDT")
        assert status == 502 and b"secret" not in body
    for base in _serving():
        assert _get(f"{base}{path}?symbol=NEARUSDT")[0] == 404


def test_context_endpoint_serves_what_the_sources_gave_and_never_fails() -> None:
    pages = {context.GLOBAL_URL: json.dumps({"data": {
        "total_market_cap": {"usd": 2.8e12}, "total_volume": {"usd": 7.7e10},
        "market_cap_change_percentage_24h_usd": -3.0, "market_cap_percentage": {"btc": 59.1, "eth": 10.9}}}).encode()}

    def get(url: str) -> bytes:
        if url not in pages:
            raise OSError("source down")
        return pages[url]

    for base in _serving(context=context.MarketContext(lambda: ["BTC"], get=get)):
        status, headers, body = _get(base + "/api/context")
        answer = json.loads(body)
        assert status == 200 and headers["Content-Type"].startswith("application/json")
        assert answer["global"]["btc_dominance_pct"] == 59.1
        assert (answer["news"], answer["calendar"], answer["fear_greed"], answer["macro"]) == (None, None, None, None)

    class Broken:
        def get(self) -> dict[str, Any]:
            raise RuntimeError("something unexpected")

    for base in _serving(context=Broken()):
        assert _get(base + "/api/context")[::2] == (200, b"{}")
    for base in _serving():  # with the outside data turned off: empty, not an error
        assert _get(base + "/api/context")[::2] == (200, b"{}")


def test_the_page_may_only_connect_out_to_the_public_stream() -> None:
    for base in _serving(stream="wss://fstream.binance.com"):
        policy = _get(base + "/")[1]["Content-Security-Policy"]
        assert "connect-src 'self' wss://fstream.binance.com;" in policy and "script-src 'self';" in policy
    for base in _serving():
        policy = _get(base + "/")[1]["Content-Security-Policy"]
        assert "connect-src 'self';" in policy and "wss:" not in policy


def test_candles_endpoint_serves_the_interval_asked_for() -> None:
    asked: list[tuple[str, str]] = []

    def feed(symbol: str, interval: str) -> dict[str, list[float]]:
        asked.append((symbol, interval))
        return {"time": [1_791_800_000, 1_791_800_300], "open": [1.0, 1.1], "high": [1.2, 1.3], "low": [0.9, 1.0],
                "close": [1.1, 1.25], "volume": [5_000.0, 7_500.0]}

    for base in _candles_server(feed):
        status, headers, body = _get(f"{base}/api/candles?symbol=1000PEPEUSDT&interval=5m")
        assert status == 200 and headers["Content-Type"].startswith("application/json")
        answer = json.loads(body)
        assert (answer["symbol"], answer["interval"], answer["close"]) == ("1000PEPEUSDT", "5m", [1.1, 1.25])
        _get(f"{base}/api/candles?symbol=1000PEPEUSDT&interval=5m")
        _get(f"{base}/api/candles?symbol=1000PEPEUSDT&interval=1h")
        assert asked == [("1000PEPEUSDT", "5m"), ("1000PEPEUSDT", "1h")]  # the repeated one comes from the cache


@pytest.mark.parametrize("query", [
    "symbol=BTCUSDT&interval=1s", "symbol=BTCUSDT&interval=", "symbol=BTCUSD&interval=5m", "symbol=btcusdt&interval=5m",
    "symbol=BTCUSDT%26limit%3D1500&interval=5m", "symbol=..%2F..%2Fx&interval=5m", "interval=5m", "",
])
def test_candles_endpoint_refuses_what_it_does_not_serve(query: str) -> None:
    asked: list[Any] = []
    for base in _candles_server(lambda symbol, interval: asked.append(symbol) or {}):
        assert _get(f"{base}/api/candles?{query}")[0] == 400
    assert asked == []  # nothing that was refused gets requested outside


def test_candles_endpoint_reports_a_failed_read_without_details() -> None:
    def feed(symbol: str, interval: str) -> dict[str, list[float]]:
        raise OSError("https://secret.example/internal-path")

    log_lines: list[str] = []
    sink = logger.add(log_lines.append, format="{level}|{message}")
    try:
        for base in _candles_server(feed):
            status, _, body = _get(f"{base}/api/candles?symbol=BTCUSDT&interval=5m")
            assert status == 502 and body.decode() == "could not read the candles"
    finally:
        logger.remove(sink)
    assert any(line.startswith("WARNING|") for line in log_lines) and not any("secret" in line for line in log_lines)
    for base in _candles_server(None):
        assert _get(f"{base}/api/candles?symbol=BTCUSDT&interval=5m")[0] == 404


def test_a_failing_snapshot_is_reported_without_details_and_not_cached(served: tuple[str, _CountingPanel]) -> None:
    base, panel = served
    panel.error = RuntimeError("/app/data/unreadable disk")
    status, _, body = _get(base + "/api/snapshot")
    assert status == 500 and json.loads(body) == {"error": "could not build the dashboard"}
    panel.error = None
    assert _get(base + "/api/snapshot")[0] == 200 and panel.calls == 2


def test_a_stale_snapshot_is_served_while_one_request_rebuilds_it(traded: Path) -> None:
    class SlowPanel(_CountingPanel):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.gate: threading.Event | None = None
            self.waiting = threading.Event()

        def snapshot(self) -> dict[str, Any]:
            if self.gate is not None:
                self.waiting.set()
                self.gate.wait(timeout=5)
            return super().snapshot()

    panel = SlowPanel(traded)
    server = BasketDashboardServer(("127.0.0.1", 0), panel, cache_seconds=0.0)
    first = server.snapshot()
    panel.gate = threading.Event()
    rebuilt: list[bytes] = []
    rebuilding = threading.Thread(target=lambda: rebuilt.append(server.snapshot()))
    rebuilding.start()
    try:
        assert panel.waiting.wait(timeout=5)  # the rebuild has started and is waiting on Binance
        started = time.monotonic()
        assert server.snapshot() == first and time.monotonic() - started < 1.0
    finally:
        panel.gate.set()
        rebuilding.join(timeout=5)
        server.server_close()

    panel.gate, panel.error = None, RuntimeError("Binance down")
    assert len(rebuilt) == 1
    assert server.snapshot() == rebuilt[0]  # a failed rebuild keeps serving the last good one
    assert panel.calls == 3


@pytest.mark.parametrize(("host", "allowed"), [
    ("127.0.0.1:8770", True), ("localhost:9000", True), ("LOCALHOST", True), ("[::1]:8770", True),
    ("localhost.:8770", True), ("evil.example:8770", False), ("127.0.0.1.evil.example", False), ("", False),
])
def test_only_requests_addressed_to_this_machine_are_answered(
    served: tuple[str, _CountingPanel], host: str, allowed: bool,
) -> None:
    base, _ = served
    request = urllib.request.Request(base + "/api/snapshot", headers={"Host": host})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    assert (status == 200) is allowed and (allowed or status == 403)


def test_an_unreadable_upstream_answer_is_a_gateway_error_not_a_bad_request() -> None:
    def feed(symbol: str) -> dict[str, Any]:
        raise json.JSONDecodeError("Expecting value", "<html>", 0)

    for base in _serving(derivatives=LiveBySymbol(feed, ttl_seconds=60.0)):
        assert _get(f"{base}/api/derivatives?symbol=NEARUSDT")[0] == 502


def test_report_download(served: tuple[str, _CountingPanel], traded: Path) -> None:
    base, _ = served
    markdown_path, json_path = report_paths(traded / "reports" / "dry_run", NEXT_DAY)

    status, headers, body = _get(base + "/api/report.md")
    assert status == 200 and headers["Content-Type"].startswith("text/markdown")
    assert headers["Content-Disposition"] == f'attachment; filename="daily_audit_{NEXT_DAY}.md"'
    assert body == markdown_path.read_bytes()

    status, headers, body = _get(f"{base}/api/report.json?date={NEXT_DAY}")
    assert status == 200 and headers["Content-Type"].startswith("application/json") and body == json_path.read_bytes()

    status, headers, body = _get(f"{base}/api/report.md?date={DAY}")
    assert status == 200 and headers["Content-Disposition"].endswith(f'daily_audit_{DAY}.md"')


@pytest.mark.parametrize("query", ["date=2026-10-11", "date=..%2F..%2Fpaper_futures", "date=%20", "date=%2Fetc%2Fpasswd"])
def test_report_download_refuses_unknown_dates(served: tuple[str, _CountingPanel], query: str) -> None:
    status, headers, body = _get(f"{served[0]}/api/report.md?{query}")
    assert status == 404 and headers["Content-Type"].startswith("text/plain")
    assert body == "no report for that date".encode()


def test_health_endpoint(served: tuple[str, _CountingPanel]) -> None:
    status, _, body = _get(served[0] + "/healthz")
    assert (status, body) == (200, b"ok")


# --- Startup -----------------------------------------------------------------


def test_serve_in_background_answers_from_a_thread(traded: Path) -> None:
    settings = Settings(_env_file=None, state_dir=traded, futures={"dashboard_port": _free_port()})
    server = dashboard.serve_in_background(settings, "dry_run")
    assert server is not None
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert _get(base + "/healthz")[0] == 200
        assert json.loads(_get(base + "/api/snapshot")[2])["report"]["cycle_date"] == NEXT_DAY
    finally:
        server.shutdown()
        server.server_close()


def test_a_busy_port_leaves_the_cycle_without_panel_but_running(traded: Path) -> None:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}")
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        settings = Settings(_env_file=None, state_dir=traded, futures={"dashboard_port": taken.getsockname()[1]})
        try:
            assert dashboard.serve_in_background(settings, "dry_run") is None
        finally:
            logger.remove(sink)
    assert any(line.startswith("ERROR|The basket dashboard could not start") for line in lines)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_main_reads_the_dry_run_files_and_listens_on_localhost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: dict[str, Any] = {}

    class Stub:
        def __init__(self, address: tuple[str, int], panel: BasketPanel, **options: Any) -> None:
            self.server_address = address
            built.update(address=address, panel=panel)

        def serve_forever(self) -> None: ...

        def server_close(self) -> None: ...

        def shutdown(self) -> None: ...

    settings = Settings(_env_file=None, state_dir=tmp_path, futures={"execution_mode": "LIVE"})
    monkeypatch.setattr(dashboard, "get_settings", lambda: settings)
    monkeypatch.setattr(dashboard, "configure_logging", lambda _: None)
    monkeypatch.setattr(dashboard, "BasketDashboardServer", Stub)
    monkeypatch.setattr(dashboard.signal, "signal", lambda *_: None)

    assert dashboard.main([]) == 0
    assert built["address"] == ("127.0.0.1", 8770) and built["panel"]._state_file.parent == tmp_path
    # Unacknowledged LIVE runs in DRY_RUN: the dashboard looks at the files of that account.
    assert built["panel"]._state_file == tmp_path / "runtime_state_dry_run.json"
    assert built["panel"]._reports == tmp_path / "reports" / "dry_run"
    assert built["panel"]._paper_file == tmp_path / "paper_futures.json"

    assert dashboard.main(["--port", "9001"]) == 0 and built["address"] == ("127.0.0.1", 9001)


def test_main_follows_the_testnet_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict[str, Any] = {}

    class Stub:
        def __init__(self, address: tuple[str, int], panel: BasketPanel, **options: Any) -> None:
            self.server_address = address
            built.update(panel=panel, options=options)

        def serve_forever(self) -> None: ...

        def server_close(self) -> None: ...

    settings = Settings(_env_file=None, state_dir=tmp_path, futures={"execution_mode": "TESTNET", "dashboard_port": 8800, "run_delay_seconds": 600})
    monkeypatch.setattr(dashboard, "get_settings", lambda: settings)
    monkeypatch.setattr(dashboard, "configure_logging", lambda _: None)
    monkeypatch.setattr(dashboard, "BasketDashboardServer", Stub)
    monkeypatch.setattr(dashboard.signal, "signal", lambda *_: None)
    assert dashboard.main([]) == 0
    assert built["panel"]._reports == tmp_path / "reports" / "testnet" and built["panel"]._paper_file is None
    # The next cycle is announced with the configured delay.
    assert built["panel"]._run_delay == timedelta(minutes=10)
    assert isinstance(built["panel"]._quotes, LiveQuotes)
    # Each account reads its own network; the outside data and the live stream follow the default setting.
    assert built["panel"]._stream == "wss://fstream.binancefuture.com" == built["options"]["stream"]
    assert isinstance(built["options"]["context"], context.MarketContext)
    assert isinstance(built["options"]["derivatives"], LiveBySymbol) and isinstance(built["options"]["book"], LiveBySymbol)

    private = Settings(_env_file=None, state_dir=tmp_path, futures={"panel_external_data": False})
    monkeypatch.setattr(dashboard, "get_settings", lambda: private)
    assert dashboard.main([]) == 0
    # Without outside data: no context and no browser connection; what the server asks Binance for goes on.
    assert (built["panel"]._stream, built["options"]["context"], built["options"]["stream"]) == (None, None, None)
    assert isinstance(built["options"]["candles"], LiveCandles)


def test_a_client_that_leaves_mid_answer_is_not_an_error(served: tuple[str, _CountingPanel]) -> None:
    base, _ = served
    port = int(base.rsplit(":", 1)[1])
    log_lines: list[str] = []
    sink = logger.add(log_lines.append, format="{level}|{message}", level="WARNING")
    try:
        for _ in range(5):
            with socket.create_connection(("127.0.0.1", port)) as connection:
                connection.sendall(b"GET /api/snapshot HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
        assert _get(base + "/api/snapshot")[0] == 200
    finally:
        logger.remove(sink)
    assert log_lines == []
