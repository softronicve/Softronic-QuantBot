"""Tests of the quant floor (``src/ui/floor.py``) and the ``/api/floor`` endpoint."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from src.ui.floor import ALERT, IDLE, OK, WATCH, quant_floor
from tests.test_basket_dashboard import _CountingPanel, _get, _panel, _quoted, served, traded  # noqa: F401 (fixtures)
from tests.test_daily_reporter import _reporter, _snapshot


def _report(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return _reporter(tmp_path).build(_snapshot(**overrides))


def _live(symbol: str = "AAAUSDT", price: float = 105.0, stop: float = 95.0, channel: float = 90.0) -> dict[str, Any]:
    return {"symbol": symbol, "quantity": 10.0, "entry_price": 101.0, "price": price, "stop_price": stop,
            "exit_channel": channel, "pnl_usd": 10.0 * (price - 101.0), "status": "HOLD"}


def test_without_a_report_every_desk_is_idle() -> None:
    floor = quant_floor(None, [], {}, [])
    assert floor["cycle_date"] is None and {desk["light"] for desk in floor["desks"].values()} == {IDLE}


def test_the_five_desks_carry_the_cycle_figures(tmp_path: Path) -> None:
    check = {"positions": 2, "protected": 2, "repaired": [], "uncovered": [], "unprotected": []}
    report = _report(tmp_path, stop_check=check)
    history = [{"cycle_date": report["cycle_date"], "completed_at": "2026-10-09T00:01:45+00:00"}]
    radar = {"rows": {"AAAUSDT": {"price": 105.0, "entry_channel": 103.0}}}
    desks = quant_floor(report, [_live()], radar, history)["desks"]

    assert desks["data"]["completed_at"] == "2026-10-09T00:01:45+00:00"
    assert desks["data"]["symbols_with_data"] == desks["data"]["symbols"] == 4
    assert desks["signal"]["regime"] == report["observations"]["regime"]
    strengths = [row["strength"] for row in desks["signal"]["top"]]
    assert strengths == sorted(strengths, reverse=True) and len(strengths) <= 5
    assert all(row["symbol"] != "AAAUSDT" for row in desks["signal"]["top"])  # held positions are not candidates
    risk = desks["risk"]
    assert risk["heat_pct"] == report["account"]["heat_pct"] and risk["stops_ok"] is risk["stops_checked"] is True
    assert risk["stops"] == "OK (2 of 2 positions protected)"
    (position,) = desks["execution"]["positions"]
    # The stop (95) is above the channel (90): it is the level that would sell first.
    assert position["nearest_exit"] == 95.0 and position["nearest_exit_is_stop"] is True
    assert position["exit_distance_pct"] == pytest.approx((95.0 / 105.0 - 1.0) * 100.0)
    assert position["high_entry"] == 103.0 and position["r_multiple"] == pytest.approx(40.0 / 60.0)
    assert [prompt["title"] for prompt in desks["audit"]["prompts"]] == [
        "Volatility anomalies", "Correlation and concentration", "Execution quality"]
    json.dumps(desks, allow_nan=False)


def test_lights(tmp_path: Path) -> None:
    report = _report(tmp_path, stop_check={"positions": 1, "protected": 0, "repaired": [], "uncovered": [],
                                           "unprotected": ["AAAUSDT"]})
    desks = quant_floor(report, [_live()], {}, [])["desks"]
    assert desks["risk"]["light"] == ALERT and desks["risk"]["stops_ok"] is False

    hot = copy.deepcopy(_report(tmp_path))
    hot["account"]["heat_pct"] = 3.5  # 87 % of the 4 % limit
    assert quant_floor(hot, [], {}, [])["desks"]["risk"]["light"] == WATCH
    assert quant_floor(hot, [], {}, [])["desks"]["execution"]["light"] == IDLE

    near = quant_floor(_report(tmp_path), [_live(price=96.0)], {}, [])["desks"]["execution"]
    assert near["light"] == WATCH  # one percent above its stop
    assert quant_floor(_report(tmp_path), [_live(price=120.0)], {}, [])["desks"]["execution"]["light"] == OK

    failed = copy.deepcopy(_report(tmp_path))
    failed["status"] = "INCOMPLETE"
    desks = quant_floor(failed, [], {}, [])["desks"]
    assert desks["data"]["light"] == ALERT and desks["audit"]["light"] == WATCH
    audited = copy.deepcopy(_report(tmp_path))
    audited["llm_audit"] = {"model": "m", "answers": []}
    assert quant_floor(audited, [], {}, [])["desks"]["audit"]["light"] == OK


def test_a_report_from_before_these_figures_still_builds_the_floor(tmp_path: Path) -> None:
    old = copy.deepcopy(_report(tmp_path))
    for key in ("closed_trades", "stop_check"):
        old.pop(key)
    for key in ("regime", "benchmarks", "heat_discards"):
        old["observations"].pop(key)
    desks = quant_floor(old, [_live()], {}, [])["desks"]
    assert desks["risk"]["stops"] == "not checked" and desks["execution"]["settlement"] == []
    assert desks["signal"]["regime"] == {} and desks["signal"]["light"] == IDLE
    assert desks["risk"]["stops_checked"] is False and desks["risk"]["light"] != ALERT


def test_the_floor_endpoint_serves_the_desks_of_the_cached_snapshot(served: tuple[str, _CountingPanel]) -> None:  # noqa: F811
    base, panel = served
    status, headers, body = _get(base + "/api/floor")
    floor = json.loads(body)
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert set(floor["desks"]) == {"data", "signal", "risk", "execution", "audit"}
    snapshot = json.loads(_get(base + "/api/snapshot")[2])
    assert snapshot["floor"] == floor and panel.calls == 1  # one composition serves both


def test_the_page_offers_the_floor_next_to_the_inspector(served: tuple[str, _CountingPanel]) -> None:  # noqa: F811
    page = _get(served[0] + "/")[2]
    for anchor in (b'id="inspector-views"', b'data-view="floor"', b'id="floor-room"', b'id="floor-detail"',
                   b'/static/floor.js', "Quant floor".encode()):
        assert anchor in page


def test_the_panel_snapshot_includes_the_floor_with_live_prices(traded: Path) -> None:  # noqa: F811
    snapshot = _panel(traded, quotes=_quoted(110.0)).snapshot()
    (position,) = snapshot["floor"]["desks"]["execution"]["positions"]
    assert position["price"] == 110.0 and position["stop_price"] == snapshot["positions"][0]["stop_price"]


def test_the_signal_desk_carries_turbulence_and_the_edge_ratio(tmp_path: Path) -> None:
    report = _report(tmp_path)
    report["observations"]["turbulence"] = {"value": 1.2, "raw": 4.8, "coins": 4, "lookback": 252,
                                            "history_days": 40, "percentile": 97.5}
    signal = quant_floor(report, [], {}, [])["desks"]["signal"]
    assert signal["turbulence"]["percentile"] == 97.5 and "edge_ratio" in signal
