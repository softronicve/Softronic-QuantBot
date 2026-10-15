"""Tests of the record of executed cycles (``DailyStateStore``)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from src.engine.state_store import COMPLETED, STARTED, DailyStateStore, StateError

DAY = "2026-10-09"


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "runtime_state.json"


def test_a_new_store_has_executed_nothing(path: Path) -> None:
    store = DailyStateStore(path)
    assert not store.is_cycle_already_executed(DAY)
    assert store.get_cycle(DAY) is None and store.dispatched(DAY) == {}
    assert not path.exists()


def test_a_started_cycle_is_not_executed_yet(path: Path) -> None:
    store = DailyStateStore(path)
    store.mark_cycle_started(DAY)
    assert not store.is_cycle_already_executed(DAY)
    assert store.get_cycle(DAY)["status"] == STARTED


def test_a_completed_cycle_is_executed_and_keeps_its_summary(path: Path) -> None:
    store = DailyStateStore(path)
    store.mark_cycle_completed(DAY, {"opens": 2, "closes": ["BTCUSDT"]})
    assert store.is_cycle_already_executed(DAY)
    assert not store.is_cycle_already_executed("2026-10-10")
    cycle = store.get_cycle(DAY)
    assert cycle["status"] == COMPLETED and cycle["summary"] == {"opens": 2, "closes": ["BTCUSDT"]}
    assert cycle["started_at"] <= cycle["completed_at"]


def test_state_survives_a_restart(path: Path) -> None:
    first = DailyStateStore(path)
    first.record_dispatch(DAY, "BTCUSDT", "OPEN_LONG", {"status": "FILLED", "quantity": 0.012})
    first.mark_cycle_completed(DAY, {"ok": True})

    again = DailyStateStore(path)
    assert again.is_cycle_already_executed(DAY)
    assert again.dispatched(DAY) == {"BTCUSDT:OPEN_LONG": {"status": "FILLED", "quantity": 0.012}}
    assert not path.with_suffix(".tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_each_dispatch_is_written_as_soon_as_it_is_recorded(path: Path) -> None:
    store = DailyStateStore(path)
    store.record_dispatch(DAY, "BTCUSDT", "CLOSE_LONG", {"status": "FILLED"})
    # Without completing the cycle: if the process died here, the order is already on record.
    survivor = DailyStateStore(path)
    assert not survivor.is_cycle_already_executed(DAY)
    assert list(survivor.dispatched(DAY)) == ["BTCUSDT:CLOSE_LONG"]


def test_starting_twice_does_not_erase_what_was_dispatched(path: Path) -> None:
    store = DailyStateStore(path)
    store.record_dispatch(DAY, "BTCUSDT", "OPEN_LONG", {"status": "FILLED"})
    store.mark_cycle_started(DAY)
    assert list(store.dispatched(DAY)) == ["BTCUSDT:OPEN_LONG"]


def test_what_the_store_returns_cannot_change_it(path: Path) -> None:
    store = DailyStateStore(path)
    store.record_dispatch(DAY, "BTCUSDT", "OPEN_LONG", {"status": "FILLED"})
    store.dispatched(DAY)["BTCUSDT:OPEN_LONG"]["status"] = "BROKEN"
    store.get_cycle(DAY)["status"] = COMPLETED
    assert store.dispatched(DAY)["BTCUSDT:OPEN_LONG"]["status"] == "FILLED"
    assert not store.is_cycle_already_executed(DAY)


@pytest.mark.parametrize("content", ["", "{ broken", "[]", '{"other": 1}', '{"cycles": [1, 2]}'])
def test_an_unreadable_file_is_an_error_not_an_empty_state(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(StateError, match="cannot be read"):
        DailyStateStore(path)


@pytest.mark.parametrize("cycle_date", ["", "today", "2026-13-01", "2026-10-9", "09/10/2026", "20261009", None, 20261009])
def test_cycle_dates_must_be_iso_dates(path: Path, cycle_date: Any) -> None:
    store = DailyStateStore(path)
    for call in (store.is_cycle_already_executed, store.mark_cycle_started, store.get_cycle):
        with pytest.raises(StateError, match="YYYY-MM-DD"):
            call(cycle_date)
    with pytest.raises(StateError, match="YYYY-MM-DD"):
        store.mark_cycle_completed(cycle_date, {})
    assert not path.exists()


def test_values_json_cannot_hold_are_saved_as_text(path: Path) -> None:
    store = DailyStateStore(path)
    store.mark_cycle_completed(DAY, {"quantity": Decimal("0.012"), "where": Path("data")})
    assert DailyStateStore(path).get_cycle(DAY)["summary"] == {"quantity": "0.012", "where": "data"}


def test_only_the_most_recent_cycles_are_kept(path: Path) -> None:
    store = DailyStateStore(path, keep_cycles=3)
    for day in range(1, 8):
        store.mark_cycle_completed(f"2026-10-{day:02d}", {})
    assert sorted(json.loads(path.read_text(encoding="utf-8"))["cycles"]) == ["2026-10-05", "2026-10-06", "2026-10-07"]
    assert store.is_cycle_already_executed("2026-10-07") and not store.is_cycle_already_executed("2026-10-01")
