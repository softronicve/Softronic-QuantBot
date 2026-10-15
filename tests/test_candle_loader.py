"""Tests of the daily candle loader (``DailyCandleLoader``)."""

from __future__ import annotations

from typing import Any

import pytest
from loguru import logger

from src.data.candle_loader import DailyCandleLoader, last_closed_day_ms
from src.data.candle_loader import CANDLE_COLUMNS
from src.execution.errors import BrokerError
from tests.fake_futures import DAY_MS, NOW_MS, daily_klines

TODAY = (NOW_MS // DAY_MS) * DAY_MS
YESTERDAY = TODAY - DAY_MS


class Source:
    """The minimum the loader asks of the exchange."""

    def __init__(self, klines: dict[str, Any], now_ms: int = NOW_MS) -> None:
        self.klines = klines
        self.now_ms = now_ms
        self.asked: list[tuple[str, int]] = []

    def server_time_ms(self) -> int:
        return self.now_ms

    def fetch_daily_klines(self, symbol: str, limit: int) -> list[list[Any]]:
        self.asked.append((symbol, limit))
        series = self.klines[symbol]
        if isinstance(series, Exception):
            raise series
        return series[-limit:]


@pytest.fixture
def log() -> Any:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}")
    yield lines
    logger.remove(sink)


def test_last_closed_day_is_yesterday_utc() -> None:
    assert last_closed_day_ms(NOW_MS) == YESTERDAY
    assert last_closed_day_ms(TODAY) == YESTERDAY
    assert last_closed_day_ms(TODAY - 1) == YESTERDAY - DAY_MS


def test_loads_closed_candles_with_the_project_columns() -> None:
    source = Source({"BTCUSDT": daily_klines(104.0, n=150)})
    frame = DailyCandleLoader(source, candles=120).load(["BTCUSDT"])["BTCUSDT"]

    assert list(frame.columns) == CANDLE_COLUMNS
    assert len(frame) == 120 and list(frame.index) == list(range(120))
    assert frame["open_time"].iloc[-1] == YESTERDAY
    assert frame["close_time"].iloc[-1] == TODAY - 1
    assert frame["close"].iloc[-1] == pytest.approx(104.0)
    assert frame[["open", "high", "low", "close", "volume"]].dtypes.astype(str).tolist() == ["float64"] * 5
    assert source.asked == [("BTCUSDT", 121)]


def test_the_candle_still_forming_is_never_returned() -> None:
    forming = DailyCandleLoader(Source({"X": daily_klines(104.0, forming=True)}), candles=150).load(["X"])["X"]
    closed = DailyCandleLoader(Source({"X": daily_klines(104.0, forming=False)}), candles=150).load(["X"])["X"]
    assert forming.equals(closed)
    assert (forming["close_time"] < NOW_MS).all()


def test_a_candle_is_closed_only_once_its_close_time_has_passed() -> None:
    # One millisecond before midnight, the candle of "today" is still open.
    source = Source({"X": daily_klines(104.0, n=150, forming=False)}, now_ms=TODAY - 1)
    assert DailyCandleLoader(source, candles=149).load(["X"])["X"]["open_time"].iloc[-1] == YESTERDAY - DAY_MS


def test_stale_series_is_dropped(log: list[str]) -> None:
    stale = daily_klines(104.0, now_ms=NOW_MS - DAY_MS, forming=False)
    assert DailyCandleLoader(Source({"X": stale}), candles=100).load(["X"]) == {}
    assert any(line.startswith("WARNING|X: daily candles dropped") and "should be from 2026-10-08" in line
               for line in log)


def test_series_with_a_gap_is_dropped(log: list[str]) -> None:
    holed = daily_klines(104.0)
    del holed[70]
    assert DailyCandleLoader(Source({"X": holed}), candles=100).load(["X"]) == {}
    assert any("there is a gap between" in line for line in log)


def test_duplicated_or_unordered_candles_are_dropped() -> None:
    doubled = daily_klines(104.0)
    doubled.insert(60, doubled[60])
    swapped = daily_klines(104.0)
    swapped[100], swapped[101] = swapped[101], swapped[100]
    assert DailyCandleLoader(Source({"X": doubled, "Y": swapped}), candles=100).load(["X", "Y"]) == {}


def test_candles_that_are_not_whole_utc_days_are_dropped(log: list[str]) -> None:
    shifted = [[row[0] + 3_600_000, *row[1:6], row[6] + 3_600_000, *row[7:]] for row in daily_klines(104.0)]
    short = daily_klines(104.0)
    short[120][6] -= 1_000
    assert DailyCandleLoader(Source({"X": shifted, "Y": short}, now_ms=NOW_MS + 3_600_000), candles=100).load(
        ["X", "Y"]) == {}
    assert sum("are not a full UTC day" in line for line in log) == 2


def test_series_shorter_than_requested_is_dropped(log: list[str]) -> None:
    source = Source({"NEWUSDT": daily_klines(104.0, n=80)})
    assert DailyCandleLoader(source, candles=150).load(["NEWUSDT"]) == {}
    assert any("there are only 80 closed candles and 150 are needed" in line for line in log)


@pytest.mark.parametrize("raw", [[], [["this", "is", "no", "candle"]], [[1, "1", "1"]], None])
def test_empty_or_unreadable_answers_are_dropped(raw: Any) -> None:
    assert DailyCandleLoader(Source({"X": raw if raw is not None else [None]}), candles=10).load(["X"]) == {}


def test_one_symbol_failing_does_not_stop_the_rest(log: list[str]) -> None:
    source = Source({"AUSDT": daily_klines(104.0), "BUSDT": BrokerError("no network"), "CUSDT": daily_klines(100.0)})
    loaded = DailyCandleLoader(source, candles=150).load(["AUSDT", "BUSDT", "CUSDT"])
    assert list(loaded) == ["AUSDT", "CUSDT"]
    assert any(line.startswith("WARNING|BUSDT: no daily candles") for line in log)


def test_without_exchange_time_nothing_can_be_validated() -> None:
    source = Source({"X": daily_klines(104.0)})
    source.server_time_ms = lambda: (_ for _ in ()).throw(BrokerError("no network"))  # type: ignore[method-assign]
    with pytest.raises(BrokerError):
        DailyCandleLoader(source, candles=150).load(["X"])
    assert source.asked == []


@pytest.mark.parametrize("candles", [0, 1, -5, 2.5, True, "500"])
def test_invalid_candle_count_is_rejected(candles: Any) -> None:
    with pytest.raises(ValueError, match="candles"):
        DailyCandleLoader(Source({}), candles=candles)
