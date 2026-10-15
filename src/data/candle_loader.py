"""Closed daily candles of the basket, checked before they are handed over.

The strategy decides on the last closed candle of each symbol. This module
makes sure that candle really is the one of the last full day and that none is
missing behind it. A symbol that fails this is skipped with a warning, and the
rest of the cycle goes on.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np
import pandas as pd
from loguru import logger

from src.execution.errors import BrokerError

DAY_MS = 86_400_000
CANDLE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time"]


class KlineSource(Protocol):
    """Where the candles come from; ``BinanceFuturesExecutor`` satisfies it."""

    def server_time_ms(self) -> int: ...

    def fetch_daily_klines(self, symbol: str, limit: int) -> list[list[Any]]: ...


def last_closed_day_ms(now_ms: int) -> int:
    """Open time (ms, UTC) of the last daily candle already closed at the instant ``now_ms``."""
    return (now_ms // DAY_MS - 1) * DAY_MS


class DailyCandleLoader:
    """Downloads the closed daily candles of each symbol and drops the series that cannot be trusted."""

    def __init__(self, source: KlineSource, candles: int = 500) -> None:
        """
        Args:
            source: the one that talks to the exchange.
            candles: closed candles delivered per symbol. The strategy's ATR
                is computed over all the ones it receives, so it must be the
                same number it was measured with (500). A symbol with less
                history than that is skipped.
        """
        if isinstance(candles, bool) or not isinstance(candles, int) or candles < 2:
            raise ValueError(f"candles must be an integer >= 2, got {candles!r}")
        self._source = source
        self._candles = candles

    def load(self, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
        """Closed candles per symbol, from the oldest to the most recent.

        Returns:
            ``{symbol: DataFrame}`` with ``open_time``, ``open``, ``high``,
            ``low``, ``close``, ``volume`` and ``close_time``. Only the
            symbols whose series passes the checks are included.

        Raises:
            BrokerError: if the exchange time cannot be read. Without it
                there is no way to know which candle is closed.
        """
        now = self._source.server_time_ms()
        expected_open = last_closed_day_ms(now)
        loaded: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                # One extra: the most recent one the exchange returns is the current day's.
                raw = self._source.fetch_daily_klines(symbol, self._candles + 1)
            except BrokerError as error:
                logger.warning("{}: no daily candles, {}", symbol, error)
                continue
            try:
                frame = _to_frame(raw)
            except (ValueError, TypeError, IndexError) as error:
                logger.warning("{}: unreadable daily candles ({}); skipped", symbol, error)
                continue
            closed = frame[frame["close_time"] < now].reset_index(drop=True)
            problem = _problem(closed, expected_open, self._candles)
            if problem:
                logger.warning("{}: daily candles dropped, {}", symbol, problem)
                continue
            loaded[symbol] = closed.iloc[-self._candles :].reset_index(drop=True)
        return loaded


def _to_frame(raw: list[list[Any]]) -> pd.DataFrame:
    """The candles as Binance gives them, with the common columns of the project."""
    rows = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), int(k[6])) for k in raw]
    return pd.DataFrame(rows, columns=CANDLE_COLUMNS)


def _problem(closed: pd.DataFrame, expected_open: int, wanted: int) -> str | None:
    """Why a series of closed candles is no good; ``None`` if it is fine."""
    if closed.empty:
        return "there is no closed candle"
    opens = closed["open_time"].to_numpy(dtype=np.int64)
    closes = closed["close_time"].to_numpy(dtype=np.int64)
    if (opens % DAY_MS != 0).any() or (closes - opens != DAY_MS - 1).any():
        return "some candles are not a full UTC day"
    last = int(opens[-1])
    if last != expected_open:
        return (
            f"the last closed candle is from {_day(last)} and should be from {_day(expected_open)}"
        )
    steps = np.diff(opens)
    if (steps != DAY_MS).any():
        gap = int(np.argmax(steps != DAY_MS))
        return f"there is a gap between {_day(int(opens[gap]))} and {_day(int(opens[gap + 1]))}"
    if len(closed) < wanted:
        return f"there are only {len(closed)} closed candles and {wanted} are needed"
    return None


def _day(open_ms: int) -> str:
    return str(pd.to_datetime(open_ms, unit="ms").date())
