"""Module 4 (risk): volatility N, asymptotic size, 2N stop and Donchian exit.

Deterministic control layer: it contains no ML and it is the only one that
knows the account capital.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


class RiskError(ValueError):
    """The data does not allow a position to be sized safely."""


@dataclass(frozen=True, slots=True)
class PositionPlan:
    """Result of sizing one *Unit*."""

    units: float
    atr: float
    stop_price: float
    # True if the size was set by the capital cap and not by the risk rule.
    capped: bool


def wilder_atr(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int
) -> NDArray[np.float64]:
    """ATR with Wilder smoothing, candle by candle.

    The first ``period`` candles are ``NaN``: the True Range of each candle
    needs the previous close, and the average starts with ``period`` ranges.
    """
    hi = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    cl = np.asarray(close, dtype=np.float64)
    if not (hi.ndim == lo.ndim == cl.ndim == 1 and hi.shape == lo.shape == cl.shape):
        raise RiskError("high, low and close must be 1-D series of equal length")
    if cl.size < period + 1:
        raise RiskError(f"the ATR needs {period + 1} candles, got {cl.size}")
    if not (np.isfinite(hi).all() and np.isfinite(lo).all() and np.isfinite(cl).all()):
        raise RiskError("the candles contain non-finite values")

    previous_close = cl[:-1]
    true_range = np.maximum.reduce(
        [hi[1:] - lo[1:], np.abs(hi[1:] - previous_close), np.abs(lo[1:] - previous_close)]
    )
    series = np.full(cl.shape, np.nan)
    atr = float(true_range[:period].mean())
    series[period] = atr
    for index, value in enumerate(true_range[period:], start=period + 1):
        atr = (atr * (period - 1) + float(value)) / period
        series[index] = atr
    return series


class TurtleEngine:
    """Turtle rules: ``N = ATR(100)``, 1 % risk per Unit and the stop at ``2N``."""

    def __init__(
        self,
        atr_period: int = 100,
        risk_per_unit: float = 0.01,
        max_capital_fraction: float = 0.10,
        stop_atr_multiple: float = 2.0,
        exit_channel_period: int = 10,
    ) -> None:
        if atr_period < 2:
            raise ValueError(f"atr_period must be >= 2, got {atr_period}")
        if exit_channel_period < 1:
            raise ValueError(f"exit_channel_period must be >= 1, got {exit_channel_period}")
        if not 0.0 < risk_per_unit <= 1.0:
            raise ValueError(f"risk_per_unit must be in (0, 1], got {risk_per_unit}")
        if not 0.0 < max_capital_fraction <= 1.0:
            raise ValueError(
                f"max_capital_fraction must be in (0, 1], got {max_capital_fraction}"
            )
        if not stop_atr_multiple > 0.0:
            raise ValueError(f"stop_atr_multiple must be > 0, got {stop_atr_multiple}")

        self._atr_period = atr_period
        self._risk_per_unit = risk_per_unit
        self._max_capital_fraction = max_capital_fraction
        self._stop_atr_multiple = stop_atr_multiple
        self._exit_channel_period = exit_channel_period

    @property
    def min_candles(self) -> int:
        """Candles the full engine needs (ATR and exit channel)."""
        return max(self._atr_period, self._exit_channel_period) + 1

    @property
    def exit_channel_period(self) -> int:
        return self._exit_channel_period

    def atr(self, high: ArrayLike, low: ArrayLike, close: ArrayLike) -> float:
        """Computes ``N``: ATR with Wilder smoothing over ``atr_period`` candles."""
        return float(self.atr_series(high, low, close)[-1])

    def atr_series(self, high: ArrayLike, low: ArrayLike, close: ArrayLike) -> NDArray[np.float64]:
        """``N`` candle by candle; ``NaN`` until there are ``atr_period`` previous ranges."""
        return wilder_atr(high, low, close, self._atr_period)

    def position_size(self, capital: float, atr: float, price: float) -> float:
        """``min(capital * risk / ATR, capital * cap / price)``.

        The second term is the asymptotic cap: when ``ATR -> 0`` the risk rule
        diverges and the size stays bounded at ``max_capital_fraction`` of the
        capital.
        """
        self._require_positive("capital", capital)
        self._require_positive("price", price)
        if not math.isfinite(atr) or atr < 0.0:
            raise RiskError(f"invalid ATR: {atr}")

        cap_units = capital * self._max_capital_fraction / price
        if atr == 0.0:
            return cap_units
        return min(capital * self._risk_per_unit / atr, cap_units)

    def stop_price(self, entry_price: float, atr: float, side: str = "LONG") -> float:
        """Fixed stop loss at ``stop_atr_multiple * ATR`` from the entry.

        Below it for a buy; above it for a short sell.
        """
        self._require_positive("entry price", entry_price)
        if not math.isfinite(atr) or atr <= 0.0:
            raise RiskError(f"invalid ATR for setting the stop: {atr}")
        if side == "SHORT":
            return entry_price + self._stop_atr_multiple * atr
        stop = entry_price - self._stop_atr_multiple * atr
        if stop <= 0.0:
            raise RiskError(
                f"stop not positive ({stop:.8f}): volatility is too high for the price"
            )
        return stop

    def exit_channel(self, low: ArrayLike) -> float:
        """Donchian exit channel: lowest low of the candles before the last one.

        The last candle is left out on purpose: it is the one compared against
        the channel, and if it were part of it, it could never break it.
        """
        lows = np.asarray(low, dtype=np.float64)
        if lows.ndim != 1 or lows.size < self._exit_channel_period + 1:
            raise RiskError(
                f"the exit channel needs {self._exit_channel_period + 1} candles"
            )
        window = lows[-(self._exit_channel_period + 1) : -1]
        if not np.isfinite(window).all():
            raise RiskError("the exit channel candles contain non-finite values")
        return float(window.min())

    def exit_channel_series(self, low: ArrayLike) -> NDArray[np.float64]:
        """Exit channel candle by candle: lowest low of the previous ``exit_channel_period``."""
        lows = np.asarray(low, dtype=np.float64)
        series = np.full(lows.shape, np.nan)
        period = self._exit_channel_period
        if lows.size > period:
            windows = np.lib.stride_tricks.sliding_window_view(lows, period)
            series[period:] = windows[:-1].min(axis=1)
        return series

    def short_exit_channel(self, high: ArrayLike) -> float:
        """Exit channel of a short: highest high of the candles before the last one."""
        highs = np.asarray(high, dtype=np.float64)
        if highs.ndim != 1 or highs.size < self._exit_channel_period + 1:
            raise RiskError(
                f"the exit channel needs {self._exit_channel_period + 1} candles"
            )
        window = highs[-(self._exit_channel_period + 1) : -1]
        if not np.isfinite(window).all():
            raise RiskError("the exit channel candles contain non-finite values")
        return float(window.max())

    def short_exit_channel_series(self, high: ArrayLike) -> NDArray[np.float64]:
        """Exit channel of a short, candle by candle: highest high of the previous ones."""
        highs = np.asarray(high, dtype=np.float64)
        series = np.full(highs.shape, np.nan)
        period = self._exit_channel_period
        if highs.size > period:
            windows = np.lib.stride_tricks.sliding_window_view(highs, period)
            series[period:] = windows[:-1].max(axis=1)
        return series

    def breaks_exit_channel(self, low: ArrayLike, close: float) -> bool:
        """Turtle trend loss: the close breaks below the channel low."""
        return close < self.exit_channel(low)

    def plan(self, capital: float, price: float, atr: float, side: str = "LONG") -> PositionPlan:
        """Sizes a full Unit from the total capital and ``N``.

        The size is the same in both directions; only the side of the stop changes.
        """
        stop = self.stop_price(price, atr, side)
        units = self.position_size(capital, atr, price)
        capped = units < capital * self._risk_per_unit / atr
        return PositionPlan(units=units, atr=atr, stop_price=stop, capped=capped)

    @staticmethod
    def _require_positive(name: str, value: float) -> None:
        if not math.isfinite(value) or value <= 0.0:
            raise RiskError(f"invalid {name}: {value}")
