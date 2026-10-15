"""20/10 channel breakout, long only, sized by ``ATRSizer`` and cleared by the portfolio.

It buys when the close is above the high of the previous 20 candles and exits
when the close falls below the low of the previous 10. The initial stop sits
2 ATR from the entry. ``ATRSizer`` sets the size and, before proposing the
order, ``PortfolioRiskManager`` is asked whether the portfolio can take one
more position.

What backs this rule and what does not: in the backtest on DAILY candles of
twelve cryptocurrencies (2019-2026, long only, 0.10 % fee per side, one
account per coin) the basket gained 41 %, against 21 % for holding the same
exposure without trading and 23 % for entering at random with the same exits
(p = 0.002). This module reproduces those same 419 trades when the portfolio
limits do not step in. It is a historical result, concentrated in 2020-21 and
2023-24, measured on coins that are still large today (the ones that
disappeared are not in the sample) and not yet validated going forward. On
one-hour candles the same rule did not beat the passive portfolio, and in
forex it gave no edge on any timeframe.

On daily candles the 2 ATR stop is far from the price: a position of 10 % of
the balance risks between 0.5 % and 1 %. With the default limits of
``PortfolioRiskManager`` open risk (4 %) runs out with four to seven
positions, before the notional does (ten).

The strategy only decides: it returns what it would do and why. It sends no
orders, does not watch the stop (that level travels with the order and the
broker executes it) and keeps no state between calls.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from src.risk.portfolio_risk import ASSET_CLASSES, ActivePosition, PortfolioRiskManager
from src.risk.position_sizer import ATRSizer
from src.risk.turtle_engine import wilder_atr

FloatArray = NDArray[np.float64]

PRICE_COLUMNS = ("open", "high", "low", "close")
LONG = 1

# Reasons for a decision. Portfolio rejections come with their own code.
BREAKOUT = "BREAKOUT"
EXIT_CHANNEL_BROKEN = "EXIT_CHANNEL_BROKEN"
NO_BREAKOUT = "NO_BREAKOUT"
TREND_INTACT = "TREND_INTACT"
SIZE_TOO_SMALL = "SIZE_TOO_SMALL"
STOP_NOT_POSITIVE = "STOP_NOT_POSITIVE"


class StrategyError(ValueError):
    """The strategy cannot be evaluated with the arguments given."""


class MarketDataError(StrategyError):
    """The candles cannot be trusted: they are missing, out of order or inconsistent."""


class Action(Enum):
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """What the strategy needs to know about the symbol to size an order."""

    symbol: str
    asset_class: str
    # Broker's volume step, in lots.
    step_size: float
    # Units of the asset per lot; 1 where there are no lots.
    contract_size: float = 1.0
    min_volume: float = 0.0
    max_volume: float | None = None
    # Tick value and size, if the asset is not quoted in the account currency.
    tick_value: float | None = None
    tick_size: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise StrategyError(f"symbol must be a non-empty string, got {self.symbol!r}")
        if self.asset_class not in ASSET_CLASSES:
            raise StrategyError(f"asset_class must be one of {ASSET_CLASSES}, got {self.asset_class!r}")


@dataclass(frozen=True, slots=True)
class ChannelLevels:
    """The four numbers the strategy decides with on the last closed candle."""

    close: float
    # High of the previous candles: closing above it is the entry signal.
    entry_channel: float
    # Low of the previous candles: closing below it is the exit signal.
    exit_channel: float
    atr: float

    @property
    def breakout_strength(self) -> float:
        """How far the close is above the entry channel, in ATR; negative if it is not above it."""
        return (self.close - self.entry_channel) / self.atr if self.atr > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class Decision:
    """What the strategy would do with the last closed candle, and the data it decided with."""

    action: Action
    symbol: str
    reason: str
    close: float
    # High of the previous candles that has to be beaten to enter.
    entry_channel: float
    # Low of the previous candles; closing below it exits the position.
    exit_channel: float
    atr: float
    # Only on an entry: lots, initial stop, value and risk in the account currency.
    volume: float = 0.0
    stop_price: float | None = None
    nominal_value: float = 0.0
    risk_amount: float = 0.0
    asset_class: str = ""

    def to_active_position(self) -> ActivePosition:
        """The position left open if the entry is filled, for the portfolio check."""
        if self.action is not Action.ENTER_LONG:
            raise StrategyError(f"only an entry opens a position; this decision is {self.action.value}")
        return ActivePosition(self.symbol, LONG, self.nominal_value, self.risk_amount, self.asset_class)


class TrendBreakoutStrategy:
    """Channel breakout, long only: decides entries and exits on closed candles."""

    def __init__(
        self,
        sizer: ATRSizer | None = None,
        portfolio: PortfolioRiskManager | None = None,
        *,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 100,
        stop_atr_multiple: float = 2.0,
        risk_pct: float = 1.0,
        max_leverage: float = 0.1,
        max_log_return: float | None = None,
    ) -> None:
        """
        Args:
            sizer: what computes the size; by default, an ``ATRSizer``.
            portfolio: what approves the entry; by default, a
                ``PortfolioRiskManager`` with its stock limits.
            entry_period: candles in the entry channel.
            exit_period: candles in the exit channel.
            atr_period: candles in the ATR (Wilder smoothing). 100 is the value
                the strategy was measured with.
            stop_atr_multiple: ATRs between the entry and the initial stop.
            risk_pct: percentage of the balance lost if that stop is hit.
            max_leverage: maximum notional value of the position, in times the balance.
            max_log_return: if given, a candle whose close moves more than
                this (in log terms) from the previous one is treated as
                corrupt data. Not applied by default: on daily candles a
                10 % or 50 % move is market, not an error, and that is how
                the strategy was measured.
        """
        for name, value in (("entry_period", entry_period), ("exit_period", exit_period), ("atr_period", atr_period)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise StrategyError(f"{name} must be an integer >= 1, got {value!r}")
        if atr_period < 2:
            raise StrategyError(f"atr_period must be >= 2, got {atr_period}")
        for name, number in (("stop_atr_multiple", stop_atr_multiple), ("risk_pct", risk_pct),
                             ("max_leverage", max_leverage)):
            if not _is_number(number) or number <= 0.0:
                raise StrategyError(f"{name} must be a positive number, got {number!r}")
        if max_log_return is not None and (not _is_number(max_log_return) or max_log_return <= 0.0):
            raise StrategyError(f"max_log_return must be a positive number or None, got {max_log_return!r}")

        self._sizer = sizer if sizer is not None else ATRSizer()
        self._portfolio = portfolio if portfolio is not None else PortfolioRiskManager()
        self._entry_period = entry_period
        self._exit_period = exit_period
        self._atr_period = atr_period
        self._stop_atr_multiple = float(stop_atr_multiple)
        self._risk_pct = float(risk_pct)
        self._max_leverage = float(max_leverage)
        self._max_log_return = max_log_return

    @property
    def entry_period(self) -> int:
        return self._entry_period

    @property
    def exit_period(self) -> int:
        return self._exit_period

    @property
    def atr_period(self) -> int:
        return self._atr_period

    @property
    def stop_atr_multiple(self) -> float:
        return self._stop_atr_multiple

    @property
    def risk_pct(self) -> float:
        return self._risk_pct

    @property
    def max_leverage(self) -> float:
        return self._max_leverage

    @property
    def min_candles(self) -> int:
        """Closed candles needed: those for the channels and the ATR, plus the one evaluated."""
        return max(self._entry_period, self._exit_period, self._atr_period) + 1

    def protective_stop_distance(self, candles: pd.DataFrame) -> float:
        """Distance, in price, at which the strategy would place an entry's stop today.

        It is used to estimate the risk of a position found open with no stop
        at the broker.

        Raises:
            StrategyError, MarketDataError: as in :meth:`evaluate`.
        """
        return self.levels(candles).atr * self._stop_atr_multiple

    def levels(self, candles: pd.DataFrame) -> ChannelLevels:
        """Close, channels and ATR of the last candle, as :meth:`evaluate` uses them.

        Raises:
            StrategyError, MarketDataError: as in :meth:`evaluate`.
        """
        high, low, close = self._validated(candles)
        return ChannelLevels(
            close=float(close[-1]),
            entry_channel=float(high[-(self._entry_period + 1) : -1].max()),
            exit_channel=float(low[-(self._exit_period + 1) : -1].min()),
            atr=float(wilder_atr(high, low, close, self._atr_period)[-1]),
        )

    def evaluate(
        self,
        candles: pd.DataFrame,
        instrument: InstrumentSpec,
        *,
        account_balance: float,
        open_positions: Sequence[ActivePosition] = (),
        check_portfolio: bool = True,
    ) -> Decision:
        """Decides on the last candle of ``candles``.

        Args:
            candles: closed candles, from oldest to newest, with ``open``,
                ``high``, ``low`` and ``close``. The last one is the one
                evaluated; the candle in progress is never passed. The ATR is
                computed over all of them, so it is best to always pass the
                same number (the bot uses 500).
            instrument: trading rules of the symbol.
            account_balance: account balance.
            open_positions: every open position in the portfolio. If one is
                on this symbol, the strategy can only hold or close it.
            check_portfolio: with ``False`` the entry comes back already
                sized but without going through the portfolio. It is for
                whoever allocates risk across several candidates at once, who
                must ask the portfolio afterwards in the order they choose.

        Raises:
            StrategyError: if the arguments do not have the expected shape.
            MarketDataError: if the candles are missing or inconsistent.
            RiskError: if the balance or the symbol's rules do not allow
                sizing (raised by ``ATRSizer``).
        """
        if not isinstance(instrument, InstrumentSpec):
            raise StrategyError(f"instrument must be an InstrumentSpec, got {type(instrument).__name__}")
        levels = self.levels(candles)
        entry_channel, exit_channel, atr, last = levels.entry_channel, levels.exit_channel, levels.atr, levels.close

        def decide(action: Action, reason: str, **order: Any) -> Decision:
            return Decision(action, instrument.symbol, reason, last, entry_channel, exit_channel, atr,
                            asset_class=instrument.asset_class, **order)

        if any(position.symbol == instrument.symbol for position in open_positions):
            if last < exit_channel:
                logger.info("{}: exit, the close {:g} is below the low of {} candles ({:g})",
                            instrument.symbol, last, self._exit_period, exit_channel)
                return decide(Action.EXIT_LONG, EXIT_CHANNEL_BROKEN)
            return decide(Action.HOLD, TREND_INTACT)

        if not last > entry_channel:
            return decide(Action.HOLD, NO_BREAKOUT)

        sized = self._sizer.size_position(
            account_balance=account_balance, risk_pct=self._risk_pct, atr_val=atr,
            multiplier=self._stop_atr_multiple, contract_size=instrument.contract_size,
            step_size=instrument.step_size, asset_price=last, max_leverage=self._max_leverage,
            min_volume=instrument.min_volume, max_volume=instrument.max_volume,
            tick_value=instrument.tick_value, tick_size=instrument.tick_size,
        )
        stop_price = last - sized.stop_distance
        if stop_price <= 0.0:
            logger.warning("{}: breakout not traded, the stop at {:g} ATR would be at {:g}",
                           instrument.symbol, self._stop_atr_multiple, stop_price)
            return decide(Action.HOLD, STOP_NOT_POSITIVE)
        if sized.lots <= 0.0:
            return decide(Action.HOLD, SIZE_TOO_SMALL)

        if check_portfolio:
            approved, verdict = self._portfolio.can_open_position(
                account_balance, list(open_positions), sized.nominal_value, sized.risk_amount,
                instrument.symbol, LONG, instrument.asset_class,
            )
            if not approved:
                return decide(Action.HOLD, verdict)

        logger.info(
            "{}: buy {:g} lots, the close {:g} is above the high of {} candles ({:g}) | "
            "stop {:g} | notional {:.2f} | risk {:.2f}",
            instrument.symbol, sized.lots, last, self._entry_period, entry_channel, stop_price,
            sized.nominal_value, sized.risk_amount,
        )
        return decide(Action.ENTER_LONG, BREAKOUT, volume=sized.lots, stop_price=stop_price,
                      nominal_value=sized.nominal_value, risk_amount=sized.risk_amount)

    def _validated(self, candles: pd.DataFrame) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Highs, lows and closes, after checking that the candles are usable.

        It is a check of shape, not of market: there is no filter here whose
        state a jump could contaminate, so a large move is not discarded
        unless ``max_log_return`` asks for it.
        """
        if not isinstance(candles, pd.DataFrame):
            raise StrategyError(f"candles must be a DataFrame, got {type(candles).__name__}")
        missing = [column for column in PRICE_COLUMNS if column not in candles.columns]
        if missing:
            raise StrategyError(f"the candles are missing the columns {missing}")
        if len(candles) < self.min_candles:
            raise MarketDataError(f"{self.min_candles} candles required, got {len(candles)}")
        try:
            prices = candles[list(PRICE_COLUMNS)].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise MarketDataError(f"the candles have non-numeric prices: {error}") from error
        if not np.isfinite(prices).all():
            raise MarketDataError(f"the candles have non-finite prices (row {_first_row(~np.isfinite(prices))})")
        if (prices <= 0.0).any():
            raise MarketDataError(f"the candles have non-positive prices (row {_first_row(prices <= 0.0)})")
        open_, high, low, close = prices.T
        incoherent = (high < np.maximum(open_, close)) | (low > np.minimum(open_, close))
        if incoherent.any():
            raise MarketDataError(
                f"inconsistent candle at row {int(np.argmax(incoherent))}: the high or the low "
                "does not contain the open and the close"
            )
        if "close_time" in candles.columns:
            times = candles["close_time"].to_numpy()
            if (times[1:] <= times[:-1]).any():
                raise MarketDataError(
                    f"the candles are not in chronological order (row {int(np.argmax(times[1:] <= times[:-1])) + 1})"
                )
        if self._max_log_return is not None:
            jumps = np.abs(np.diff(np.log(close)))
            if (jumps > self._max_log_return).any():
                row = int(np.argmax(jumps > self._max_log_return)) + 1
                raise MarketDataError(
                    f"jump of {jumps[row - 1]:.4f} in the close at row {row}, above the "
                    f"allowed maximum ({self._max_log_return:g})"
                )
        return high, low, close


def _first_row(mask: NDArray[np.bool_]) -> int:
    return int(np.argmax(mask.any(axis=1)))


def _is_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(value)
    except TypeError:
        return False
