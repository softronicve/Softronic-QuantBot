"""Tests for the 20/10 breakout strategy (``TrendBreakoutStrategy``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src.risk.portfolio_risk import (
    CORRELATED_EXPOSURE_BLOCKED,
    MAX_HEAT_EXCEEDED,
    MAX_LEVERAGE_EXCEEDED,
    ActivePosition,
    PortfolioRiskManager,
)
from src.risk.position_sizer import ATRSizer, SizedPosition
from src.risk.turtle_engine import RiskError, wilder_atr
from src.strategies.trend_breakout import (
    BREAKOUT,
    EXIT_CHANNEL_BROKEN,
    NO_BREAKOUT,
    SIZE_TOO_SMALL,
    STOP_NOT_POSITIVE,
    TREND_INTACT,
    Action,
    InstrumentSpec,
    MarketDataError,
    StrategyError,
    TrendBreakoutStrategy,
)

DAY_MS = 86_400_000
BALANCE = 100_000.0
BTC = InstrumentSpec("BTCUSDT", "crypto", step_size=0.00001)
ETH = InstrumentSpec("ETHUSDT", "crypto", step_size=0.0001)
EURUSD = InstrumentSpec("EURUSD", "forex", step_size=0.01, contract_size=100_000.0, min_volume=0.01)
GBPUSD = InstrumentSpec("GBPUSD", "forex", step_size=0.01, contract_size=100_000.0, min_volume=0.01)


@pytest.fixture
def strategy() -> TrendBreakoutStrategy:
    return TrendBreakoutStrategy()


def _candles(closes: list[float] | np.ndarray, spread: float = 1.0) -> pd.DataFrame:
    """Daily candles with the given close; each one opens at the previous close."""
    close = np.asarray(closes, dtype=np.float64)
    open_ = np.concatenate([[close[0]], close[:-1]])
    open_time = np.arange(close.size, dtype=np.int64) * DAY_MS
    return pd.DataFrame({
        "open_time": open_time, "open": open_, "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread, "close": close, "volume": 1_000.0,
        "close_time": open_time + DAY_MS - 1,
    })


def _range(last: float, n: int = 150, level: float = 100.0, scale: float = 1.0) -> pd.DataFrame:
    """A sideways range between 98 and 102 (highs up to 103, lows down to 97) and a last candle at `last`.

    With ``scale=0.01`` the same prices sit around 1, like a currency pair.
    """
    wave = level + 2.0 * np.sin(np.arange(n - 1) * 0.7)
    candles = _candles([*wave, last])
    candles[["open", "high", "low", "close"]] *= scale
    return candles


def _held(symbol: str = "BTCUSDT", asset_class: str = "crypto", nominal: float = 10_000.0, risk: float = 100.0) -> ActivePosition:
    return ActivePosition(symbol, 1, nominal, risk, asset_class)


# --- Entry ------------------------------------------------------------------


def test_buys_when_the_close_breaks_the_twenty_candle_high(strategy: TrendBreakoutStrategy) -> None:
    candles = _range(last=104.0)
    decision = strategy.evaluate(candles, BTC, account_balance=BALANCE)
    high, low, close = (candles[c].to_numpy() for c in ("high", "low", "close"))
    atr = float(wilder_atr(high, low, close, 100)[-1])

    assert decision.action is Action.ENTER_LONG and decision.reason == BREAKOUT
    assert decision.symbol == "BTCUSDT" and decision.asset_class == "crypto"
    assert decision.close == 104.0
    assert decision.entry_channel == high[-21:-1].max() < 104.0
    assert decision.exit_channel == low[-11:-1].min()
    assert decision.atr == pytest.approx(atr)
    # Stop at 2 ATR and size from ATRSizer: here the 10 % balance cap decides.
    assert decision.stop_price == pytest.approx(104.0 - 2.0 * atr)
    assert decision.volume == pytest.approx(10_000.0 / 104.0, abs=1e-5)
    assert decision.nominal_value == pytest.approx(10_000.0, abs=0.01)
    assert decision.risk_amount == pytest.approx(decision.volume * 2.0 * atr)


def test_touching_the_channel_is_not_breaking_it(strategy: TrendBreakoutStrategy) -> None:
    channel = float(_range(last=100.0)["high"].iloc[-21:-1].max())
    assert strategy.evaluate(_range(last=channel), BTC, account_balance=BALANCE).reason == NO_BREAKOUT
    assert strategy.evaluate(_range(last=channel + 0.01), BTC, account_balance=BALANCE).action is Action.ENTER_LONG


def test_the_evaluated_candle_is_not_part_of_its_own_channel(strategy: TrendBreakoutStrategy) -> None:
    # The last candle makes a high of 111, but the channel only looks at the previous 20.
    decision = strategy.evaluate(_range(last=110.0), BTC, account_balance=BALANCE)
    assert decision.entry_channel < 104.0 and decision.action is Action.ENTER_LONG


def test_no_breakout_no_trade_and_never_a_short(strategy: TrendBreakoutStrategy) -> None:
    for last in (100.0, 97.5, 80.0):  # inside the range, below the exit channel and in a full drop
        decision = strategy.evaluate(_range(last=last), BTC, account_balance=BALANCE)
        assert (decision.action, decision.reason) == (Action.HOLD, NO_BREAKOUT)
        assert decision.volume == 0.0 and decision.stop_price is None
    assert {action.value for action in Action} == {"ENTER_LONG", "EXIT_LONG", "HOLD"}


def test_only_the_last_twenty_candles_form_the_entry_channel(strategy: TrendBreakoutStrategy) -> None:
    closes = np.full(150, 100.0)
    closes[100] = 130.0  # a spike 49 candles ago, outside the channel
    closes[-1] = 103.0
    assert strategy.evaluate(_candles(closes), BTC, account_balance=BALANCE).action is Action.ENTER_LONG
    closes[135] = 130.0  # the same spike 14 candles ago: inside
    assert strategy.evaluate(_candles(closes), BTC, account_balance=BALANCE).reason == NO_BREAKOUT


# --- Exit -------------------------------------------------------------------


def test_exits_when_the_close_loses_the_ten_candle_low(strategy: TrendBreakoutStrategy) -> None:
    held = [_held("BTCUSDT")]
    exit_channel = float(_range(last=100.0)["low"].iloc[-11:-1].min())
    falling = strategy.evaluate(_range(last=exit_channel - 0.01), BTC, account_balance=BALANCE, open_positions=held)
    assert (falling.action, falling.reason) == (Action.EXIT_LONG, EXIT_CHANNEL_BROKEN)
    touching = strategy.evaluate(_range(last=exit_channel), BTC, account_balance=BALANCE, open_positions=held)
    assert (touching.action, touching.reason) == (Action.HOLD, TREND_INTACT)


def test_while_in_position_a_new_breakout_does_not_add_to_it(strategy: TrendBreakoutStrategy) -> None:
    decision = strategy.evaluate(_range(last=110.0), BTC, account_balance=BALANCE, open_positions=[_held("BTCUSDT")])
    assert (decision.action, decision.reason) == (Action.HOLD, TREND_INTACT)
    # A position on another symbol does not count as being in this one.
    other = strategy.evaluate(_range(last=110.0), BTC, account_balance=BALANCE, open_positions=[_held("ETHUSDT")])
    assert other.action is Action.ENTER_LONG


def test_the_exit_channel_uses_ten_candles_and_the_entry_twenty() -> None:
    closes = np.full(150, 100.0)
    closes[134] = 90.0  # a low 15 candles ago: outside the exit channel, inside the entry one
    closes[-1] = 98.5
    held = [_held("BTCUSDT")]
    assert TrendBreakoutStrategy().evaluate(_candles(closes), BTC, account_balance=BALANCE, open_positions=held).action is Action.EXIT_LONG
    wide = TrendBreakoutStrategy(exit_period=20)
    assert wide.evaluate(_candles(closes), BTC, account_balance=BALANCE, open_positions=held).reason == TREND_INTACT


# --- Size: ATRSizer ---------------------------------------------------------


class SpySizer(ATRSizer):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def size_position(self, **arguments: Any) -> SizedPosition:
        self.calls.append(arguments)
        return super().size_position(**arguments)


def test_the_sizer_receives_the_strategy_and_instrument_parameters() -> None:
    sizer = SpySizer()
    instrument = InstrumentSpec("USDJPY", "forex", step_size=0.01, contract_size=100_000.0, min_volume=0.01,
                                max_volume=50.0, tick_value=0.66, tick_size=0.001)
    strategy = TrendBreakoutStrategy(sizer, stop_atr_multiple=3.0, risk_pct=0.5, max_leverage=0.2, atr_period=50)
    candles = _range(last=104.0)
    decision = strategy.evaluate(candles, instrument, account_balance=250_000.0)
    (call,) = sizer.calls
    assert call == {
        "account_balance": 250_000.0, "risk_pct": 0.5, "atr_val": decision.atr, "multiplier": 3.0,
        "contract_size": 100_000.0, "step_size": 0.01, "asset_price": 104.0, "max_leverage": 0.2,
        "min_volume": 0.01, "max_volume": 50.0, "tick_value": 0.66, "tick_size": 0.001,
    }
    assert decision.stop_price == pytest.approx(104.0 - 3.0 * decision.atr)
    high, low, close = (candles[c].to_numpy() for c in ("high", "low", "close"))
    assert decision.atr == pytest.approx(float(wilder_atr(high, low, close, 50)[-1]))


def test_the_sizer_is_only_asked_on_a_breakout() -> None:
    sizer = SpySizer()
    strategy = TrendBreakoutStrategy(sizer)
    strategy.evaluate(_range(last=100.0), BTC, account_balance=BALANCE)
    strategy.evaluate(_range(last=90.0), BTC, account_balance=BALANCE, open_positions=[_held("BTCUSDT")])
    assert sizer.calls == []


def test_a_size_below_the_broker_minimum_is_not_traded(strategy: TrendBreakoutStrategy) -> None:
    # 10 % of 10 000 USD is 0.0096 lots of EURUSD at 1.04: below the minimum of 0.01.
    decision = strategy.evaluate(_range(last=104.0, scale=0.01), EURUSD, account_balance=10_000.0)
    assert (decision.action, decision.reason) == (Action.HOLD, SIZE_TOO_SMALL)
    assert decision.volume == 0.0


def test_a_stop_that_would_not_be_positive_is_not_traded() -> None:
    wild = _candles(np.concatenate([np.tile([10.0, 1.0], 75)[:-1], [50.0]]), spread=0.1)
    decision = TrendBreakoutStrategy(stop_atr_multiple=20.0).evaluate(wild, BTC, account_balance=BALANCE)
    assert (decision.action, decision.reason) == (Action.HOLD, STOP_NOT_POSITIVE)


def test_an_invalid_balance_is_a_risk_error(strategy: TrendBreakoutStrategy) -> None:
    with pytest.raises(RiskError, match="account_balance"):
        strategy.evaluate(_range(last=104.0), BTC, account_balance=-500.0)


# --- Portfolio: PortfolioRiskManager ----------------------------------------


class SpyPortfolio(PortfolioRiskManager):
    def __init__(self, **limits: Any) -> None:
        super().__init__(**limits)
        self.calls: list[tuple[Any, ...]] = []

    def can_open_position(self, *arguments: Any) -> tuple[bool, str]:
        self.calls.append(arguments)
        return super().can_open_position(*arguments)


def test_the_portfolio_is_consulted_with_the_sized_position() -> None:
    portfolio = SpyPortfolio()
    held = [_held("ETHUSDT")]
    decision = TrendBreakoutStrategy(portfolio=portfolio).evaluate(
        _range(last=104.0), BTC, account_balance=BALANCE, open_positions=held
    )
    (call,) = portfolio.calls
    assert call == (BALANCE, held, decision.nominal_value, decision.risk_amount, "BTCUSDT", 1, "crypto")
    assert decision.action is Action.ENTER_LONG


def test_the_portfolio_is_not_consulted_without_an_order_to_approve() -> None:
    portfolio = SpyPortfolio()
    strategy = TrendBreakoutStrategy(portfolio=portfolio)
    strategy.evaluate(_range(last=100.0), BTC, account_balance=BALANCE)  # no breakout
    strategy.evaluate(_range(last=90.0), BTC, account_balance=BALANCE, open_positions=[_held("BTCUSDT")])  # exit
    strategy.evaluate(_range(last=104.0, scale=0.01), EURUSD, account_balance=10_000.0)  # size too small
    assert portfolio.calls == []


def test_a_full_portfolio_blocks_the_entry(strategy: TrendBreakoutStrategy) -> None:
    full = [_held(f"COIN{i}USDT") for i in range(10)]  # ten positions of 10 %
    decision = strategy.evaluate(_range(last=104.0), BTC, account_balance=BALANCE, open_positions=full)
    assert (decision.action, decision.reason) == (Action.HOLD, MAX_LEVERAGE_EXCEEDED)
    assert decision.volume == 0.0 and decision.stop_price is None
    nine = strategy.evaluate(_range(last=104.0), BTC, account_balance=BALANCE, open_positions=full[:9])
    assert nine.action is Action.ENTER_LONG


def test_too_much_open_risk_blocks_the_entry(strategy: TrendBreakoutStrategy) -> None:
    hot = [_held("ETHUSDT", risk=3_990.0)]
    decision = strategy.evaluate(_range(last=104.0), BTC, account_balance=BALANCE, open_positions=hot)
    assert (decision.action, decision.reason) == (Action.HOLD, MAX_HEAT_EXCEEDED)


def test_correlated_forex_is_blocked_and_a_crypto_basket_is_not(strategy: TrendBreakoutStrategy) -> None:
    eurusd_open = [_held("EURUSD", "forex")]
    blocked = strategy.evaluate(_range(last=104.0, scale=0.01), GBPUSD, account_balance=BALANCE, open_positions=eurusd_open)
    assert strategy.evaluate(_range(last=104.0, scale=0.01), GBPUSD, account_balance=BALANCE).volume == 0.09
    assert (blocked.action, blocked.reason) == (Action.HOLD, CORRELATED_EXPOSURE_BLOCKED)
    basket = [_held("BTCUSDT"), _held("SOLUSDT"), _held("ADAUSDT")]
    assert strategy.evaluate(_range(last=104.0), ETH, account_balance=BALANCE, open_positions=basket).action is Action.ENTER_LONG


def _twelve_breakouts(strategy: TrendBreakoutStrategy) -> tuple[list[str], list[ActivePosition]]:
    candles = _range(last=104.0)
    positions: list[ActivePosition] = []
    verdicts = []
    for index in range(12):
        coin = InstrumentSpec(f"COIN{index}USDT", "crypto", step_size=0.00001)
        decision = strategy.evaluate(candles, coin, account_balance=BALANCE, open_positions=positions)
        verdicts.append(decision.reason)
        if decision.action is Action.ENTER_LONG:
            positions.append(decision.to_active_position())
    return verdicts, positions


def test_with_the_default_limits_open_risk_fills_before_the_nominal(strategy: TrendBreakoutStrategy) -> None:
    # On daily candles the stop is far away: here each 10 % position risks 0.56 % of the balance,
    # so the 4 % open risk limit is reached with seven positions, not ten.
    verdicts, positions = _twelve_breakouts(strategy)
    assert verdicts == [BREAKOUT] * 7 + [MAX_HEAT_EXCEEDED] * 5
    assert positions[0].risk_usd == pytest.approx(558.0, abs=1.0)
    assert sum(position.risk_usd for position in positions) <= 0.04 * BALANCE


def test_with_room_for_risk_twelve_breakouts_open_ten_positions() -> None:
    strategy = TrendBreakoutStrategy(portfolio=PortfolioRiskManager(max_portfolio_heat_pct=10.0))
    verdicts, positions = _twelve_breakouts(strategy)
    assert verdicts == [BREAKOUT] * 10 + [MAX_LEVERAGE_EXCEEDED] * 2
    assert sum(position.nominal_value for position in positions) <= BALANCE


def test_the_portfolio_check_can_be_left_to_the_caller() -> None:
    portfolio = SpyPortfolio()
    strategy = TrendBreakoutStrategy(portfolio=portfolio)
    full = [_held(f"COIN{i}USDT") for i in range(10)]
    assert strategy.evaluate(_range(last=104.0), BTC, account_balance=BALANCE, open_positions=full).reason == MAX_LEVERAGE_EXCEEDED
    raw = strategy.evaluate(_range(last=104.0), BTC, account_balance=BALANCE, open_positions=full, check_portfolio=False)
    # The same entry, already sized, but without the portfolio verdict: whoever allocates the risk will ask for it.
    assert (raw.action, raw.reason) == (Action.ENTER_LONG, BREAKOUT)
    assert raw.volume > 0.0 and raw.stop_price is not None
    assert len(portfolio.calls) == 1


def test_a_decision_becomes_the_position_the_portfolio_will_see(strategy: TrendBreakoutStrategy) -> None:
    decision = strategy.evaluate(_range(last=104.0), BTC, account_balance=BALANCE)
    assert decision.to_active_position() == ActivePosition(
        "BTCUSDT", 1, decision.nominal_value, decision.risk_amount, "crypto"
    )
    hold = strategy.evaluate(_range(last=100.0), BTC, account_balance=BALANCE)
    with pytest.raises(StrategyError, match="only an entry opens a position"):
        hold.to_active_position()


# --- Daily candles: large jumps ---------------------------------------------


def _crypto_daily() -> pd.DataFrame:
    """A year of daily candles with days of +25 %, -18 % and +40 %: market, not corrupt data."""
    rng = np.random.default_rng(21)
    returns = rng.normal(0.002, 0.03, size=400)
    returns[[150, 260, 399]] = [np.log(1.25), np.log(0.82), np.log(1.40)]
    return _candles(100.0 * np.exp(np.cumsum(returns)), spread=0.5)


def test_daily_moves_above_ten_percent_are_evaluated_not_rejected(strategy: TrendBreakoutStrategy) -> None:
    candles = _crypto_daily()
    closes = candles["close"].to_numpy()
    assert np.abs(np.diff(np.log(closes))).max() > 0.30

    # A move like this is market, not corrupt data: the last candle (+40 %) is a breakout.
    decision = strategy.evaluate(candles, BTC, account_balance=BALANCE)
    assert (decision.action, decision.reason) == (Action.ENTER_LONG, BREAKOUT)


def test_a_jump_limit_can_be_set_explicitly() -> None:
    candles = _crypto_daily()
    with pytest.raises(MarketDataError, match=r"jump of 0\.22\d+ in the close at row 150"):
        TrendBreakoutStrategy(max_log_return=0.20).evaluate(candles, BTC, account_balance=BALANCE)
    assert TrendBreakoutStrategy(max_log_return=0.50).evaluate(candles, BTC, account_balance=BALANCE).action is Action.ENTER_LONG


# --- Corrupt data -----------------------------------------------------------


def _broken(column: str, row: int, value: Any) -> pd.DataFrame:
    candles = _range(last=104.0)
    candles[column] = candles[column].astype(object) if isinstance(value, str) else candles[column]
    candles.loc[row, column] = value
    return candles


@pytest.mark.parametrize(
    ("candles", "message"),
    [
        (_broken("close", 120, float("nan")), "non-finite prices \\(row 120\\)"),
        (_broken("high", 30, float("inf")), "non-finite prices \\(row 30\\)"),
        (_broken("low", 75, 0.0), "non-positive prices \\(row 75\\)"),
        (_broken("open", 10, -5.0), "non-positive prices \\(row 10\\)"),
        (_broken("high", 60, 50.0), "inconsistent candle at row 60"),
        (_broken("low", 61, 500.0), "inconsistent candle at row 61"),
        (_broken("close", 5, "abc"), "non-numeric prices"),
        (_broken("close_time", 40, 0), "not in chronological order \\(row 40\\)"),
    ],
)
def test_corrupt_candles_are_rejected(strategy: TrendBreakoutStrategy, candles: pd.DataFrame, message: str) -> None:
    with pytest.raises(MarketDataError, match=message):
        strategy.evaluate(candles, BTC, account_balance=BALANCE)


def test_repeated_candles_are_rejected(strategy: TrendBreakoutStrategy) -> None:
    candles = _range(last=104.0)
    doubled = pd.concat([candles.iloc[:100], candles.iloc[99:]], ignore_index=True)
    with pytest.raises(MarketDataError, match="chronological order \\(row 100\\)"):
        strategy.evaluate(doubled, BTC, account_balance=BALANCE)


def test_too_few_candles_are_rejected(strategy: TrendBreakoutStrategy) -> None:
    assert strategy.min_candles == 101
    with pytest.raises(MarketDataError, match="101 candles required, got 100"):
        strategy.evaluate(_range(last=104.0, n=100), BTC, account_balance=BALANCE)
    assert strategy.evaluate(_range(last=104.0, n=101), BTC, account_balance=BALANCE).action is Action.ENTER_LONG
    assert TrendBreakoutStrategy(entry_period=55, exit_period=20, atr_period=20).min_candles == 56


@pytest.mark.parametrize("candles", [None, [1.0, 2.0], np.zeros((150, 4)), {"close": [1.0]}])
def test_candles_must_be_a_dataframe(strategy: TrendBreakoutStrategy, candles: Any) -> None:
    with pytest.raises(StrategyError, match="candles must be a DataFrame"):
        strategy.evaluate(candles, BTC, account_balance=BALANCE)


def test_missing_columns_and_wrong_instrument(strategy: TrendBreakoutStrategy) -> None:
    with pytest.raises(StrategyError, match=r"missing the columns \['high', 'low'\]"):
        strategy.evaluate(_range(last=104.0).drop(columns=["high", "low"]), BTC, account_balance=BALANCE)
    with pytest.raises(StrategyError, match="instrument must be an InstrumentSpec"):
        strategy.evaluate(_range(last=104.0), "BTCUSDT", account_balance=BALANCE)  # type: ignore[arg-type]
    # Without the time column it is evaluated all the same: the order is up to the caller.
    assert strategy.evaluate(_range(last=104.0)[["open", "high", "low", "close"]], BTC, account_balance=BALANCE).action is Action.ENTER_LONG


def test_errors_are_value_errors() -> None:
    assert issubclass(MarketDataError, StrategyError) and issubclass(StrategyError, ValueError)


# --- Configuration ----------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"entry_period": 0}, {"entry_period": 2.5}, {"entry_period": True}, {"entry_period": None},
        {"exit_period": 0}, {"exit_period": "10"},
        {"atr_period": 1}, {"atr_period": 0}, {"atr_period": 14.0},
        {"stop_atr_multiple": 0.0}, {"stop_atr_multiple": -2.0}, {"stop_atr_multiple": float("nan")},
        {"risk_pct": 0.0}, {"risk_pct": "1"}, {"risk_pct": None},
        {"max_leverage": 0.0}, {"max_leverage": float("inf")},
        {"max_log_return": 0.0}, {"max_log_return": -0.1}, {"max_log_return": "0.5"}, {"max_log_return": True},
    ],
)
def test_invalid_strategy_parameters(arguments: dict[str, Any]) -> None:
    with pytest.raises(StrategyError, match=next(iter(arguments))):
        TrendBreakoutStrategy(**arguments)


def test_invalid_instrument_spec() -> None:
    with pytest.raises(StrategyError, match="symbol"):
        InstrumentSpec("", "crypto", step_size=0.001)
    with pytest.raises(StrategyError, match="asset_class"):
        InstrumentSpec("AAPL", "stocks", step_size=1.0)
    # ATRSizer validates the symbol's numbers as soon as there is something to size.
    with pytest.raises(RiskError, match="step_size"):
        TrendBreakoutStrategy().evaluate(_range(last=104.0), InstrumentSpec("BTCUSDT", "crypto", step_size=0.0),
                                         account_balance=BALANCE)


# --- Logging ----------------------------------------------------------------


def test_entries_and_exits_are_logged_at_info(strategy: TrendBreakoutStrategy) -> None:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}", level="INFO")
    try:
        strategy.evaluate(_range(last=100.0), BTC, account_balance=BALANCE)
        strategy.evaluate(_range(last=104.0), BTC, account_balance=BALANCE)
        strategy.evaluate(_range(last=90.0), BTC, account_balance=BALANCE, open_positions=[_held("BTCUSDT")])
    finally:
        logger.remove(sink)
    assert len(lines) == 3  # the portfolio approval, the buy and the exit; holding writes nothing
    assert lines[0].startswith("INFO|Entry on BTCUSDT approved")
    assert lines[1].startswith("INFO|BTCUSDT: buy 96.1538 lots, the close 104 is above the high of 20 candles")
    assert lines[2].startswith("INFO|BTCUSDT: exit, the close 90 is below the low of 10 candles")
