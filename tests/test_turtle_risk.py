"""Tests for Module 4 (risk): ATR-100, asymptotic lot sizing and the 2N stop."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.turtle_engine import RiskError, TurtleEngine


@pytest.fixture
def turtle() -> TurtleEngine:
    return TurtleEngine(atr_period=100, risk_per_unit=0.01, max_capital_fraction=0.10)


def _flat_candles(n: int, price: float = 100.0, spread: float = 10.0) -> tuple[np.ndarray, ...]:
    close = np.full(n, price)
    return close + spread / 2.0, close - spread / 2.0, close


# --- Construction ---------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"atr_period": 1},
        {"risk_per_unit": 0.0},
        {"risk_per_unit": 1.5},
        {"max_capital_fraction": 0.0},
        {"max_capital_fraction": 1.1},
        {"stop_atr_multiple": 0.0},
        {"exit_channel_period": 0},
    ],
)
def test_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TurtleEngine(**kwargs)


# --- N = ATR --------------------------------------------------------------


def test_atr_of_constant_range_equals_that_range(turtle: TurtleEngine) -> None:
    high, low, close = _flat_candles(300, spread=10.0)
    assert turtle.atr(high, low, close) == pytest.approx(10.0)


def test_atr_follows_wilder_smoothing() -> None:
    turtle = TurtleEngine(atr_period=3)
    close = np.array([100.0, 102.0, 101.0, 105.0, 104.0, 108.0])
    high = close + np.array([1.0, 1.0, 2.0, 1.0, 3.0, 1.0])
    low = close - np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

    true_range = [
        max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        for i in range(1, 6)
    ]
    expected = float(np.mean(true_range[:3]))
    for value in true_range[3:]:
        expected = (expected * 2.0 + value) / 3.0

    assert turtle.atr(high, low, close) == pytest.approx(expected)


def test_atr_counts_gaps_against_previous_close() -> None:
    turtle = TurtleEngine(atr_period=2)
    close = np.array([100.0, 100.0, 150.0])
    # The last candle has a range of 2 but opens with a gap of 50 over the previous close.
    high, low = close + 1.0, close - 1.0

    assert turtle.atr(high, low, close) == pytest.approx((2.0 + 51.0) / 2.0)


def test_atr_requires_period_plus_one_candles(turtle: TurtleEngine) -> None:
    assert turtle.min_candles == 101
    turtle.atr(*_flat_candles(101))
    with pytest.raises(RiskError, match="101 candles"):
        turtle.atr(*_flat_candles(100))


def test_atr_rejects_corrupted_candles(turtle: TurtleEngine) -> None:
    high, low, close = _flat_candles(200)
    high[50] = np.nan
    with pytest.raises(RiskError, match="non-finite"):
        turtle.atr(high, low, close)


def test_atr_rejects_mismatched_series(turtle: TurtleEngine) -> None:
    high, low, close = _flat_candles(200)
    with pytest.raises(RiskError, match="equal length"):
        turtle.atr(high[:-1], low, close)


# --- Asymptotic position size ---------------------------------------------


def test_size_follows_one_percent_risk_when_volatility_is_high(turtle: TurtleEngine) -> None:
    # 1 % of 10 000 = 100 USDT of risk per N; with N = 500 -> 0.2 units.
    assert turtle.position_size(capital=10_000.0, atr=500.0, price=100.0) == pytest.approx(0.2)


def test_size_is_capped_at_ten_percent_of_capital_when_volatility_is_low(turtle: TurtleEngine) -> None:
    # The risk rule would ask for 10 000 units; the cap allows 1 000 / 100 = 10.
    assert turtle.position_size(capital=10_000.0, atr=0.01, price=100.0) == pytest.approx(10.0)


@pytest.mark.parametrize("atr", [1e-3, 1e-9, 1e-300, 0.0])
def test_size_stays_bounded_as_atr_tends_to_zero(turtle: TurtleEngine, atr: float) -> None:
    units = turtle.position_size(capital=10_000.0, atr=atr, price=100.0)
    assert units == pytest.approx(10.0)
    assert units * 100.0 <= 0.10 * 10_000.0 + 1e-9


@pytest.mark.parametrize("atr", [0.5, 5.0, 50.0, 500.0, 5000.0])
@pytest.mark.parametrize("price", [0.5, 100.0, 60_000.0])
def test_notional_never_exceeds_the_capital_cap(turtle: TurtleEngine, atr: float, price: float) -> None:
    units = turtle.position_size(capital=25_000.0, atr=atr, price=price)
    assert units * price <= 0.10 * 25_000.0 * (1.0 + 1e-12)
    assert units * atr <= 0.01 * 25_000.0 * (1.0 + 1e-12)


@pytest.mark.parametrize(
    ("capital", "atr", "price"),
    [(0.0, 10.0, 100.0), (-5.0, 10.0, 100.0), (1000.0, -1.0, 100.0), (1000.0, 10.0, 0.0),
     (1000.0, float("nan"), 100.0), (float("inf"), 10.0, 100.0)],
)
def test_size_rejects_invalid_inputs(turtle: TurtleEngine, capital: float, atr: float, price: float) -> None:
    with pytest.raises(RiskError):
        turtle.position_size(capital, atr, price)


# --- Stop loss 2N ---------------------------------------------------------


def test_stop_sits_two_atr_below_entry(turtle: TurtleEngine) -> None:
    assert turtle.stop_price(entry_price=30_000.0, atr=250.0) == pytest.approx(29_500.0)


def test_stop_multiple_is_configurable() -> None:
    assert TurtleEngine(stop_atr_multiple=3.0).stop_price(100.0, 5.0) == pytest.approx(85.0)


def test_stop_at_or_below_zero_is_rejected(turtle: TurtleEngine) -> None:
    with pytest.raises(RiskError, match="stop not positive"):
        turtle.stop_price(entry_price=100.0, atr=50.0)


@pytest.mark.parametrize("atr", [0.0, -1.0, float("nan")])
def test_stop_requires_a_valid_atr(turtle: TurtleEngine, atr: float) -> None:
    with pytest.raises(RiskError, match="invalid ATR"):
        turtle.stop_price(100.0, atr)


# --- Full plan ------------------------------------------------------------


def test_plan_risks_two_percent_of_capital_at_the_stop_when_uncapped(turtle: TurtleEngine) -> None:
    # Risk rule in force: 1 % per N and the stop at 2N -> a 2 % loss at the stop.
    plan = turtle.plan(capital=10_000.0, price=1_000.0, atr=200.0)

    assert not plan.capped
    assert plan.units == pytest.approx(0.5)
    assert plan.stop_price == pytest.approx(600.0)
    assert plan.units * (1_000.0 - plan.stop_price) == pytest.approx(0.02 * 10_000.0)


def test_plan_flags_when_the_capital_cap_is_binding(turtle: TurtleEngine) -> None:
    plan = turtle.plan(capital=10_000.0, price=30_000.0, atr=250.0)

    assert plan.capped
    assert plan.units * 30_000.0 == pytest.approx(1_000.0)
    # With the cap in force the loss at the stop is below the nominal 2 %.
    assert plan.units * (30_000.0 - plan.stop_price) < 0.02 * 10_000.0


# --- Donchian exit (trend loss) -------------------------------------------


def test_exit_channel_is_the_lowest_low_of_the_previous_ten_candles(turtle: TurtleEngine) -> None:
    low = np.array([50.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0, 90.5, 80.0])
    # The last candle (80, the one evaluated) and those before the window (50) are ignored.
    assert turtle.exit_channel(low) == 90.5


@pytest.mark.parametrize(("close", "breaks"), [(90.4, True), (90.5, False), (95.0, False)])
def test_trend_is_lost_only_when_the_close_goes_below_the_channel(
    turtle: TurtleEngine, close: float, breaks: bool
) -> None:
    low = np.array([99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0, 90.5, 89.0])
    assert turtle.breaks_exit_channel(low, close) is breaks


def test_exit_channel_period_is_configurable() -> None:
    low = np.array([10.0, 30.0, 20.0, 40.0, 5.0])
    assert TurtleEngine(exit_channel_period=2).exit_channel(low) == 20.0
    assert TurtleEngine(exit_channel_period=4).exit_channel(low) == 10.0


def test_exit_channel_needs_enough_clean_candles(turtle: TurtleEngine) -> None:
    with pytest.raises(RiskError, match="11 candles"):
        turtle.exit_channel(np.full(10, 100.0))
    low = np.full(20, 100.0)
    low[15] = np.nan
    with pytest.raises(RiskError, match="non-finite"):
        turtle.exit_channel(low)


def test_min_candles_covers_the_longest_lookback() -> None:
    assert TurtleEngine(atr_period=5, exit_channel_period=30).min_candles == 31


# --- Series for the dashboard ---------------------------------------------


def test_series_agree_with_the_values_the_bot_uses() -> None:
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=400))
    high, low = close + rng.uniform(0.1, 2.0, size=400), close - rng.uniform(0.1, 2.0, size=400)
    turtle = TurtleEngine(atr_period=100, exit_channel_period=10)

    atr = turtle.atr_series(high, low, close)
    channel = turtle.exit_channel_series(low)

    assert atr.shape == channel.shape == (400,)
    assert np.isnan(atr[:100]).all() and np.isfinite(atr[100:]).all()
    assert np.isnan(channel[:10]).all() and np.isfinite(channel[10:]).all()
    # At each candle, the series equals what the bot would have computed up to it.
    for end in (101, 150, 263, 400):
        assert atr[end - 1] == pytest.approx(turtle.atr(high[:end], low[:end], close[:end]))
        assert channel[end - 1] == turtle.exit_channel(low[:end])


def test_exit_channel_series_of_a_short_window_is_empty(turtle: TurtleEngine) -> None:
    assert np.isnan(turtle.exit_channel_series(np.full(10, 5.0))).all()


# --- Short selling --------------------------------------------------------


def test_stop_of_a_short_sits_two_atr_above_entry(turtle: TurtleEngine) -> None:
    assert turtle.stop_price(100.0, 1.5, "SHORT") == pytest.approx(103.0)


def test_short_plan_has_the_same_size_and_the_stop_on_the_other_side(turtle: TurtleEngine) -> None:
    buy, short = turtle.plan(10_000.0, 100.0, 1.5), turtle.plan(10_000.0, 100.0, 1.5, "SHORT")

    assert short.units == buy.units and short.capped == buy.capped
    assert (buy.stop_price, short.stop_price) == (pytest.approx(97.0), pytest.approx(103.0))


def test_short_exit_channel_is_the_highest_high_of_the_previous_ten_candles(turtle: TurtleEngine) -> None:
    highs = np.concatenate([np.full(5, 500.0), np.linspace(100.0, 109.0, 10), [50.0]])

    # Neither the current candle (50) nor the ones before the last ten (500) count.
    assert turtle.short_exit_channel(highs) == 109.0
    assert turtle.short_exit_channel_series(highs)[-1] == 109.0
    assert np.isnan(turtle.short_exit_channel_series(highs)[:10]).all()
    with pytest.raises(RiskError, match="exit channel"):
        turtle.short_exit_channel(highs[:10])
