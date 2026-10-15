"""Tests for position sizing (``ATRSizer``): fixed risk and leverage cap."""

from __future__ import annotations

from typing import Any

import pytest
from loguru import logger

from src.risk.position_sizer import MAX_LEVERAGE, ATRSizer
from src.risk.turtle_engine import RiskError, TurtleEngine

BTC = {"contract_size": 1.0, "step_size": 0.001, "asset_price": 60_000.0}
EURUSD = {"contract_size": 100_000.0, "step_size": 0.01, "asset_price": 1.08}
GOLD = {"contract_size": 100.0, "step_size": 0.01, "asset_price": 2_400.0}


@pytest.fixture
def sizer() -> ATRSizer:
    return ATRSizer()


def _log(level: str = "DEBUG") -> tuple[list[str], int]:
    lines: list[str] = []
    return lines, logger.add(lines.append, format="{level}|{message}", level=level)


def _nominal(lots: float, instrument: dict[str, float]) -> float:
    return lots * instrument["contract_size"] * instrument["asset_price"]


# --- Reference cases --------------------------------------------------------


def test_crypto_futures_sizing(sizer: ATRSizer) -> None:
    # 10 USD of risk / 3 000 USD of distance = 0.003333… BTC -> 0.003.
    lots = sizer.calculate_position_size(
        account_balance=1_000.0, risk_pct=1.0, atr_val=1_500.0, multiplier=2.0,
        max_leverage=1.0, **BTC,
    )
    assert lots == 0.003
    # 180 USD of notional, 18 % of the balance: with the default cap (10 %) it ends at 0.001.
    assert _nominal(lots, BTC) == pytest.approx(180.0)
    assert sizer.calculate_position_size(account_balance=1_000.0, atr_val=1_500.0, **BTC) == 0.001


def test_forex_eurusd_sizing(sizer: ATRSizer) -> None:
    # 100 USD / 0.0040 = 25 000 units = 0.25 lots: 27 000 USD of notional, 2.7 times the balance.
    arguments = {"account_balance": 10_000.0, "risk_pct": 1.0, "atr_val": 0.0020, "multiplier": 2.0}
    assert sizer.calculate_position_size(max_leverage=3.0, **arguments, **EURUSD) == 0.25
    # With less leverage allowed the cap decides: 20 000 / 108 000 = 0.185 -> 0.18.
    assert sizer.calculate_position_size(max_leverage=2.0, **arguments, **EURUSD) == 0.18
    # Unleveraged: 10 000 / 108 000 = 0.0926 -> 0.09.
    assert sizer.calculate_position_size(max_leverage=1.0, **arguments, **EURUSD) == 0.09
    # Default cap (10 %): 1 000 / 108 000 = 0.0093 lots, below the minimum of 0.01.
    assert sizer.calculate_position_size(**arguments, **EURUSD) == 0.0


def test_gold_xauusd_sizing(sizer: ATRSizer) -> None:
    # 75 USD / 24 USD = 3.125 ounces = 0.03125 lots -> 0.03: 7 200 USD of notional, 1.44x.
    arguments = {"account_balance": 5_000.0, "risk_pct": 1.5, "atr_val": 12.0, "multiplier": 2.0}
    assert sizer.calculate_position_size(max_leverage=3.0, **arguments, **GOLD) == 0.03
    assert sizer.calculate_position_size(max_leverage=1.5, **arguments, **GOLD) == 0.03
    # Unleveraged: 5 000 / 240 000 = 0.0208 -> 0.02.
    assert sizer.calculate_position_size(max_leverage=1.0, **arguments, **GOLD) == 0.02
    # Default cap (10 %): 500 / 240 000 = 0.0021 lots, below the minimum of 0.01.
    assert sizer.calculate_position_size(**arguments, **GOLD) == 0.0


def test_invalid_inputs(sizer: ATRSizer) -> None:
    with pytest.raises(ValueError, match="atr_val"):
        sizer.calculate_position_size(account_balance=1_000.0, atr_val=0.0, **BTC)
    with pytest.raises(ValueError, match="account_balance"):
        sizer.calculate_position_size(account_balance=-500.0, atr_val=1_500.0, **BTC)


def test_insufficient_capital(sizer: ATRSizer) -> None:
    lines, sink = _log()
    try:
        # 0.50 USD of risk / 4 000 USD of distance = 0.000125 BTC < 0.001.
        lots = sizer.calculate_position_size(
            account_balance=50.0, risk_pct=1.0, atr_val=2_000.0, multiplier=2.0,
            max_leverage=1.0, **BTC,
        )
    finally:
        logger.remove(sink)

    assert lots == 0.0
    assert len(lines) == 1
    assert lines[0].startswith("WARNING|Not enough capital for the configured risk")
    assert "0.000125 lots" in lines[0] and "minimum step is 0.001" in lines[0]


# --- Leverage cap -----------------------------------------------------------


def test_spot_never_buys_more_than_the_balance(sizer: ATRSizer) -> None:
    # By risk it would be 1.341 BTC (114 900 USD) with a balance of 100 000 USD.
    spot = {"contract_size": 1.0, "step_size": 0.001, "asset_price": 85_669.0}
    lots = sizer.calculate_position_size(
        account_balance=100_000.0, atr_val=372.77, max_leverage=1.0, **spot
    )
    assert lots == 1.167
    assert 99_900.0 < _nominal(lots, spot) <= 100_000.0


def test_the_cap_only_ever_reduces_the_size(sizer: ATRSizer) -> None:
    arguments = {"account_balance": 100_000.0, "atr_val": 0.0010, **EURUSD}
    sizes = [
        sizer.calculate_position_size(max_leverage=leverage, **arguments)
        for leverage in (0.1, 0.5, 1.0, 2.0, 3.0)
    ]
    # By risk it would be 5 lots (5.4x): the cap decides in all of them and grows with it.
    assert sizes == [0.09, 0.46, 0.92, 1.85, 2.77]


@pytest.mark.parametrize("instrument", [BTC, EURUSD, GOLD])
@pytest.mark.parametrize("leverage", [0.1, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("atr_fraction", [0.0005, 0.002, 0.01, 0.05])
def test_neither_limit_is_ever_exceeded(
    sizer: ATRSizer, instrument: dict[str, float], leverage: float, atr_fraction: float
) -> None:
    balance, atr = 250_000.0, instrument["asset_price"] * atr_fraction
    lots = sizer.calculate_position_size(
        account_balance=balance, atr_val=atr, max_leverage=leverage, **instrument
    )
    assert lots > 0.0
    assert _nominal(lots, instrument) <= balance * leverage * (1 + 1e-12)
    assert lots * instrument["contract_size"] * atr * 2.0 <= balance * 0.01 * (1 + 1e-12)


def test_the_default_cap_is_a_tenth_of_the_balance(sizer: ATRSizer) -> None:
    # By risk it would be 0.25 lots (60 000 USD); 10 % of 100 000 is 0.0416 -> 0.04.
    lots = sizer.calculate_position_size(account_balance=100_000.0, atr_val=20.0, **GOLD)
    assert lots == 0.04
    assert lots == sizer.calculate_position_size(
        account_balance=100_000.0, atr_val=20.0, max_leverage=0.1, **GOLD
    )
    # Ten positions like this fit in the balance without leveraging the account.
    assert 10 * _nominal(lots, GOLD) <= 100_000.0


def test_the_default_cap_is_the_one_the_bot_uses_today(sizer: ATRSizer) -> None:
    # TurtleEngine caps each trade at 10 % of capital: that is max_leverage = 0.1.
    balance, price, atr = 100_000.0, 4_155.7, 16.214
    plan = TurtleEngine(max_capital_fraction=0.10).plan(balance, price, atr)
    lots = sizer.calculate_position_size(
        account_balance=balance, atr_val=atr, contract_size=100.0, step_size=0.01, asset_price=price
    )
    assert plan.capped
    assert lots == 0.02 == int(plan.units / 100.0 / 0.01) * 0.01


def test_too_little_capital_for_the_cap_says_so(sizer: ATRSizer) -> None:
    lines, sink = _log()
    try:
        # By risk it would be 0.0025 BTC, but 50 USD unleveraged is 0.00083 BTC.
        lots = sizer.calculate_position_size(
            account_balance=50.0, atr_val=100.0, max_leverage=1.0, **BTC
        )
    finally:
        logger.remove(sink)

    assert lots == 0.0
    assert len(lines) == 1
    assert lines[0].startswith("WARNING|Not enough capital for the maximum leverage: 1x of 50.00 USD")


def test_leverage_above_the_ceiling_is_rejected(sizer: ATRSizer) -> None:
    assert MAX_LEVERAGE == 3.0
    arguments = {"account_balance": 10_000.0, "atr_val": 0.0020, **EURUSD}
    assert sizer.calculate_position_size(max_leverage=MAX_LEVERAGE, **arguments) == 0.25
    with pytest.raises(RiskError, match="max_leverage cannot exceed 3"):
        sizer.calculate_position_size(max_leverage=30.0, **arguments)


# --- Formula and rounding ---------------------------------------------------


def test_risk_and_multiplier_scale_the_size(sizer: ATRSizer) -> None:
    base = {"account_balance": 100_000.0, "atr_val": 20.0, "max_leverage": 3.0, **GOLD}
    assert sizer.calculate_position_size(**base) == 0.25
    assert sizer.calculate_position_size(risk_pct=2.0, **base) == 0.5
    assert sizer.calculate_position_size(multiplier=4.0, **base) == 0.12  # 0.125 rounded down
    assert sizer.calculate_position_size(risk_pct=0.5, multiplier=1.0, **base) == 0.25


def test_rounds_down_and_never_up(sizer: ATRSizer) -> None:
    # 1 000 / (2 x 25.1) = 19.92 ounces = 0.1992 lots -> 0.19, not 0.20.
    lots = sizer.calculate_position_size(
        account_balance=100_000.0, atr_val=25.1, max_leverage=1.0, **GOLD
    )
    assert lots == 0.19


@pytest.mark.parametrize("hundredths", range(1, 60))
def test_an_exact_multiple_of_the_step_is_kept(sizer: ATRSizer, hundredths: int) -> None:
    # With floats, 7 / 100 can end up as 0.0699… and lose a step when rounding.
    balance = hundredths * 100.0  # risk of `hundredths` USD with the stop at 1 USD/ounce
    lots = sizer.calculate_position_size(
        account_balance=balance, atr_val=0.5, contract_size=100.0, step_size=0.01, asset_price=1.0
    )
    assert lots == pytest.approx(hundredths / 100.0, abs=1e-12)
    assert round(lots / 0.01) == hundredths


def test_a_step_that_is_not_a_power_of_ten(sizer: ATRSizer) -> None:
    # 0.3333… contracts with a step of 0.05 -> 0.30.
    lots = sizer.calculate_position_size(
        account_balance=100_000.0, atr_val=1_500.0, contract_size=1.0, step_size=0.05,
        asset_price=60_000.0, max_leverage=1.0,
    )
    assert lots == pytest.approx(0.30, abs=1e-12)


def test_exactly_one_step_is_enough(sizer: ATRSizer) -> None:
    lines, sink = _log("WARNING")
    try:
        # 1 % of 4 000 = 40 USD with the stop at 40 USD/ounce -> 1 ounce = exactly 0.01 lots.
        lots = sizer.calculate_position_size(
            account_balance=4_000.0, atr_val=20.0, max_leverage=1.0, **GOLD
        )
    finally:
        logger.remove(sink)

    assert lots == 0.01
    assert lines == []


def test_a_normal_size_logs_the_calculation_at_debug_only(sizer: ATRSizer) -> None:
    lines, sink = _log()
    try:
        sizer.calculate_position_size(account_balance=100_000.0, atr_val=20.0, max_leverage=1.0, **GOLD)
        sizer.calculate_position_size(account_balance=100_000.0, atr_val=5.0, max_leverage=1.0, **GOLD)
    finally:
        logger.remove(sink)

    assert len(lines) == 2
    assert lines[0].startswith("DEBUG|Position size: 0.25 lots, limited by risk")
    assert "notional 60000.00 USD (0.60x)" in lines[0] and "loss at the stop 1000.00 USD" in lines[0]
    # With the ATR at a quarter, risk would ask for 1 lot (2.4x): the cap decides.
    assert lines[1].startswith("DEBUG|Position size: 0.41 lots, limited by leverage")
    assert "loss at the stop 410.00 USD" in lines[1]


# --- Validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "account_balance", "atr_val", "contract_size", "step_size", "asset_price",
        "risk_pct", "multiplier", "max_leverage",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_arguments_that_are_not_positive(sizer: ATRSizer, name: str, value: float) -> None:
    arguments: dict[str, Any] = {"account_balance": 100_000.0, "atr_val": 20.0, **GOLD, name: value}
    with pytest.raises(ValueError, match=name):
        sizer.calculate_position_size(**arguments)


@pytest.mark.parametrize(
    "name",
    [
        "account_balance", "atr_val", "contract_size", "step_size", "asset_price",
        "risk_pct", "multiplier", "max_leverage",
    ],
)
@pytest.mark.parametrize("value", [None, "100", [1.0], True])
def test_rejects_arguments_that_are_not_numbers(sizer: ATRSizer, name: str, value: object) -> None:
    # Corrupt broker data must not surface as a TypeError or be traded (True would count as 1.0).
    arguments: dict[str, Any] = {"account_balance": 100_000.0, "atr_val": 20.0, **GOLD, name: value}
    with pytest.raises(RiskError, match=f"{name} must be a number, got"):
        sizer.calculate_position_size(**arguments)


def test_accepts_the_number_types_the_bot_handles(sizer: ATRSizer) -> None:
    import numpy as np

    expected = sizer.calculate_position_size(account_balance=100_000.0, atr_val=20.0, **GOLD)
    assert expected == sizer.calculate_position_size(
        account_balance=100_000, atr_val=np.float64(20.0), contract_size=100, step_size=0.01,
        asset_price=np.float64(2_400.0),
    )


def test_errors_are_risk_errors_like_the_rest_of_the_module(sizer: ATRSizer) -> None:
    with pytest.raises(RiskError, match="atr_val must be a positive number, got 0.0"):
        sizer.calculate_position_size(account_balance=100_000.0, atr_val=0.0, **GOLD)


def test_rejects_risking_more_than_the_whole_account(sizer: ATRSizer) -> None:
    with pytest.raises(ValueError, match="risk_pct cannot exceed 100"):
        sizer.calculate_position_size(account_balance=100_000.0, atr_val=20.0, risk_pct=150.0, **GOLD)


def test_the_price_is_required(sizer: ATRSizer) -> None:
    # Without a price there is no way to value the position: the cap is not optional.
    with pytest.raises(TypeError, match="asset_price"):
        sizer.calculate_position_size(  # type: ignore[call-arg]
            account_balance=100_000.0, atr_val=20.0, contract_size=100.0, step_size=0.01
        )


def test_arguments_are_keyword_only(sizer: ATRSizer) -> None:
    # Eight numbers in a row in a positional call get mixed up with no warning.
    with pytest.raises(TypeError):
        sizer.calculate_position_size(100_000.0, 1.0, 20.0, 2.0, 100.0, 0.01, 2_400.0, 1.0)  # type: ignore[misc]


# --- Broker minimum and maximum volume ---------------------------------------


def test_below_the_broker_minimum_returns_zero_and_warns(sizer: ATRSizer) -> None:
    arguments = {"account_balance": 100_000.0, "atr_val": 20.0, **GOLD}  # 0.04 lots with the 10 % cap
    lines, sink = _log()
    try:
        lots = sizer.calculate_position_size(min_volume=0.05, **arguments)
    finally:
        logger.remove(sink)

    assert lots == 0.0
    assert lines == [
        "WARNING|Size below the broker's minimum volume: 0.04 lots (limited by leverage) "
        "and the minimum is 0.05. No position opened\n"
    ]


def test_exactly_the_broker_minimum_is_traded(sizer: ATRSizer) -> None:
    arguments = {"account_balance": 100_000.0, "atr_val": 20.0, **GOLD}
    assert sizer.calculate_position_size(min_volume=0.04, **arguments) == 0.04
    assert sizer.calculate_position_size(min_volume=0.01, **arguments) == 0.04
    # With no minimum (the default) the result is the usual one.
    assert sizer.calculate_position_size(**arguments) == sizer.calculate_position_size(min_volume=0.0, **arguments)


def test_the_minimum_is_checked_after_rounding_to_the_step(sizer: ATRSizer) -> None:
    # 0.1992 lots before rounding: above a minimum of 0.195, but what would be traded is 0.19.
    arguments = {"account_balance": 100_000.0, "atr_val": 25.1, "max_leverage": 1.0, **GOLD}
    assert sizer.calculate_position_size(min_volume=0.19, **arguments) == 0.19
    assert sizer.calculate_position_size(min_volume=0.195, **arguments) == 0.0


def test_too_little_capital_is_reported_as_such_even_with_a_minimum(sizer: ATRSizer) -> None:
    lines, sink = _log("WARNING")
    try:
        lots = sizer.calculate_position_size(account_balance=1_000.0, atr_val=20.0, min_volume=0.01, **GOLD)
    finally:
        logger.remove(sink)
    assert lots == 0.0
    assert len(lines) == 1 and lines[0].startswith("WARNING|Not enough capital for the maximum leverage")


def test_size_is_capped_at_the_broker_maximum(sizer: ATRSizer) -> None:
    arguments = {"account_balance": 100_000.0, "atr_val": 20.0, "max_leverage": 3.0, **GOLD}  # 0.25 lots
    assert sizer.calculate_position_size(**arguments) == 0.25
    assert sizer.calculate_position_size(max_volume=0.10, **arguments) == 0.10
    assert sizer.calculate_position_size(max_volume=0.25, **arguments) == 0.25
    assert sizer.calculate_position_size(max_volume=50.0, **arguments) == 0.25  # a generous maximum changes nothing
    # A maximum that is not a multiple of the step is lowered to the previous one: never exceeded.
    assert sizer.calculate_position_size(max_volume=0.105, **arguments) == 0.10


def test_the_log_says_when_the_broker_maximum_decides(sizer: ATRSizer) -> None:
    lines, sink = _log()
    try:
        sizer.calculate_position_size(account_balance=100_000.0, atr_val=20.0, max_leverage=3.0, max_volume=0.10, **GOLD)
    finally:
        logger.remove(sink)
    assert len(lines) == 1
    assert lines[0].startswith("DEBUG|Position size: 0.1 lots, limited by the broker's maximum volume")
    assert "loss at the stop 400.00 USD" in lines[0]


def test_minimum_and_maximum_together(sizer: ATRSizer) -> None:
    arguments = {"account_balance": 100_000.0, "atr_val": 20.0, "max_leverage": 3.0, **GOLD}
    assert sizer.calculate_position_size(min_volume=0.10, max_volume=0.10, **arguments) == 0.10
    with pytest.raises(RiskError, match=r"max_volume \(0.05\) cannot be less than min_volume \(0.1\)"):
        sizer.calculate_position_size(min_volume=0.10, max_volume=0.05, **arguments)


def test_a_maximum_below_the_step_cannot_be_traded(sizer: ATRSizer) -> None:
    lines, sink = _log("WARNING")
    try:
        lots = sizer.calculate_position_size(
            account_balance=100_000.0, atr_val=20.0, max_leverage=3.0, max_volume=0.005, **GOLD
        )
    finally:
        logger.remove(sink)
    assert lots == 0.0
    assert lines == ["WARNING|The broker's maximum volume (0.005) is below the lot step (0.01). No position opened\n"]


@pytest.mark.parametrize("value", [None, "0.01", [0.01], True, -0.01, float("nan"), float("inf")])
def test_invalid_min_volume(sizer: ATRSizer, value: object) -> None:
    arguments: dict[str, Any] = {"account_balance": 100_000.0, "atr_val": 20.0, **GOLD}
    with pytest.raises(RiskError, match="min_volume"):
        sizer.calculate_position_size(min_volume=value, **arguments)


@pytest.mark.parametrize("value", ["5", [5.0], True, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_max_volume(sizer: ATRSizer, value: object) -> None:
    arguments: dict[str, Any] = {"account_balance": 100_000.0, "atr_val": 20.0, **GOLD}
    with pytest.raises(RiskError, match="max_volume"):
        sizer.calculate_position_size(max_volume=value, **arguments)


# --- Tick value: assets not quoted in the account currency ------------------

USDJPY = {"contract_size": 100_000.0, "step_size": 0.01, "asset_price": 150.0}
# Dollar account: a tick of 0.001 yen on a lot of 100 000 USD is worth 100 yen = 0.6667 USD.
USDJPY_TICK = {"tick_size": 0.001, "tick_value": 100.0 / 150.0}


def test_tick_value_of_a_dollar_quoted_pair_changes_nothing(sizer: ATRSizer) -> None:
    arguments = {"account_balance": 10_000.0, "atr_val": 0.0020, "max_leverage": 3.0, **EURUSD}
    # EURUSD with a dollar account: a tick of 0.00001 on 100 000 EUR is worth 1 USD.
    assert sizer.calculate_position_size(tick_size=0.00001, tick_value=1.0, **arguments) == 0.25
    assert sizer.calculate_position_size(**arguments) == 0.25
    gold = {"account_balance": 100_000.0, "atr_val": 20.0, "max_leverage": 3.0, **GOLD}
    assert sizer.calculate_position_size(tick_size=0.01, tick_value=1.0, **gold) == 0.25


def test_tick_value_converts_the_risk_to_the_account_currency(sizer: ATRSizer) -> None:
    # Stop at 1 yen: one lot loses 100 000 yen = 666.67 USD. With 1 000 USD of risk, 1.5 lots.
    arguments = {"account_balance": 100_000.0, "atr_val": 0.50, "max_leverage": 3.0, **USDJPY}
    lots = sizer.calculate_position_size(**arguments, **USDJPY_TICK)
    assert lots == 1.5
    assert lots * (1.0 / 0.001) * USDJPY_TICK["tick_value"] == pytest.approx(1_000.0)
    # Without the tick value yen would be taken for dollars: 0.01 lots, 150 times less.
    assert sizer.calculate_position_size(**arguments) == 0.01


def test_tick_value_converts_the_notional_too(sizer: ATRSizer) -> None:
    # One lot of USDJPY is 100 000 USD of notional, not 15 million.
    arguments = {"account_balance": 100_000.0, "atr_val": 0.50, **USDJPY, **USDJPY_TICK}
    assert sizer.calculate_position_size(**arguments) == 0.10  # default cap: 10 % of the balance
    assert sizer.calculate_position_size(max_leverage=1.0, **arguments) == 1.0
    assert sizer.calculate_position_size(max_leverage=2.0, **arguments) == 1.5  # now risk decides


def test_tick_value_on_a_cross_with_neither_side_in_the_account_currency(sizer: ATRSizer) -> None:
    # EURGBP with a dollar account and the pound at 1.27 USD: a tick of 0.00001 is worth 1.27 USD.
    eurgbp = {"contract_size": 100_000.0, "step_size": 0.01, "asset_price": 0.85, "tick_size": 0.00001, "tick_value": 1.27}
    arguments = {"account_balance": 100_000.0, "atr_val": 0.0010, **eurgbp}
    # Risk: 1 000 / (0.0020 / 0.00001 x 1.27) = 3.937 lots. Cap at 3x: 300 000 / 107 950 = 2.779.
    assert sizer.calculate_position_size(max_leverage=3.0, **arguments) == 2.77
    assert sizer.calculate_position_size(max_leverage=3.0, risk_pct=0.5, **arguments) == 1.96


def test_tick_value_and_tick_size_go_together(sizer: ATRSizer) -> None:
    arguments = {"account_balance": 100_000.0, "atr_val": 0.50, **USDJPY}
    with pytest.raises(RiskError, match="tick_value and tick_size go together"):
        sizer.calculate_position_size(tick_value=0.6667, **arguments)
    with pytest.raises(RiskError, match="tick_value and tick_size go together"):
        sizer.calculate_position_size(tick_size=0.001, **arguments)


@pytest.mark.parametrize("name", ["tick_value", "tick_size"])
@pytest.mark.parametrize("value", ["0.001", [0.001], True, 0.0, -0.001, float("nan"), float("inf")])
def test_invalid_tick_arguments(sizer: ATRSizer, name: str, value: object) -> None:
    arguments: dict[str, Any] = {"account_balance": 100_000.0, "atr_val": 0.50, **USDJPY, **USDJPY_TICK, name: value}
    with pytest.raises(RiskError, match=name):
        sizer.calculate_position_size(**arguments)


def test_all_broker_constraints_at_once(sizer: ATRSizer) -> None:
    arguments = {"account_balance": 100_000.0, "atr_val": 0.50, "max_leverage": 3.0, **USDJPY, **USDJPY_TICK}
    assert sizer.calculate_position_size(min_volume=0.01, max_volume=1.0, **arguments) == 1.0
    assert sizer.calculate_position_size(min_volume=2.0, **arguments) == 0.0


# --- Sized position: lots, notional and risk --------------------------------


def test_size_position_reports_value_and_risk_in_the_account_currency(sizer: ATRSizer) -> None:
    from src.risk.position_sizer import LIMITED_BY_LEVERAGE, LIMITED_BY_MAX_VOLUME, LIMITED_BY_RISK

    arguments = {"account_balance": 100_000.0, "atr_val": 20.0, **GOLD}
    capped = sizer.size_position(**arguments)  # 10 % cap: 0.04 lots
    assert (capped.lots, capped.limited_by) == (0.04, LIMITED_BY_LEVERAGE)
    assert capped.nominal_value == pytest.approx(0.04 * 100.0 * 2_400.0)
    assert capped.risk_amount == pytest.approx(0.04 * 100.0 * 40.0)
    assert capped.stop_distance == 40.0

    by_risk = sizer.size_position(max_leverage=3.0, **arguments)
    assert (by_risk.lots, by_risk.limited_by) == (0.25, LIMITED_BY_RISK)
    assert by_risk.risk_amount == pytest.approx(1_000.0)  # exactly 1 %
    assert sizer.size_position(max_leverage=3.0, max_volume=0.1, **arguments).limited_by == LIMITED_BY_MAX_VOLUME

    yen = sizer.size_position(account_balance=100_000.0, atr_val=0.50, max_leverage=3.0, **USDJPY, **USDJPY_TICK)
    assert yen.lots == 1.5
    assert yen.nominal_value == pytest.approx(150_000.0)  # dollars, not yen
    assert yen.risk_amount == pytest.approx(1_000.0)


def test_size_position_and_calculate_position_size_agree(sizer: ATRSizer) -> None:
    for arguments in (
        {"account_balance": 100_000.0, "atr_val": 20.0, **GOLD},
        {"account_balance": 10_000.0, "atr_val": 0.0020, "max_leverage": 3.0, **EURUSD},
        {"account_balance": 1_000.0, "atr_val": 20.0, **GOLD},  # not enough: 0.0
        {"account_balance": 100_000.0, "atr_val": 20.0, "min_volume": 0.05, **GOLD},  # below the minimum: 0.0
    ):
        sized = sizer.size_position(**arguments)
        assert sized.lots == sizer.calculate_position_size(**arguments)
        if sized.lots == 0.0:
            assert sized.nominal_value == 0.0 and sized.risk_amount == 0.0
