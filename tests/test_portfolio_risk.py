"""Tests for the portfolio risk check (``PortfolioRiskManager``)."""

from __future__ import annotations

from typing import Any

import pytest
from loguru import logger

from src.risk.portfolio_risk import (
    APPROVED,
    CORRELATED_EXPOSURE_BLOCKED,
    MAX_HEAT_EXCEEDED,
    MAX_LEVERAGE_EXCEEDED,
    ActivePosition,
    PortfolioRiskManager,
)
from src.risk.position_sizer import ATRSizer
from src.risk.turtle_engine import RiskError

BALANCE = 100_000.0


@pytest.fixture
def manager() -> PortfolioRiskManager:
    """With total notional capped at half the balance, so the examples stay small."""
    return PortfolioRiskManager(max_portfolio_leverage=0.5, max_portfolio_heat_pct=3.0)


def _position(symbol: str = "BTCUSDT", direction: int = 1, nominal: float = 10_000.0, risk: float = 500.0,
              asset_class: str | None = None) -> ActivePosition:
    if asset_class is None:
        asset_class = "crypto" if symbol.endswith("USDT") else "commodity" if symbol.startswith("XAU") else "forex"
    return ActivePosition(symbol, direction, nominal, risk, asset_class)


def _ask(manager: PortfolioRiskManager, positions: list[ActivePosition], symbol: str = "ETHUSDT", direction: int = 1,
         nominal: float = 10_000.0, risk: float = 500.0, asset_class: str | None = None,
         balance: float = BALANCE) -> tuple[bool, str]:
    candidate = _position(symbol, direction, nominal, risk, asset_class)
    return manager.can_open_position(balance, positions, nominal, risk, symbol, direction, candidate.asset_class)


def _log(level: str = "INFO") -> tuple[list[str], int]:
    lines: list[str] = []
    return lines, logger.add(lines.append, format="{level}|{message}", level=level)


# --- Approval ---------------------------------------------------------------


def test_empty_portfolio_approves(manager: PortfolioRiskManager) -> None:
    assert _ask(manager, []) == (True, APPROVED)
    assert manager.can_open_position(BALANCE, [], 10_000.0, 500.0, "EURUSD", 1, "forex") == (True, "APPROVED")


def test_lightly_loaded_portfolio_approves(manager: PortfolioRiskManager) -> None:
    positions = [_position("BTCUSDT"), _position("XAUUSD", risk=800.0)]
    # Notional 30 000 (0.3x) and risk 1 800 (1.8 %): within both limits.
    assert _ask(manager, positions, "ETHUSDT") == (True, APPROVED)
    assert _ask(manager, positions, "EURUSD", direction=-1) == (True, APPROVED)


def test_defaults() -> None:
    manager = PortfolioRiskManager()
    assert _ask(manager, [], nominal=100_000.0, risk=4_000.0) == (True, APPROVED)  # exactly 1.0x and 4 %
    assert _ask(manager, [], nominal=100_001.0, risk=100.0) == (False, MAX_LEVERAGE_EXCEEDED)
    assert _ask(manager, [], nominal=1_000.0, risk=4_001.0) == (False, MAX_HEAT_EXCEEDED)
    # Correlation is only checked in forex: the crypto basket is left free.
    assert _ask(manager, [_position("EURUSD")], "GBPUSD") == (False, CORRELATED_EXPOSURE_BLOCKED)
    assert _ask(manager, [_position("BTCUSDT"), _position("ETHUSDT")], "SOLUSDT") == (True, APPROVED)
    assert _ask(manager, [_position("XAUUSD")], "XAUUSD") == (True, APPROVED)


# --- Total notional leverage ------------------------------------------------


def test_rejects_when_total_nominal_exceeds_the_leverage_limit(manager: PortfolioRiskManager) -> None:
    positions = [_position("BTCUSDT", nominal=25_000.0, risk=100.0), _position("XAUUSD", nominal=20_000.0, risk=100.0)]
    assert _ask(manager, positions, nominal=5_000.0, risk=100.0) == (True, APPROVED)  # 50 000: exactly 0.5x
    assert _ask(manager, positions, nominal=5_500.0, risk=100.0) == (False, MAX_LEVERAGE_EXCEEDED)
    assert _ask(manager, positions, nominal=10_000.0, risk=100.0) == (False, "MAX_LEVERAGE_EXCEEDED")


def test_shorts_add_to_the_nominal_instead_of_netting(manager: PortfolioRiskManager) -> None:
    positions = [_position("BTCUSDT", direction=-1, nominal=45_000.0, risk=100.0)]
    assert _ask(manager, positions, "ETHUSDT", direction=1, nominal=10_000.0, risk=100.0) == (False, MAX_LEVERAGE_EXCEEDED)


def test_the_leverage_limit_scales_with_the_balance(manager: PortfolioRiskManager) -> None:
    positions = [_position("BTCUSDT", nominal=45_000.0, risk=100.0)]
    assert _ask(manager, positions, nominal=10_000.0, risk=100.0, balance=110_000.0) == (True, APPROVED)
    assert _ask(manager, positions, nominal=10_000.0, risk=100.0, balance=109_000.0) == (False, MAX_LEVERAGE_EXCEEDED)


def test_ten_default_sized_positions_fill_the_portfolio() -> None:
    # ATRSizer caps each position at 10 % of the balance and the portfolio at 100 %: ten coins fit.
    manager = PortfolioRiskManager()
    lots = ATRSizer().calculate_position_size(
        account_balance=BALANCE, atr_val=372.77, contract_size=1.0, step_size=0.001, asset_price=85_669.0
    )
    nominal = lots * 85_669.0
    positions: list[ActivePosition] = []
    for index in range(10):
        symbol = f"COIN{index}USDT"
        assert _ask(manager, positions, symbol, nominal=nominal, risk=100.0) == (True, APPROVED)
        positions.append(_position(symbol, nominal=nominal, risk=100.0))
    assert sum(position.nominal_value for position in positions) <= BALANCE  # no debt
    assert _ask(manager, positions, "COIN10USDT", nominal=nominal, risk=100.0) == (False, MAX_LEVERAGE_EXCEEDED)


def test_floating_point_noise_at_the_limit_is_not_a_rejection() -> None:
    assert 0.1 + 0.1 + 0.1 > 0.3  # the floating point sum goes over by a hair
    leverage = PortfolioRiskManager(max_portfolio_leverage=0.3, max_portfolio_heat_pct=100.0)
    positions = [_position("BTCUSDT", nominal=0.1, risk=0.0), _position("XAUUSD", nominal=0.1, risk=0.0)]
    assert _ask(leverage, positions, nominal=0.1, risk=0.001, balance=1.0) == (True, APPROVED)
    assert _ask(leverage, positions, nominal=0.1001, risk=0.001, balance=1.0) == (False, MAX_LEVERAGE_EXCEEDED)
    heat = PortfolioRiskManager(max_portfolio_leverage=3.0, max_portfolio_heat_pct=0.3)
    positions = [_position("BTCUSDT", nominal=1.0, risk=0.1), _position("XAUUSD", nominal=1.0, risk=0.1)]
    assert _ask(heat, positions, nominal=1.0, risk=0.1, balance=100.0) == (True, APPROVED)
    assert _ask(heat, positions, nominal=1.0, risk=0.1001, balance=100.0) == (False, MAX_HEAT_EXCEEDED)


# --- Open risk (heat) -------------------------------------------------------


def test_rejects_when_open_risk_exceeds_the_heat_limit(manager: PortfolioRiskManager) -> None:
    positions = [_position("BTCUSDT", risk=1_000.0), _position("XAUUSD", risk=1_000.0)]
    assert _ask(manager, positions, risk=1_000.0) == (True, APPROVED)  # 3 000: exactly 3 %
    assert _ask(manager, positions, risk=1_500.0) == (False, MAX_HEAT_EXCEEDED)
    assert _ask(manager, positions, risk=1_000.01) == (False, "MAX_HEAT_EXCEEDED")


def test_a_position_with_its_stop_at_breakeven_frees_its_risk(manager: PortfolioRiskManager) -> None:
    risky = [_position("BTCUSDT", risk=1_500.0), _position("XAUUSD", risk=1_500.0)]
    assert _ask(manager, risky, risk=500.0) == (False, MAX_HEAT_EXCEEDED)
    protected = [_position("BTCUSDT", risk=0.0), _position("XAUUSD", risk=1_500.0)]
    assert _ask(manager, protected, risk=500.0) == (True, APPROVED)


def test_leverage_is_checked_before_heat_and_heat_before_correlation(manager: PortfolioRiskManager) -> None:
    eurusd = _position("EURUSD", nominal=45_000.0, risk=2_800.0)
    assert _ask(manager, [eurusd], "GBPUSD", nominal=10_000.0, risk=500.0) == (False, MAX_LEVERAGE_EXCEEDED)
    assert _ask(manager, [eurusd], "GBPUSD", nominal=5_000.0, risk=500.0) == (False, MAX_HEAT_EXCEEDED)
    assert _ask(manager, [eurusd], "GBPUSD", nominal=5_000.0, risk=200.0) == (False, CORRELATED_EXPOSURE_BLOCKED)


# --- Correlated exposure ----------------------------------------------------


def test_long_gbpusd_is_blocked_while_long_eurusd_is_open(manager: PortfolioRiskManager) -> None:
    positions = [_position("EURUSD", direction=1)]
    assert _ask(manager, positions, "GBPUSD", direction=1) == (False, CORRELATED_EXPOSURE_BLOCKED)
    assert _ask(manager, positions, "GBPUSD", direction=-1) == (True, APPROVED)  # opposite direction
    shorts = [_position("EURUSD", direction=-1)]
    assert _ask(manager, shorts, "GBPUSD", direction=-1) == (False, "CORRELATED_EXPOSURE_BLOCKED")
    assert _ask(manager, shorts, "GBPUSD", direction=1) == (True, APPROVED)


def test_the_rule_looks_at_currencies_not_at_the_order_side(manager: PortfolioRiskManager) -> None:
    long_eurusd = [_position("EURUSD", direction=1)]  # long euros, short dollars
    # Selling USDJPY is also selling dollars: same bet, even though one order is a buy and the other a sell.
    assert _ask(manager, long_eurusd, "USDJPY", direction=-1) == (False, CORRELATED_EXPOSURE_BLOCKED)
    # Buying USDJPY or USDCHF is buying dollars: it offsets, it does not add up.
    assert _ask(manager, long_eurusd, "USDJPY", direction=1) == (True, APPROVED)
    assert _ask(manager, long_eurusd, "USDCHF", direction=1) == (True, APPROVED)
    # A cross shares the euro: buying EURGBP repeats the euro buy; selling it does not.
    assert _ask(manager, long_eurusd, "EURGBP", direction=1) == (False, CORRELATED_EXPOSURE_BLOCKED)
    assert _ask(manager, long_eurusd, "EURGBP", direction=-1) == (True, APPROVED)
    # With no currency in common there is nothing to block.
    assert _ask(manager, long_eurusd, "AUDJPY", direction=1) == (True, APPROVED)


def test_a_second_position_in_the_same_pair_is_also_correlated(manager: PortfolioRiskManager) -> None:
    positions = [_position("EURUSD", direction=1)]
    assert _ask(manager, positions, "EURUSD", direction=1) == (False, CORRELATED_EXPOSURE_BLOCKED)


@pytest.mark.parametrize(("held", "wanted"), [("EURUSD.m", "GBPUSDm"), ("eurusd", "gbpusd"), ("EURUSD-ECN", "GBPUSD.pro")])
def test_broker_suffixes_and_case_do_not_hide_the_pair(manager: PortfolioRiskManager, held: str, wanted: str) -> None:
    assert _ask(manager, [_position(held, asset_class="forex")], wanted, asset_class="forex") == (
        False, CORRELATED_EXPOSURE_BLOCKED,
    )


def test_other_asset_classes_are_not_limited_by_default(manager: PortfolioRiskManager) -> None:
    assert _ask(manager, [_position("BTCUSDT")], "ETHUSDT") == (True, APPROVED)
    assert _ask(manager, [_position("EURUSD")], "XAUUSD") == (True, APPROVED)  # gold is not a currency pair
    assert _ask(manager, [_position("XAUUSD")], "GBPUSD") == (True, APPROVED)
    assert _ask(manager, [_position("BTCUSDT")], "EURUSD") == (True, APPROVED)


def test_crypto_can_be_treated_as_one_correlated_class() -> None:
    manager = PortfolioRiskManager(correlated_classes=("forex", "crypto"))
    held = [_position("BTCUSDT", direction=1)]
    assert _ask(manager, held, "ETHUSDT", direction=1) == (False, CORRELATED_EXPOSURE_BLOCKED)
    assert _ask(manager, held, "ETHUSDT", direction=-1) == (True, APPROVED)
    assert _ask(manager, held, "EURUSD", direction=1) == (True, APPROVED)  # another class, another bet
    assert _ask(PortfolioRiskManager(correlated_classes=()), [_position("EURUSD")], "GBPUSD") == (True, APPROVED)


def test_a_forex_symbol_that_is_not_a_pair_falls_back_to_the_class_rule(manager: PortfolioRiskManager) -> None:
    held = [_position("DXY", direction=1, asset_class="forex")]
    assert _ask(manager, held, "USDX", direction=1, asset_class="forex") == (False, CORRELATED_EXPOSURE_BLOCKED)
    assert _ask(manager, held, "USDX", direction=-1, asset_class="forex") == (True, APPROVED)


def test_more_correlated_positions_can_be_allowed() -> None:
    manager = PortfolioRiskManager(max_correlated_positions=2)
    one = [_position("EURUSD")]
    two = [*one, _position("GBPUSD")]
    assert _ask(manager, one, "GBPUSD") == (True, APPROVED)
    assert _ask(manager, two, "AUDUSD") == (False, CORRELATED_EXPOSURE_BLOCKED)
    assert _ask(manager, two, "USDCAD", direction=1) == (True, APPROVED)


# --- Logging ----------------------------------------------------------------


def test_approvals_are_logged_at_info_and_rejections_at_warning(manager: PortfolioRiskManager) -> None:
    lines, sink = _log()
    try:
        _ask(manager, [_position("EURUSD")], "XAUUSD")
        _ask(manager, [_position("EURUSD")], "GBPUSD")
        _ask(manager, [_position("EURUSD", nominal=45_000.0)], "XAUUSD")
        _ask(manager, [_position("EURUSD", risk=2_800.0)], "XAUUSD")
    finally:
        logger.remove(sink)

    assert lines == [
        "INFO|Entry on XAUUSD approved: portfolio notional 20000.00 (0.200x of 0.5x) | "
        "open risk 1000.00 (1.000 % of 3 %) | 1 prior positions\n",
        "WARNING|Entry on GBPUSD rejected (CORRELATED_EXPOSURE_BLOCKED): EURUSD already bets the same way "
        "and the maximum of correlated positions is 1\n",
        "WARNING|Entry on XAUUSD rejected (MAX_LEVERAGE_EXCEEDED): portfolio notional would reach 55000.00 "
        "(0.550x the balance of 100000.00) and the maximum is 0.5x\n",
        "WARNING|Entry on XAUUSD rejected (MAX_HEAT_EXCEEDED): open risk would reach 3300.00 "
        "(3.300 % of the balance of 100000.00) and the maximum is 3 %\n",
    ]


def test_the_positions_list_is_not_modified(manager: PortfolioRiskManager) -> None:
    positions = [_position("EURUSD"), _position("BTCUSDT")]
    before = list(positions)
    _ask(manager, positions, "GBPUSD")
    _ask(manager, positions, "ETHUSDT")
    assert positions == before
    assert _ask(manager, tuple(positions), "ETHUSDT") == (True, APPROVED)  # a tuple works too


# --- Corrupt data -----------------------------------------------------------

INVALID_NUMBERS = [None, "100000", [1.0], True, float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("balance", [*INVALID_NUMBERS, 0.0, -500.0])
def test_invalid_balance(manager: PortfolioRiskManager, balance: Any) -> None:
    with pytest.raises(RiskError, match="account_balance must be a positive number"):
        manager.can_open_position(balance, [], 10_000.0, 500.0, "EURUSD", 1, "forex")


@pytest.mark.parametrize("value", [*INVALID_NUMBERS, 0.0, -1.0])
def test_invalid_new_nominal_and_risk(manager: PortfolioRiskManager, value: Any) -> None:
    with pytest.raises(RiskError, match="new_nominal"):
        manager.can_open_position(BALANCE, [], value, 500.0, "EURUSD", 1, "forex")
    # A new order with no declared risk (stop) cannot be evaluated.
    with pytest.raises(RiskError, match="new_risk_usd"):
        manager.can_open_position(BALANCE, [], 10_000.0, value, "EURUSD", 1, "forex")


@pytest.mark.parametrize("symbol", [None, "", "   ", 123, ["EURUSD"]])
def test_invalid_symbol(manager: PortfolioRiskManager, symbol: Any) -> None:
    with pytest.raises(RiskError, match="new_symbol"):
        manager.can_open_position(BALANCE, [], 10_000.0, 500.0, symbol, 1, "forex")


@pytest.mark.parametrize("direction", [None, 0, 2, -2, 1.0, "1", True, "long"])
def test_invalid_direction(manager: PortfolioRiskManager, direction: Any) -> None:
    with pytest.raises(RiskError, match="new_direction must be"):
        manager.can_open_position(BALANCE, [], 10_000.0, 500.0, "EURUSD", direction, "forex")


@pytest.mark.parametrize("asset_class", [None, "", "stocks", "Forex", "FOREX", 1])
def test_invalid_asset_class(manager: PortfolioRiskManager, asset_class: Any) -> None:
    with pytest.raises(RiskError, match="new_asset_class must be one of"):
        manager.can_open_position(BALANCE, [], 10_000.0, 500.0, "EURUSD", 1, asset_class)


@pytest.mark.parametrize("positions", [None, "EURUSD", 5, {"symbol": "EURUSD"}, [{"symbol": "EURUSD"}], [None], [("EURUSD", 1)]])
def test_invalid_positions(manager: PortfolioRiskManager, positions: Any) -> None:
    with pytest.raises(RiskError, match="current_positions"):
        manager.can_open_position(BALANCE, positions, 10_000.0, 500.0, "GBPUSD", 1, "forex")


@pytest.mark.parametrize(
    "fields",
    [
        {"symbol": ""}, {"symbol": None},
        {"direction": 0}, {"direction": True}, {"direction": 1.0},
        {"nominal_value": 0.0}, {"nominal_value": -10_000.0}, {"nominal_value": float("nan")},
        {"nominal_value": float("inf")}, {"nominal_value": "10000"}, {"nominal_value": None},
        {"risk_usd": -1.0}, {"risk_usd": float("nan")}, {"risk_usd": float("inf")}, {"risk_usd": None},
        {"asset_class": "stocks"}, {"asset_class": None},
    ],
)
def test_a_corrupt_position_cannot_be_built(fields: dict[str, Any]) -> None:
    valid: dict[str, Any] = {"symbol": "EURUSD", "direction": 1, "nominal_value": 10_000.0, "risk_usd": 500.0,
                             "asset_class": "forex"}
    with pytest.raises(RiskError, match=next(iter(fields))):
        ActivePosition(**{**valid, **fields})


def test_errors_are_value_errors(manager: PortfolioRiskManager) -> None:
    with pytest.raises(ValueError):
        manager.can_open_position(-500.0, [], 10_000.0, 500.0, "EURUSD", 1, "forex")


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_portfolio_leverage": 0.0}, {"max_portfolio_leverage": -0.5}, {"max_portfolio_leverage": float("nan")},
        {"max_portfolio_leverage": "0.5"}, {"max_portfolio_leverage": None},
        {"max_portfolio_heat_pct": 0.0}, {"max_portfolio_heat_pct": 101.0}, {"max_portfolio_heat_pct": float("inf")},
        {"max_correlated_positions": 0}, {"max_correlated_positions": 1.5}, {"max_correlated_positions": True},
        {"max_correlated_positions": None},
        {"correlated_classes": "forex"}, {"correlated_classes": ("stocks",)}, {"correlated_classes": None},
    ],
)
def test_invalid_limits(arguments: dict[str, Any]) -> None:
    with pytest.raises(RiskError, match=next(iter(arguments))):
        PortfolioRiskManager(**arguments)
