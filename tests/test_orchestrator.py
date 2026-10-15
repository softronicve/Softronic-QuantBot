"""Tests for the basket orchestrator (``BasketOrchestrator``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src.engine.orchestrator import (
    CLOSE_LONG,
    HOLD,
    INVALID_DATA,
    MARKET,
    NO_DATA,
    NO_INSTRUMENT,
    NOT_MANAGED,
    OPEN_LONG,
    BasketOrchestrator,
    CandidateSignal,
    OrderIntent,
)
from src.risk.portfolio_risk import (
    CORRELATED_EXPOSURE_BLOCKED,
    MAX_HEAT_EXCEEDED,
    MAX_LEVERAGE_EXCEEDED,
    ActivePosition,
    PortfolioRiskManager,
)
from src.risk.turtle_engine import RiskError
from src.strategies.trend_breakout import (
    BREAKOUT,
    EXIT_CHANNEL_BROKEN,
    NO_BREAKOUT,
    SIZE_TOO_SMALL,
    TREND_INTACT,
    Action,
    InstrumentSpec,
    StrategyError,
    TrendBreakoutStrategy,
)

DAY_MS = 86_400_000
BALANCE = 100_000.0
COINS = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT", "GGGUSDT", "HHHUSDT"]
INSTRUMENTS = {symbol: InstrumentSpec(symbol, "crypto", step_size=0.00001) for symbol in COINS} | {
    "EURUSD": InstrumentSpec("EURUSD", "forex", step_size=0.01, contract_size=1_000.0),
    "GBPUSD": InstrumentSpec("GBPUSD", "forex", step_size=0.01, contract_size=1_000.0),
}


def _candles(last: float, n: int = 150) -> pd.DataFrame:
    """A range between 98 and 102 (highs up to 103, lows down to 97) and a last candle at `last`.

    In this range the ATR is around 2.9: a position of 10 % of the balance risks 0.56 %.
    """
    close = np.array([*(100.0 + 2.0 * np.sin(np.arange(n - 1) * 0.7)), last])
    open_ = np.concatenate([[close[0]], close[:-1]])
    open_time = np.arange(n, dtype=np.int64) * DAY_MS
    return pd.DataFrame({
        "open_time": open_time, "open": open_, "high": np.maximum(open_, close) + 1.0,
        "low": np.minimum(open_, close) - 1.0, "close": close, "volume": 1_000.0,
        "close_time": open_time + DAY_MS - 1,
    })


BREAKS = 104.0  # closes above the 20-candle high (~103)
QUIET = 100.0  # inside the range
FALLS = 90.0  # falls below the 10-candle low


def _orchestrator(**limits: Any) -> BasketOrchestrator:
    return BasketOrchestrator(PortfolioRiskManager(**limits), TrendBreakoutStrategy(), INSTRUMENTS)


def _held(symbol: str, risk: float = 558.0, nominal: float = 10_000.0, direction: int = 1) -> ActivePosition:
    return ActivePosition(symbol, direction, nominal, risk, INSTRUMENTS[symbol].asset_class if symbol in INSTRUMENTS else "crypto")


def _actions(intents: list[OrderIntent]) -> dict[str, tuple[str, str]]:
    return {intent.symbol: (intent.action, intent.reason) for intent in intents}


def _opened(intents: list[OrderIntent]) -> list[str]:
    return [intent.symbol for intent in intents if intent.action == OPEN_LONG]


def _log(level: str = "INFO") -> tuple[list[str], int]:
    lines: list[str] = []
    return lines, logger.add(lines.append, format="{level}|{message}", level=level)


# --- Basic cycle ------------------------------------------------------------


def test_an_empty_basket_does_nothing() -> None:
    assert _orchestrator().evaluate_daily_cycle({}, [], BALANCE) == []


def test_one_intent_per_symbol_with_its_reason() -> None:
    market = {"AAAUSDT": _candles(BREAKS), "BBBUSDT": _candles(QUIET), "CCCUSDT": _candles(FALLS), "DDDUSDT": _candles(QUIET)}
    positions = [_held("CCCUSDT"), _held("DDDUSDT")]
    intents = _orchestrator().evaluate_daily_cycle(market, positions, BALANCE)
    assert _actions(intents) == {
        "CCCUSDT": (CLOSE_LONG, EXIT_CHANNEL_BROKEN),
        "AAAUSDT": (OPEN_LONG, BREAKOUT),
        "DDDUSDT": (HOLD, TREND_INTACT),
        "BBBUSDT": (HOLD, NO_BREAKOUT),
    }
    # Closes go first, then buys, and last whatever does not change.
    assert [intent.action for intent in intents] == [CLOSE_LONG, OPEN_LONG, HOLD, HOLD]
    assert all(intent.order_type == MARKET for intent in intents)


def test_an_open_order_carries_volume_and_stop_and_a_close_does_not() -> None:
    market = {"AAAUSDT": _candles(BREAKS), "CCCUSDT": _candles(FALLS)}
    close, open_ = _orchestrator().evaluate_daily_cycle(market, [_held("CCCUSDT")], BALANCE)
    expected = TrendBreakoutStrategy().evaluate(_candles(BREAKS), INSTRUMENTS["AAAUSDT"], account_balance=BALANCE)
    assert open_ == OrderIntent("AAAUSDT", OPEN_LONG, expected.volume, MARKET, expected.stop_price, BREAKOUT)
    assert open_.volume == pytest.approx(96.15384) and open_.stop_loss == pytest.approx(98.196, abs=0.001)
    # A close is always of the whole position: it carries no volume or stop.
    assert close == OrderIntent("CCCUSDT", CLOSE_LONG, 0.0, MARKET, None, EXIT_CHANNEL_BROKEN)


def test_a_symbol_in_position_never_gets_a_second_entry() -> None:
    intents = _orchestrator().evaluate_daily_cycle({"AAAUSDT": _candles(110.0)}, [_held("AAAUSDT")], BALANCE)
    assert _actions(intents) == {"AAAUSDT": (HOLD, TREND_INTACT)}


def test_the_inputs_are_not_modified() -> None:
    market = {"AAAUSDT": _candles(BREAKS), "CCCUSDT": _candles(FALLS)}
    positions = [_held("CCCUSDT")]
    before_market, before_positions = {k: v.copy() for k, v in market.items()}, list(positions)
    _orchestrator().evaluate_daily_cycle(market, positions, BALANCE)
    assert positions == before_positions and list(market) == list(before_market)
    for symbol, frame in market.items():
        pd.testing.assert_frame_equal(frame, before_market[symbol])


# --- Step A: exits free risk in the same cycle ------------------------------


def test_an_exit_frees_risk_for_a_new_entry_in_the_same_cycle() -> None:
    # Seven positions of 0.558 % fill the 4 % of open risk: an eighth does not fit.
    positions = [_held(symbol) for symbol in COINS[:7]]
    market = {symbol: _candles(QUIET) for symbol in COINS[:7]} | {"HHHUSDT": _candles(BREAKS)}
    full = _actions(_orchestrator().evaluate_daily_cycle(market, positions, BALANCE))
    assert full["HHHUSDT"] == (HOLD, MAX_HEAT_EXCEEDED)

    # If one of the seven breaks its exit channel, its risk is freed for the candidate.
    market["AAAUSDT"] = _candles(FALLS)
    intents = _orchestrator().evaluate_daily_cycle(market, positions, BALANCE)
    assert _actions(intents)["AAAUSDT"] == (CLOSE_LONG, EXIT_CHANNEL_BROKEN)
    assert _actions(intents)["HHHUSDT"] == (OPEN_LONG, BREAKOUT)
    assert [intent.action for intent in intents[:2]] == [CLOSE_LONG, OPEN_LONG]  # and in that order


def test_an_exit_frees_nominal_too() -> None:
    orchestrator = _orchestrator(max_portfolio_heat_pct=50.0)
    positions = [_held(f"X{i}USDT", risk=10.0) for i in range(9)] + [_held("AAAUSDT", risk=10.0)]  # ten of 10 %
    market = {"AAAUSDT": _candles(QUIET), "HHHUSDT": _candles(BREAKS)}
    assert _actions(orchestrator.evaluate_daily_cycle(market, positions, BALANCE))["HHHUSDT"] == (HOLD, MAX_LEVERAGE_EXCEEDED)
    market["AAAUSDT"] = _candles(FALLS)
    assert _actions(orchestrator.evaluate_daily_cycle(market, positions, BALANCE))["HHHUSDT"] == (OPEN_LONG, BREAKOUT)


def test_a_position_that_is_closing_does_not_reenter() -> None:
    intents = _orchestrator().evaluate_daily_cycle({"AAAUSDT": _candles(FALLS)}, [_held("AAAUSDT")], BALANCE)
    assert [(intent.symbol, intent.action) for intent in intents] == [("AAAUSDT", CLOSE_LONG)]


# --- Step B: ranking by breakout strength -----------------------------------


def test_candidates_are_ranked_by_breakout_strength() -> None:
    market = {"AAAUSDT": _candles(103.5), "BBBUSDT": _candles(108.0), "CCCUSDT": _candles(105.0), "DDDUSDT": _candles(QUIET)}
    candidates = _orchestrator().scan_candidates(market, set(), BALANCE)
    assert [c.symbol for c in candidates] == ["BBBUSDT", "CCCUSDT", "AAAUSDT", "DDDUSDT"]
    assert [c.signal for c in candidates] == [Action.ENTER_LONG] * 3 + [Action.HOLD]
    strengths = [c.breakout_strength for c in candidates]
    assert strengths[0] > strengths[1] > strengths[2] > 0.0 and strengths[3] == 0.0

    strongest = candidates[0]
    reference = TrendBreakoutStrategy().evaluate(market["BBBUSDT"], INSTRUMENTS["BBBUSDT"], account_balance=BALANCE)
    assert strongest.breakout_strength == pytest.approx((108.0 - reference.entry_channel) / reference.atr)
    assert strongest == CandidateSignal(
        "BBBUSDT", Action.ENTER_LONG, 108.0, strongest.breakout_strength, reference.nominal_value,
        reference.risk_amount, reference.volume, reference.stop_price, reason=BREAKOUT, asset_class="crypto",
        atr_pct=reference.atr / 108.0,
    )
    quiet = candidates[3]
    assert (quiet.nominal_value, quiet.risk_usd, quiet.lot_size, quiet.stop_loss_price) == (0.0, 0.0, 0.0, 0.0)
    assert quiet.atr_pct == 0.0


def test_with_room_for_one_only_the_strongest_breakout_is_approved() -> None:
    # Six positions take up 3.35 % of risk: there is room for only one of 0.56 %.
    positions = [_held(symbol) for symbol in COINS[:6]]
    held = {symbol: _candles(QUIET) for symbol in COINS[:6]}
    signals = {"EEEUSDT": 103.5, "GGGUSDT": 108.0, "HHHUSDT": 105.0}
    del held["EEEUSDT"]
    positions = [p for p in positions if p.symbol != "EEEUSDT"] + [_held("EURUSD", nominal=1_000.0)]
    held["EURUSD"] = _candles(QUIET)
    market = held | {symbol: _candles(last) for symbol, last in signals.items()}

    intents = _orchestrator().evaluate_daily_cycle(market, positions, BALANCE)
    assert _opened(intents) == ["GGGUSDT"]  # the strongest one, not the first in the dict
    assert _actions(intents)["HHHUSDT"] == (HOLD, MAX_HEAT_EXCEEDED)
    assert _actions(intents)["EEEUSDT"] == (HOLD, MAX_HEAT_EXCEEDED)


@pytest.mark.parametrize("order", [["EEEUSDT", "GGGUSDT", "HHHUSDT"], ["HHHUSDT", "EEEUSDT", "GGGUSDT"], ["GGGUSDT", "HHHUSDT", "EEEUSDT"]])
def test_the_ranking_does_not_depend_on_the_order_of_the_data(order: list[str]) -> None:
    signals = {"EEEUSDT": 103.5, "GGGUSDT": 108.0, "HHHUSDT": 105.0}
    positions = [_held(symbol) for symbol in COINS[:4]] + [_held("EURUSD", nominal=1_000.0), _held("GBPUSD", nominal=1_000.0, direction=-1)]
    market = {symbol: _candles(signals[symbol]) for symbol in order}
    intents = _orchestrator().evaluate_daily_cycle(market, positions, BALANCE)
    assert _opened(intents) == ["GGGUSDT"]


def test_equal_strength_is_broken_by_symbol_name() -> None:
    market = {"CCCUSDT": _candles(BREAKS), "AAAUSDT": _candles(BREAKS), "BBBUSDT": _candles(BREAKS)}
    assert _opened(_orchestrator().evaluate_daily_cycle(market, [], BALANCE)) == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]


def test_equal_strength_goes_first_to_the_larger_atr_over_price(monkeypatch: pytest.MonkeyPatch) -> None:
    # (strength, ATR / price): strength decides; at equal strength, relative ATR; with all equal, the symbol.
    made = {"AAAUSDT": (1.5, 0.02), "BBBUSDT": (1.5, 0.06), "CCCUSDT": (2.0, 0.01), "DDDUSDT": (1.5, 0.06)}

    def candidate(self: BasketOrchestrator, symbol: str, candles: Any, account_balance: float) -> CandidateSignal:
        strength, atr_pct = made[symbol]
        return CandidateSignal(symbol, Action.ENTER_LONG, 100.0, strength, 1_000.0, 100.0, 10.0, 90.0, atr_pct=atr_pct)

    monkeypatch.setattr(BasketOrchestrator, "_candidate", candidate)
    ranked = _orchestrator().scan_candidates(dict.fromkeys(made), set(), BALANCE)
    assert [c.symbol for c in ranked] == ["CCCUSDT", "BBBUSDT", "DDDUSDT", "AAAUSDT"]


def test_fifteen_simultaneous_breakouts_fill_the_risk_by_strength_not_by_name() -> None:
    # The later the symbol in the alphabet, the stronger its breakout.
    symbols = [f"{chr(65 + i) * 3}USDT" for i in range(15)]
    instruments = {symbol: InstrumentSpec(symbol, "crypto", step_size=0.00001) for symbol in symbols}
    market = {symbol: _candles(103.5 + 0.5 * i) for i, symbol in enumerate(symbols)}
    orchestrator = BasketOrchestrator(PortfolioRiskManager(), TrendBreakoutStrategy(), instruments)

    intents = orchestrator.evaluate_daily_cycle(market, [], BALANCE)
    opened = _opened(intents)
    assert opened == symbols[::-1][:7]  # the seven strongest; the eighth no longer fits in the 4 %
    assert "AAAUSDT" not in opened
    rejected = {intent.reason for intent in intents if intent.action == HOLD}
    assert rejected <= {MAX_HEAT_EXCEEDED, MAX_LEVERAGE_EXCEEDED}


def test_stronger_breakouts_get_the_risk_first_when_several_fit() -> None:
    positions = [_held(symbol) for symbol in COINS[:5]]  # 2.79 %: two more fit
    market = {"FFFUSDT": _candles(103.5), "GGGUSDT": _candles(108.0), "HHHUSDT": _candles(105.0)}
    intents = _orchestrator().evaluate_daily_cycle(market, positions, BALANCE)
    assert _opened(intents) == ["GGGUSDT", "HHHUSDT"]
    assert _actions(intents)["FFFUSDT"] == (HOLD, MAX_HEAT_EXCEEDED)


# --- Step C: open risk ceiling and the other limits -------------------------


def test_orders_stop_at_the_four_percent_heat_ceiling() -> None:
    market = {symbol: _candles(BREAKS) for symbol in COINS}  # eight breakouts at once
    intents = _orchestrator().evaluate_daily_cycle(market, [], BALANCE)
    opened = [intent for intent in intents if intent.action == OPEN_LONG]
    assert len(opened) == 7
    assert [intent.reason for intent in intents if intent.action == HOLD] == [MAX_HEAT_EXCEEDED]

    # What was approved, taken together, meets the portfolio rules.
    reference = TrendBreakoutStrategy().evaluate(market[COINS[0]], INSTRUMENTS[COINS[0]], account_balance=BALANCE)
    risk = sum(reference.risk_amount for _ in opened)
    nominal = sum(reference.nominal_value for _ in opened)
    assert risk <= 0.04 * BALANCE < risk + reference.risk_amount
    assert nominal <= 1.0 * BALANCE


def test_each_approval_counts_for_the_next_candidate() -> None:
    manager = PortfolioRiskManager()
    orchestrator = BasketOrchestrator(manager, TrendBreakoutStrategy(), INSTRUMENTS)
    market = {symbol: _candles(BREAKS) for symbol in COINS}
    intents = orchestrator.evaluate_daily_cycle(market, [], BALANCE)
    # Rebuild the portfolio in order of the orders: each one was acceptable at its turn...
    reference = TrendBreakoutStrategy().evaluate(market[COINS[0]], INSTRUMENTS[COINS[0]], account_balance=BALANCE)
    projected: list[ActivePosition] = []
    for symbol in _opened(intents):
        assert manager.can_open_position(BALANCE, projected, reference.nominal_value, reference.risk_amount, symbol, 1, "crypto")[0]
        projected.append(ActivePosition(symbol, 1, reference.nominal_value, reference.risk_amount, "crypto"))
    # ...and the one left out is not acceptable with the seven in.
    assert manager.can_open_position(BALANCE, projected, reference.nominal_value, reference.risk_amount, "HHHUSDT", 1, "crypto") == (
        False, MAX_HEAT_EXCEEDED,
    )


def test_existing_positions_count_against_the_ceiling() -> None:
    positions = [_held(symbol) for symbol in COINS[:5]]
    market = {symbol: _candles(QUIET) for symbol in COINS[:5]} | {symbol: _candles(BREAKS) for symbol in COINS[5:]}
    intents = _orchestrator().evaluate_daily_cycle(market, positions, BALANCE)
    assert _opened(intents) == ["FFFUSDT", "GGGUSDT"]
    assert _actions(intents)["HHHUSDT"] == (HOLD, MAX_HEAT_EXCEEDED)


def test_the_nominal_ceiling_applies_when_risk_is_not_the_limit() -> None:
    orchestrator = _orchestrator(max_portfolio_heat_pct=50.0, max_portfolio_leverage=0.35)
    intents = orchestrator.evaluate_daily_cycle({symbol: _candles(BREAKS) for symbol in COINS[:5]}, [], BALANCE)
    assert _opened(intents) == COINS[:3]
    assert [intent.reason for intent in intents if intent.action == HOLD] == [MAX_LEVERAGE_EXCEEDED] * 2


def test_correlated_forex_entries_in_the_same_cycle_are_limited_to_one() -> None:
    market = {"EURUSD": _candles(104.5), "GBPUSD": _candles(108.0), "AAAUSDT": _candles(BREAKS), "BBBUSDT": _candles(BREAKS)}
    intents = _orchestrator().evaluate_daily_cycle(market, [], BALANCE)
    # Both currencies break out at once: the stronger one enters and the other is blocked by correlation.
    assert _actions(intents)["GBPUSD"] == (OPEN_LONG, BREAKOUT)
    assert _actions(intents)["EURUSD"] == (HOLD, CORRELATED_EXPOSURE_BLOCKED)
    assert {"AAAUSDT", "BBBUSDT"} <= set(_opened(intents))  # the crypto basket does not have that limit


def test_only_the_injected_risk_manager_decides() -> None:
    # The strategy carries a portfolio that would reject everything; it is not asked.
    strict = TrendBreakoutStrategy(portfolio=PortfolioRiskManager(max_portfolio_heat_pct=0.01))
    orchestrator = BasketOrchestrator(PortfolioRiskManager(), strict, INSTRUMENTS)
    assert _opened(orchestrator.evaluate_daily_cycle({"AAAUSDT": _candles(BREAKS)}, [], BALANCE)) == ["AAAUSDT"]


def test_a_size_below_the_minimum_is_not_a_candidate() -> None:
    intents = _orchestrator().evaluate_daily_cycle({"AAAUSDT": _candles(BREAKS)}, [], 0.0001)
    assert _actions(intents) == {"AAAUSDT": (HOLD, SIZE_TOO_SMALL)}


# --- Resilience -------------------------------------------------------------


def _corrupt() -> pd.DataFrame:
    candles = _candles(BREAKS)
    candles.loc[120, "close"] = float("nan")
    return candles


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (_corrupt(), INVALID_DATA),
        (_candles(BREAKS, n=50), INVALID_DATA),  # not enough history
        (_candles(BREAKS).drop(columns=["high"]), INVALID_DATA),
        (_candles(BREAKS).assign(low=500.0), INVALID_DATA),
        (pd.DataFrame(), NO_DATA),
        (None, NO_DATA),
        ("not candles", NO_DATA),
        ([1.0, 2.0, 3.0], NO_DATA),
    ],
)
def test_one_bad_symbol_does_not_stop_the_cycle(bad: Any, reason: str) -> None:
    market = {"AAAUSDT": _candles(BREAKS), "BBBUSDT": bad, "CCCUSDT": _candles(FALLS), "DDDUSDT": _candles(105.0)}
    intents = _orchestrator().evaluate_daily_cycle(market, [_held("CCCUSDT")], BALANCE)
    actions = _actions(intents)
    assert actions["CCCUSDT"] == (CLOSE_LONG, EXIT_CHANNEL_BROKEN)
    assert _opened(intents) == ["DDDUSDT", "AAAUSDT"]
    assert actions["BBBUSDT"][0] == HOLD and actions["BBBUSDT"][1].startswith(reason)
    assert len(intents) == 4


def test_the_reason_says_what_is_wrong_with_the_data() -> None:
    market = {"AAAUSDT": _candles(BREAKS, n=50), "BBBUSDT": _corrupt()}
    actions = _actions(_orchestrator().evaluate_daily_cycle(market, [], BALANCE))
    assert actions["AAAUSDT"] == (HOLD, "INVALID_DATA: 101 candles required, got 50")
    assert actions["BBBUSDT"] == (HOLD, "INVALID_DATA: the candles have non-finite prices (row 120)")


def test_an_open_position_with_bad_data_is_kept_and_still_counts() -> None:
    # Seven positions fill the risk. Even if one cannot be evaluated, it still takes its place.
    positions = [_held(symbol) for symbol in COINS[:7]]
    market = {symbol: _candles(QUIET) for symbol in COINS[:7]} | {"HHHUSDT": _candles(BREAKS)}
    market["AAAUSDT"] = _corrupt()
    del market["BBBUSDT"]  # its candles do not even arrive
    actions = _actions(_orchestrator().evaluate_daily_cycle(market, positions, BALANCE))
    assert actions["AAAUSDT"][0] == HOLD and actions["AAAUSDT"][1].startswith(INVALID_DATA)
    assert actions["BBBUSDT"] == (HOLD, NO_DATA)
    assert actions["HHHUSDT"] == (HOLD, MAX_HEAT_EXCEEDED)


def test_a_symbol_without_trading_rules_is_skipped() -> None:
    market = {"ZZZUSDT": _candles(BREAKS), "AAAUSDT": _candles(BREAKS)}
    actions = _actions(_orchestrator().evaluate_daily_cycle(market, [], BALANCE))
    assert actions == {"AAAUSDT": (OPEN_LONG, BREAKOUT), "ZZZUSDT": (HOLD, NO_INSTRUMENT)}


def test_a_short_position_is_left_alone_but_counts_for_the_limits() -> None:
    orchestrator = _orchestrator(max_portfolio_heat_pct=50.0, max_portfolio_leverage=0.15)
    short = _held("AAAUSDT", direction=-1)
    market = {"AAAUSDT": _candles(FALLS), "BBBUSDT": _candles(BREAKS)}
    actions = _actions(orchestrator.evaluate_daily_cycle(market, [short], BALANCE))
    assert actions["AAAUSDT"] == (HOLD, NOT_MANAGED)
    assert actions["BBBUSDT"] == (HOLD, MAX_LEVERAGE_EXCEEDED)  # 10 % from the short + 10 % new > 15 %


def test_an_unexpected_failure_in_one_symbol_is_contained() -> None:
    class Flaky(TrendBreakoutStrategy):
        def evaluate(self, candles: pd.DataFrame, instrument: InstrumentSpec, **arguments: Any) -> Any:
            if instrument.symbol == "BBBUSDT":
                raise ZeroDivisionError("something nobody foresaw")
            return super().evaluate(candles, instrument, **arguments)

    orchestrator = BasketOrchestrator(PortfolioRiskManager(), Flaky(), INSTRUMENTS)
    lines, sink = _log("ERROR")
    try:
        intents = orchestrator.evaluate_daily_cycle({"AAAUSDT": _candles(BREAKS), "BBBUSDT": _candles(BREAKS)}, [], BALANCE)
    finally:
        logger.remove(sink)
    assert _actions(intents) == {
        "AAAUSDT": (OPEN_LONG, BREAKOUT),
        "BBBUSDT": (HOLD, "INVALID_DATA: ZeroDivisionError: something nobody foresaw"),
    }
    assert lines == ["ERROR|BBBUSDT: unexpected failure while evaluating, skipped: ZeroDivisionError: something nobody foresaw\n"]


# --- Invalid arguments --------------------------------------------------------


@pytest.mark.parametrize("market", [None, [], "AAAUSDT", pd.DataFrame()])
def test_market_data_must_be_a_mapping(market: Any) -> None:
    with pytest.raises(StrategyError, match="market_data must be a mapping"):
        _orchestrator().evaluate_daily_cycle(market, [], BALANCE)


@pytest.mark.parametrize("balance", [None, "100000", True, 0.0, -500.0, float("nan"), float("inf")])
def test_invalid_balance(balance: Any) -> None:
    with pytest.raises(RiskError, match="account_balance"):
        _orchestrator().evaluate_daily_cycle({"AAAUSDT": _candles(QUIET)}, [], balance)


@pytest.mark.parametrize("positions", [None, "AAAUSDT", [{"symbol": "AAAUSDT"}], [None]])
def test_invalid_positions(positions: Any) -> None:
    with pytest.raises(RiskError, match="active_positions"):
        _orchestrator().evaluate_daily_cycle({"AAAUSDT": _candles(QUIET)}, positions, BALANCE)


def test_invalid_dependencies() -> None:
    manager, strategy = PortfolioRiskManager(), TrendBreakoutStrategy()
    with pytest.raises(StrategyError, match="risk_manager"):
        BasketOrchestrator(None, strategy, INSTRUMENTS)  # type: ignore[arg-type]
    with pytest.raises(StrategyError, match="strategy"):
        BasketOrchestrator(manager, manager, INSTRUMENTS)  # type: ignore[arg-type]
    with pytest.raises(StrategyError, match="instruments must be a mapping"):
        BasketOrchestrator(manager, strategy, list(INSTRUMENTS.values()))  # type: ignore[arg-type]
    with pytest.raises(StrategyError, match=r"instruments\['BTCUSDT'\]"):
        BasketOrchestrator(manager, strategy, {"BTCUSDT": INSTRUMENTS["AAAUSDT"]})


# --- Logging ----------------------------------------------------------------


def test_the_cycle_summary_is_logged() -> None:
    positions = [_held(symbol) for symbol in COINS[:6]]
    market = {symbol: _candles(QUIET) for symbol in COINS[:6]} | {"GGGUSDT": _candles(108.0), "HHHUSDT": _candles(BREAKS)}
    market["AAAUSDT"] = _candles(FALLS)
    market["BBBUSDT"] = _corrupt()
    market["EURUSD"] = _candles(BREAKS, n=30)
    lines, sink = _log()
    try:
        _orchestrator().evaluate_daily_cycle(market, positions, BALANCE)
    finally:
        logger.remove(sink)

    assert lines[-1] == (
        "INFO|Basket cycle: 1 closes | 2 entry candidates, 2 approved | "
        "7 positions after the cycle | 2 symbols skipped for their data\n"
    )
    assert any(line.startswith("INFO|AAAUSDT: exit") for line in lines)
    assert sum(line.startswith("WARNING|") and "unusable data" in line for line in lines) == 2


def test_the_order_in_which_risk_is_handed_out_is_logged() -> None:
    # The strategy's buy lines come out in basket order; this one gives the allocation order.
    market = {"AAAUSDT": _candles(103.5), "BBBUSDT": _candles(QUIET), "CCCUSDT": _candles(108.0),
              "DDDUSDT": _candles(105.0)}
    lines, sink = _log()
    try:
        _orchestrator().evaluate_daily_cycle(market, [], BALANCE)
    finally:
        logger.remove(sink)
    (ranking,) = [line for line in lines if "order risk is allocated" in line]
    assert [part.split()[0] for part in ranking.split(": ", 1)[1].split(", ")] == ["CCCUSDT", "DDDUSDT", "AAAUSDT"]
    assert ranking.index("CCCUSDT +") < ranking.index("DDDUSDT +") < ranking.index("AAAUSDT +")


def test_a_single_breakout_needs_no_ranking_line() -> None:
    lines, sink = _log()
    try:
        _orchestrator().evaluate_daily_cycle({"AAAUSDT": _candles(BREAKS), "BBBUSDT": _candles(QUIET)}, [], BALANCE)
    finally:
        logger.remove(sink)
    assert not any("order risk is allocated" in line for line in lines)


def test_rejected_candidates_are_logged_and_counted() -> None:
    market = {symbol: _candles(BREAKS) for symbol in COINS}
    lines, sink = _log()
    try:
        _orchestrator().evaluate_daily_cycle(market, [], BALANCE)
    finally:
        logger.remove(sink)
    assert "INFO|HHHUSDT: candidate dropped (MAX_HEAT_EXCEEDED), breakout strength 0.35 ATR\n" in lines
    assert lines[-1] == (
        "INFO|Basket cycle: 0 closes | 8 entry candidates, 7 approved, 1 for MAX_HEAT_EXCEEDED | "
        "7 positions after the cycle | 0 symbols skipped for their data\n"
    )
