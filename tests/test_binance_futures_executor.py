"""Tests of the Binance USDT-M Futures executor and of its paper account."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from loguru import logger

from src.engine.orchestrator import CLOSE_LONG, HOLD, MARKET, OPEN_LONG, OrderIntent
from src.execution.errors import BrokerError, BrokerUnavailableError, OrderRejectedError
from src.execution.binance_futures_executor import (
    BELOW_MIN_NOTIONAL,
    BELOW_MIN_QTY,
    BELOW_STEP,
    DRY_RUN,
    FILLED,
    INVALID_STOP,
    LIVE,
    NO_POSITION,
    NO_RULES,
    PARTIAL_CLOSE,
    POSITION_EXISTS,
    REJECTED,
    SIMULATED,
    SKIPPED,
    STOP_FAILED_CLOSED,
    TESTNET,
    UNPROTECTED,
    BinanceFuturesExecutor,
    MarketRules,
    PaperFuturesAccount,
)
from src.risk.portfolio_risk import ActivePosition
from tests.fake_futures import DAY_MS, NOW_MS, FakeFuturesClient, api_error, network_error, symbol_info

CYCLE = "2026-10-09"
SYMBOLS = {
    "BTCUSDT": symbol_info("BTCUSDT", step="0.001", min_qty="0.001", tick="0.10", min_notional="50"),
    "DOGEUSDT": symbol_info("DOGEUSDT", step="1", min_qty="1", tick="0.00001", min_notional="5", market_max="5000"),
}


@pytest.fixture
def client() -> FakeFuturesClient:
    fake = FakeFuturesClient(dict(SYMBOLS))
    fake.marks = {"BTCUSDT": 80_000.0, "DOGEUSDT": 0.25}
    return fake


@pytest.fixture
def log() -> Any:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}")
    yield lines
    logger.remove(sink)


def _executor(client: FakeFuturesClient, mode: str = TESTNET, **options: Any) -> BinanceFuturesExecutor:
    options.setdefault("live_trading_acknowledged", mode == LIVE)
    return BinanceFuturesExecutor(
        mode=mode, client=client, sleep=lambda _: None, clock=lambda: client.now_ms / 1000, **options
    )


def _paper(tmp_path: Path, capital: float = 10_000.0, fee_rate: float = 0.0005) -> PaperFuturesAccount:
    return PaperFuturesAccount(tmp_path / "paper.json", capital, fee_rate)


def _open(symbol: str = "BTCUSDT", volume: float = 0.0129, stop: float | None = 76_543.21) -> OrderIntent:
    return OrderIntent(symbol, OPEN_LONG, volume, MARKET, stop, "BREAKOUT")


def _close(symbol: str = "BTCUSDT") -> OrderIntent:
    return OrderIntent(symbol, CLOSE_LONG, 0.0, MARKET, None, "EXIT_CHANNEL_BROKEN")


def _sells(client: FakeFuturesClient) -> list[dict[str, Any]]:
    return [order for order in client.sent("futures_create_order") if order["side"] == "SELL"]


# --- Execution mode ----------------------------------------------------------


def test_unknown_mode_is_rejected(client: FakeFuturesClient) -> None:
    with pytest.raises(BrokerUnavailableError, match="unknown execution mode"):
        _executor(client, mode="REAL")


def test_live_without_acknowledgement_falls_back_to_dry_run(
    client: FakeFuturesClient, tmp_path: Path, log: list[str]
) -> None:
    executor = _executor(client, mode=LIVE, live_trading_acknowledged=False, paper=_paper(tmp_path))
    assert executor.mode == DRY_RUN and executor.is_simulated
    assert any(line.startswith("WARNING|") and "live_trading_acknowledged" in line for line in log)
    result = executor.execute_order_intent(_open(), CYCLE)
    assert result["status"] == SIMULATED
    assert client.sent("futures_create_order") == []


def test_live_with_acknowledgement_is_live(client: FakeFuturesClient) -> None:
    executor = _executor(client, mode=LIVE, live_trading_acknowledged=True)
    assert executor.mode == LIVE and not executor.is_simulated


@pytest.mark.parametrize("acknowledged", [False, 0, None, "", "no", "true", 1])
def test_only_true_acknowledges_live(client: FakeFuturesClient, tmp_path: Path, acknowledged: Any) -> None:
    executor = _executor(client, mode=LIVE, live_trading_acknowledged=acknowledged, paper=_paper(tmp_path))
    assert executor.mode == DRY_RUN


@pytest.mark.parametrize("mode", [TESTNET, LIVE])
def test_real_modes_require_credentials(mode: str) -> None:
    with pytest.raises(BrokerUnavailableError, match="BOT_FUTURES__API_KEY"):
        BinanceFuturesExecutor(mode=mode, live_trading_acknowledged=True)


def test_dry_run_requires_a_paper_account(client: FakeFuturesClient) -> None:
    with pytest.raises(BrokerUnavailableError, match="paper account"):
        _executor(client, mode=DRY_RUN)


@pytest.mark.parametrize("leverage", [0, -1, 1.5, True, "1"])
def test_invalid_leverage_is_rejected(client: FakeFuturesClient, leverage: Any) -> None:
    with pytest.raises(BrokerUnavailableError, match="leverage"):
        _executor(client, leverage=leverage)


@pytest.mark.parametrize("endpoint", ["create_order", "create_margin_order", "futures_coin_create_order", "transfer"])
def test_only_whitelisted_futures_endpoints_can_be_called(client: FakeFuturesClient, endpoint: str) -> None:
    with pytest.raises(RuntimeError, match="endpoint not allowed"):
        _executor(client)._once(endpoint)
    assert client.calls == []


# --- Trading rules -----------------------------------------------------------


def test_market_rules_come_from_exchange_info(client: FakeFuturesClient) -> None:
    rules = _executor(client).load_market_rules(["BTCUSDT", "DOGEUSDT"])
    assert rules["BTCUSDT"] == MarketRules(
        "BTCUSDT", Decimal("0.001"), Decimal("0.001"), Decimal("120"), Decimal("0.10"), Decimal("50")
    )
    assert rules["DOGEUSDT"].step_size == Decimal("1")
    assert rules["DOGEUSDT"].tick_size == Decimal("0.00001")
    assert rules["DOGEUSDT"].min_notional == Decimal("5")
    assert rules["DOGEUSDT"].max_qty == Decimal("5000")


def test_market_rules_are_downloaded_once(client: FakeFuturesClient) -> None:
    executor = _executor(client)
    executor.load_market_rules(["BTCUSDT"])
    executor.load_market_rules(["BTCUSDT"])
    assert len(client.sent("futures_exchange_info")) == 1
    executor.load_market_rules(["BTCUSDT", "DOGEUSDT"])
    assert len(client.sent("futures_exchange_info")) == 2


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        (None, "does not list it"),
        (symbol_info("XUSDT", status="SETTLING"), "SETTLING"),
        (symbol_info("XUSDT", contract="CURRENT_QUARTER"), "perpetual"),
        (symbol_info("XUSDT", quote="USDC"), "USDT"),
    ],
)
def test_untradable_symbols_get_no_rules(
    client: FakeFuturesClient, log: list[str], entry: dict[str, Any] | None, why: str
) -> None:
    if entry is not None:
        client.symbols["XUSDT"] = entry
    rules = _executor(client).load_market_rules(["XUSDT", "BTCUSDT"])
    assert list(rules) == ["BTCUSDT"]
    assert any(line.startswith("WARNING|XUSDT") and why in line for line in log)


def test_unreadable_filters_omit_only_that_symbol(client: FakeFuturesClient, log: list[str]) -> None:
    broken = symbol_info("XUSDT")
    broken["filters"] = [f for f in broken["filters"] if f["filterType"] != "PRICE_FILTER"]
    zero = symbol_info("YUSDT", step="0")
    client.symbols.update(XUSDT=broken, YUSDT=zero)
    assert list(_executor(client).load_market_rules(["XUSDT", "YUSDT", "DOGEUSDT"])) == ["DOGEUSDT"]
    assert sum("unreadable" in line for line in log) == 2


def test_the_stricter_of_the_two_lot_filters_wins(client: FakeFuturesClient) -> None:
    entry = symbol_info("XUSDT", step="0.01", min_qty="0.01")
    entry["filters"][2].update(stepSize="0.1", minQty="0.5", maxQty="40")
    client.symbols["XUSDT"] = entry
    rules = _executor(client).load_market_rules(["XUSDT"])["XUSDT"]
    assert (rules.step_size, rules.min_qty, rules.max_qty) == (Decimal("0.1"), Decimal("0.5"), Decimal("40"))


@pytest.mark.parametrize(
    ("step", "quantity", "expected"),
    [
        ("0.001", 0.0129, "0.012"),
        ("0.001", 0.012999999, "0.012"),
        ("0.001", 0.013, "0.013"),
        ("0.001", 0.0009, "0.000"),
        ("1", 3999.99, "3999"),
        ("1", 4000.0, "4000"),
        ("0.01", 0.07, "0.07"),  # with floats, 0.07 / 0.01 gives 6.999… and it would stay at 0.06
        ("0.1", 0.3, "0.3"),
        ("10", 125.0, "120"),
    ],
)
def test_quantities_round_down_to_the_step(step: str, quantity: float, expected: str) -> None:
    rules = MarketRules("X", Decimal(step), Decimal(step), None, Decimal("0.01"), Decimal("5"))
    rounded = rules.round_quantity(quantity)
    assert rounded == Decimal(expected)
    assert rounded % Decimal(step) == 0
    assert rounded <= Decimal(str(quantity))


@pytest.mark.parametrize(
    ("tick", "price", "expected"),
    [
        ("0.10", 76_543.21, "76543.2"),
        ("0.10", 76_543.29999, "76543.2"),
        ("0.10", 76_543.3, "76543.3"),
        ("0.00001", 0.2345678, "0.23456"),
        ("0.01", 4.35, "4.35"),  # 4.35 has no exact representation in binary
        ("5", 1234.0, "1230"),
    ],
)
def test_prices_round_down_to_the_tick(tick: str, price: float, expected: str) -> None:
    rules = MarketRules("X", Decimal("1"), Decimal("1"), None, Decimal(tick), Decimal("5"))
    rounded = rules.round_price(price)
    assert rounded == Decimal(expected)
    assert rounded % Decimal(tick) == 0
    assert rounded <= Decimal(str(price))


def test_rules_give_the_strategy_its_instrument(client: FakeFuturesClient) -> None:
    spec = _executor(client).load_market_rules(["DOGEUSDT"])["DOGEUSDT"].to_spec()
    assert (spec.symbol, spec.asset_class, spec.step_size, spec.min_volume, spec.max_volume) == (
        "DOGEUSDT", "crypto", 1.0, 1.0, 5000.0,
    )
    assert spec.contract_size == 1.0


# --- Balance and positions ---------------------------------------------------


def test_account_reads_the_usdt_balance(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [0.01, 79_000.0]
    executor = _executor(client)
    account = executor.get_account()
    assert account.wallet_balance == pytest.approx(10_000.0)
    assert account.unrealized_pnl == pytest.approx(10.0)
    assert account.available_balance == pytest.approx(10_000.0 + 10.0 - 800.0)
    assert executor.get_account_balance() == pytest.approx(9_210.0)


def test_account_without_usdt_is_an_error(client: FakeFuturesClient) -> None:
    client.futures_account = lambda **_: {"assets": [{"asset": "BNB", "walletBalance": "1"}]}  # type: ignore[method-assign]
    with pytest.raises(BrokerError, match="USDT balance"):
        _executor(client).get_account()


def test_hedge_mode_account_is_refused(client: FakeFuturesClient) -> None:
    client.dual_side = True
    with pytest.raises(BrokerUnavailableError, match="hedge mode"):
        _executor(client).get_account()


def test_open_positions_map_to_active_positions(client: FakeFuturesClient) -> None:
    client.positions.update(BTCUSDT=[0.012, 78_000.0], DOGEUSDT=[4000.0, 0.24])
    client.algo_orders += [
        {"algoId": 1, "clientAlgoId": "a", "orderType": "STOP_MARKET", "symbol": "BTCUSDT", "side": "SELL",
         "triggerPrice": "74000.0"},
        {"algoId": 2, "clientAlgoId": "b", "orderType": "STOP_MARKET", "symbol": "DOGEUSDT", "side": "SELL",
         "triggerPrice": "0.22000"},
    ]
    positions = {p.symbol: p for p in _executor(client).get_open_positions()}
    assert isinstance(positions["BTCUSDT"], ActivePosition)
    assert (positions["BTCUSDT"].direction, positions["BTCUSDT"].asset_class) == (1, "crypto")
    assert positions["BTCUSDT"].nominal_value == pytest.approx(0.012 * 80_000.0)
    assert positions["BTCUSDT"].risk_usd == pytest.approx(0.012 * 4_000.0)
    assert positions["DOGEUSDT"].nominal_value == pytest.approx(4000.0 * 0.25)
    assert positions["DOGEUSDT"].risk_usd == pytest.approx(4000.0 * 0.02)


def test_flat_rows_are_not_positions(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [0.0, 0.0]
    executor = _executor(client)
    assert executor.get_open_positions() == []
    assert client.sent("futures_get_open_algo_orders") == []


def test_a_stop_at_or_above_the_entry_carries_no_risk(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [0.012, 78_000.0]
    client.algo_orders.append({"algoId": 1, "orderType": "STOP_MARKET", "symbol": "BTCUSDT", "side": "SELL",
                               "triggerPrice": "79000.0"})
    assert _executor(client).get_open_positions()[0].risk_usd == 0.0


def test_the_highest_sell_stop_is_the_one_that_counts(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [0.01, 78_000.0]
    client.algo_orders += [
        {"algoId": 1, "orderType": "STOP_MARKET", "symbol": "BTCUSDT", "side": "SELL", "triggerPrice": "70000"},
        {"algoId": 2, "orderType": "STOP_MARKET", "symbol": "BTCUSDT", "side": "SELL", "triggerPrice": "75000"},
        {"algoId": 3, "orderType": "STOP_MARKET", "symbol": "BTCUSDT", "side": "BUY", "triggerPrice": "77900"},
        {"algoId": 4, "orderType": "TAKE_PROFIT_MARKET", "symbol": "BTCUSDT", "side": "SELL", "triggerPrice": "77950"},
    ]
    assert _executor(client).get_open_positions()[0].risk_usd == pytest.approx(0.01 * 3_000.0)


def test_position_without_stop_uses_the_protective_distance(client: FakeFuturesClient, log: list[str]) -> None:
    client.positions["BTCUSDT"] = [0.01, 78_000.0]
    position = _executor(client).get_open_positions({"BTCUSDT": 5_000.0})[0]
    assert position.risk_usd == pytest.approx(50.0)
    assert any(line.startswith("WARNING|BTCUSDT") and "without a stop on the exchange" in line for line in log)


def test_position_without_stop_or_distance_risks_its_whole_nominal(client: FakeFuturesClient, log: list[str]) -> None:
    client.positions["BTCUSDT"] = [0.01, 78_000.0]
    position = _executor(client).get_open_positions({"DOGEUSDT": 0.02})[0]
    assert position.risk_usd == pytest.approx(position.nominal_value) == pytest.approx(800.0)
    assert any("whole notional" in line for line in log)


def test_a_short_position_counts_but_is_marked_as_short(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [-0.01, 78_000.0]
    position = _executor(client).get_open_positions({"BTCUSDT": 5_000.0})[0]
    assert (position.direction, position.nominal_value) == (-1, pytest.approx(800.0))


# --- Network failures in queries ---------------------------------------------


def test_reads_are_retried_after_transient_failures(client: FakeFuturesClient, log: list[str]) -> None:
    client.fail["futures_account"] = [network_error(), api_error(-1001, "Internal error", 503)]
    assert _executor(client).get_account().wallet_balance == pytest.approx(10_000.0)
    assert len(client.sent("futures_account")) == 3
    assert sum("retrying in" in line for line in log) == 2


def test_reads_give_up_after_the_configured_retries(client: FakeFuturesClient) -> None:
    client.fail["futures_account"] = [network_error() for _ in range(10)]
    with pytest.raises(BrokerError, match="could not reach"):
        _executor(client, retries=2).get_account()
    assert len(client.sent("futures_account")) == 3


def test_a_definitive_rejection_is_not_retried(client: FakeFuturesClient) -> None:
    client.fail["futures_account"] = [api_error(-2015, "Invalid API-key, IP, or permissions for action.", 401)]
    with pytest.raises(BrokerError, match=r"\[-2015\]"):
        _executor(client).get_account()
    assert len(client.sent("futures_account")) == 1


def test_retries_wait_longer_each_time(client: FakeFuturesClient) -> None:
    waits: list[float] = []
    client.fail["futures_time"] = [network_error(), network_error(), network_error()]
    executor = BinanceFuturesExecutor(mode=TESTNET, client=client, sleep=waits.append, retry_wait_seconds=1.5)
    assert executor.server_time_ms() == NOW_MS
    assert waits == [1.5, 3.0, 6.0]


# --- Buys --------------------------------------------------------------------


def test_open_buys_then_places_the_stop_immediately(client: FakeFuturesClient) -> None:
    result = _executor(client).execute_order_intent(_open(volume=0.0129, stop=76_543.21), CYCLE)

    assert result["status"] == FILLED
    assert result["quantity"] == pytest.approx(0.012) and result["price"] == pytest.approx(80_000.0)
    assert result["stop_price"] == pytest.approx(76_543.2)
    (buy,) = client.sent("futures_create_order")
    assert buy == {
        "symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.012",
        "newOrderRespType": "RESULT", "newClientOrderId": "tb-20261009-BTCUSDT-o",
    }
    (stop,) = client.sent("futures_create_algo_order")
    assert stop == {
        "algoType": "CONDITIONAL", "symbol": "BTCUSDT", "side": "SELL", "type": "STOP_MARKET",
        "quantity": "0.012", "triggerPrice": "76543.2", "reduceOnly": "true",
        "workingType": "CONTRACT_PRICE", "clientAlgoId": "tb-20261009-BTCUSDT-s",
    }
    assert result["payloads"] == [buy, stop]
    # The stop is the call that follows the buy: there is nothing between the two.
    names = [name for name, _ in client.calls]
    assert names.index("futures_create_algo_order") == names.index("futures_create_order") + 1
    assert client.positions["BTCUSDT"][0] == pytest.approx(0.012)


def test_open_sets_the_leverage_before_buying(client: FakeFuturesClient) -> None:
    executor = _executor(client, leverage=1)
    executor.execute_order_intent(_open(), CYCLE)
    names = [name for name, _ in client.calls]
    assert client.leverage == {"BTCUSDT": 1}
    assert names.index("futures_change_leverage") < names.index("futures_create_order")
    client.positions.clear()
    executor.execute_order_intent(_open(), "2026-10-10")
    assert len(client.sent("futures_change_leverage")) == 1


def test_open_is_not_sent_if_the_leverage_cannot_be_set(client: FakeFuturesClient) -> None:
    client.fail["futures_change_leverage"] = [api_error(-4028, "Leverage 1 is not valid")]
    with pytest.raises(OrderRejectedError, match="leverage"):
        _executor(client).execute_order_intent(_open(), CYCLE)
    assert client.sent("futures_create_order") == []


def test_the_stop_protects_what_was_actually_filled(client: FakeFuturesClient) -> None:
    client.marks["DOGEUSDT"] = 0.25
    result = _executor(client).execute_order_intent(_open("DOGEUSDT", 3999.99, 0.2345678), CYCLE)
    assert client.sent("futures_create_order")[0]["quantity"] == "3999"
    stop = client.sent("futures_create_algo_order")[0]
    assert (stop["quantity"], stop["triggerPrice"]) == ("3999", "0.23456")
    assert result["quantity"] == pytest.approx(3999.0)


def test_stop_working_type_is_configurable(client: FakeFuturesClient) -> None:
    _executor(client, stop_working_type="MARK_PRICE").execute_order_intent(_open(), CYCLE)
    assert client.sent("futures_create_algo_order")[0]["workingType"] == "MARK_PRICE"


def test_volume_above_the_market_maximum_is_capped(client: FakeFuturesClient) -> None:
    _executor(client).execute_order_intent(_open("DOGEUSDT", 12_345.6, 0.2), CYCLE)
    assert client.sent("futures_create_order")[0]["quantity"] == "5000"


@pytest.mark.parametrize(
    ("symbol", "volume", "reason", "explains"),
    [
        ("BTCUSDT", 0.0012, BELOW_MIN_NOTIONAL, "minimum per order is 50"),
        ("DOGEUSDT", 19.9, BELOW_MIN_NOTIONAL, "is worth 4.75 USDT"),
        ("DOGEUSDT", 0.4, BELOW_STEP, "is below the step"),
    ],
)
def test_orders_below_the_exchange_minimums_are_not_sent(
    client: FakeFuturesClient, log: list[str], symbol: str, volume: float, reason: str, explains: str
) -> None:
    # 0.001 BTC at 40 000 is 40 USDT, below the 50 minimum of that contract.
    client.marks["BTCUSDT"] = 40_000.0
    result = _executor(client).execute_order_intent(_open(symbol, volume, 0.2 if symbol == "DOGEUSDT" else 30_000.0), CYCLE)
    assert (result["status"], result["reason"]) == (REJECTED, reason)
    assert client.sent("futures_create_order") == [] and client.sent("futures_create_algo_order") == []
    assert client.positions == {}
    warning = next(line for line in log if line.startswith(f"WARNING|{symbol}: order not sent"))
    assert reason in warning and explains in warning


def test_an_order_exactly_at_the_minimum_notional_is_sent(client: FakeFuturesClient) -> None:
    result = _executor(client).execute_order_intent(_open("DOGEUSDT", 20.0, 0.2), CYCLE)
    assert result["status"] == FILLED


def test_quantity_below_the_minimum_lot_is_not_sent(client: FakeFuturesClient) -> None:
    client.symbols["XUSDT"] = symbol_info("XUSDT", step="0.1", min_qty="1")
    client.marks["XUSDT"] = 100.0
    result = _executor(client).execute_order_intent(_open("XUSDT", 0.95, 90.0), CYCLE)
    assert (result["status"], result["reason"]) == (REJECTED, BELOW_MIN_QTY)
    assert client.sent("futures_create_order") == []


@pytest.mark.parametrize("stop", [None, 0.0, -5.0, 80_000.0, 80_000.05, 95_000.0])
def test_open_needs_a_stop_below_the_price(client: FakeFuturesClient, stop: float | None) -> None:
    result = _executor(client).execute_order_intent(_open(stop=stop), CYCLE)
    assert (result["status"], result["reason"]) == (REJECTED, INVALID_STOP)
    assert client.sent("futures_create_order") == []


def test_open_is_refused_when_the_symbol_already_has_a_position(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [0.005, 79_000.0]
    result = _executor(client).execute_order_intent(_open(), CYCLE)
    assert (result["status"], result["reason"]) == (REJECTED, POSITION_EXISTS)
    assert client.sent("futures_create_order") == []


def test_open_without_market_rules_is_refused(client: FakeFuturesClient) -> None:
    result = _executor(client).execute_order_intent(_open("NOPEUSDT", 10.0, 1.0), CYCLE)
    assert (result["status"], result["reason"]) == (REJECTED, NO_RULES)
    assert client.sent("futures_create_order") == []


def test_a_rejected_buy_places_no_stop(client: FakeFuturesClient) -> None:
    client.fail["futures_create_order"] = [api_error(-2019, "Margin is insufficient.")]
    with pytest.raises(OrderRejectedError, match="Margin is insufficient"):
        _executor(client).execute_order_intent(_open(), CYCLE)
    assert len(client.sent("futures_create_order")) == 1
    assert client.sent("futures_create_algo_order") == []


# --- Orders with no answer ---------------------------------------------------


@pytest.mark.parametrize("lost", [network_error, lambda: api_error(-1007, "Timeout waiting for response", 408)])
def test_a_buy_whose_answer_is_lost_is_not_sent_twice(client: FakeFuturesClient, lost: Any) -> None:
    client.fail_after["futures_create_order"] = [lost()]
    result = _executor(client).execute_order_intent(_open(), CYCLE)
    assert result["status"] == FILLED and result["quantity"] == pytest.approx(0.012)
    assert len(client.sent("futures_create_order")) == 1
    assert client.sent("futures_get_order")[0]["origClientOrderId"] == "tb-20261009-BTCUSDT-o"
    assert client.positions["BTCUSDT"][0] == pytest.approx(0.012)
    assert len(client.algo_orders) == 1


def test_a_buy_that_never_arrived_is_sent_again(client: FakeFuturesClient) -> None:
    client.fail["futures_create_order"] = [network_error(), network_error()]
    result = _executor(client).execute_order_intent(_open(), CYCLE)
    assert result["status"] == FILLED
    assert len(client.sent("futures_create_order")) == 3
    assert {order["newClientOrderId"] for order in client.sent("futures_create_order")} == {"tb-20261009-BTCUSDT-o"}
    assert client.positions["BTCUSDT"][0] == pytest.approx(0.012)


def test_a_filled_buy_the_exchange_cannot_find_is_recognised_by_its_position(client: FakeFuturesClient) -> None:
    client.fail_after["futures_create_order"] = [network_error()]
    client.futures_get_order = lambda **_: (_ for _ in ()).throw(api_error(-2013, "Order does not exist."))  # type: ignore[method-assign]
    result = _executor(client).execute_order_intent(_open(), CYCLE)
    assert result["status"] == FILLED and result["price"] == pytest.approx(80_000.0)
    assert len(client.sent("futures_create_order")) == 1
    assert len(client.algo_orders) == 1


def test_a_buy_that_keeps_failing_ends_in_an_error(client: FakeFuturesClient) -> None:
    client.fail["futures_create_order"] = [network_error() for _ in range(10)]
    with pytest.raises(BrokerError, match="did not arrive in 3 attempts"):
        _executor(client, retries=2).execute_order_intent(_open(), CYCLE)
    assert client.positions == {} and client.algo_orders == []


def test_a_stop_whose_answer_is_lost_is_not_placed_twice(client: FakeFuturesClient) -> None:
    client.fail_after["futures_create_algo_order"] = [network_error()]
    result = _executor(client).execute_order_intent(_open(), CYCLE)
    assert result["status"] == FILLED and result["stop_order_id"] == client.algo_orders[0]["algoId"]
    assert len(client.sent("futures_create_algo_order")) == 1 and len(client.algo_orders) == 1


def test_a_stop_that_never_arrived_is_sent_again(client: FakeFuturesClient) -> None:
    client.fail["futures_create_algo_order"] = [network_error()]
    result = _executor(client).execute_order_intent(_open(), CYCLE)
    assert result["status"] == FILLED
    assert len(client.sent("futures_create_algo_order")) == 2 and len(client.algo_orders) == 1


def test_a_partial_fill_is_protected_for_what_was_filled(client: FakeFuturesClient, log: list[str]) -> None:
    create = client.futures_create_order

    def partial(**parameters: Any) -> dict[str, Any]:
        order = create(**{**parameters, "quantity": "0.007"} if parameters["side"] == "BUY" else parameters)
        return {**order, "status": "EXPIRED"}

    client.futures_create_order = partial  # type: ignore[method-assign]
    result = _executor(client).execute_order_intent(_open(), CYCLE)
    assert result["quantity"] == pytest.approx(0.007)
    assert client.sent("futures_create_algo_order")[0]["quantity"] == "0.007"
    assert any("ended as EXPIRED" in line for line in log)


# --- Position left without a stop --------------------------------------------


def test_if_the_stop_is_rejected_the_new_position_is_closed(client: FakeFuturesClient, log: list[str]) -> None:
    client.fail["futures_create_algo_order"] = [api_error(-2021, "Order would immediately trigger.")]
    result = _executor(client).execute_order_intent(_open(), CYCLE)

    assert result["status"] == STOP_FAILED_CLOSED and "immediately trigger" in result["reason"]
    assert client.positions == {} and client.algo_orders == []
    (undo,) = _sells(client)
    assert undo["reduceOnly"] == "true" and undo["quantity"] == "0.012"
    assert undo["newClientOrderId"] == "tb-20261009-BTCUSDT-x"
    assert any(line.startswith("ERROR|BTCUSDT: could not place the stop") for line in log)


def test_if_the_stop_keeps_failing_the_new_position_is_closed(client: FakeFuturesClient) -> None:
    client.fail["futures_create_algo_order"] = [network_error() for _ in range(10)]
    result = _executor(client, retries=1).execute_order_intent(_open(), CYCLE)
    assert result["status"] == STOP_FAILED_CLOSED
    assert client.positions == {}


def test_a_position_left_without_stop_is_reported_as_critical(client: FakeFuturesClient, log: list[str]) -> None:
    client.fail["futures_create_algo_order"] = [api_error(-2021, "Order would immediately trigger.")]
    buy = client.futures_create_order

    def only_buys(**parameters: Any) -> dict[str, Any]:
        if parameters["side"] == "SELL":
            client.calls.append(("futures_create_order", dict(parameters)))
            raise api_error(-4131, "The counterparty's best price does not meet the PERCENT_PRICE filter limit")
        return buy(**parameters)

    client.futures_create_order = only_buys  # type: ignore[method-assign]
    result = _executor(client).execute_order_intent(_open(), CYCLE)

    assert result["status"] == UNPROTECTED
    assert client.positions["BTCUSDT"][0] == pytest.approx(0.012)
    assert any(line.startswith("CRITICAL|BTCUSDT: POSITION WITHOUT STOP") for line in log)


def test_an_unexpected_failure_placing_the_stop_still_closes_the_new_position(client: FakeFuturesClient) -> None:
    client.fail["futures_create_algo_order"] = [ValueError("unreadable answer")]
    result = _executor(client).execute_order_intent(_open(), CYCLE)
    assert result["status"] == STOP_FAILED_CLOSED and "unreadable answer" in result["reason"]
    assert client.positions == {}


@pytest.mark.parametrize("code", [-1021, -1003, -1008])
def test_a_stop_refused_for_a_passing_reason_is_sent_again(client: FakeFuturesClient, code: int) -> None:
    client.fail["futures_create_algo_order"] = [api_error(code, "try again")]
    result = _executor(client).execute_order_intent(_open(), CYCLE)
    assert result["status"] == FILLED  # not flattened for a clock skew or a rate limit
    assert len(client.sent("futures_create_algo_order")) == 2 and len(client.algo_orders) == 1


# --- Closes ------------------------------------------------------------------


def _held(client: FakeFuturesClient, symbol: str = "BTCUSDT", quantity: float = 0.012) -> None:
    client.positions[symbol] = [quantity, 78_000.0 if symbol == "BTCUSDT" else 0.24]
    client.algo_orders.append({
        "algoId": 77, "clientAlgoId": f"tb-20261001-{symbol}-s", "orderType": "STOP_MARKET", "symbol": symbol,
        "side": "SELL", "triggerPrice": "74000.0", "quantity": str(quantity), "reduceOnly": True,
    })


def test_close_sells_the_whole_position_reduce_only(client: FakeFuturesClient) -> None:
    _held(client)
    result = _executor(client).execute_order_intent(_close(), CYCLE)

    assert result["status"] == FILLED and result["quantity"] == pytest.approx(0.012)
    (sell,) = client.sent("futures_create_order")
    assert sell == {
        "symbol": "BTCUSDT", "side": "SELL", "type": "MARKET", "quantity": "0.012", "reduceOnly": "true",
        "newOrderRespType": "RESULT", "newClientOrderId": "tb-20261009-BTCUSDT-c",
    }
    assert client.positions == {}


def test_close_cancels_the_orphan_stop_after_selling(client: FakeFuturesClient) -> None:
    _held(client)
    result = _executor(client).execute_order_intent(_close(), CYCLE)
    assert client.algo_orders == [] and result["cancelled_stops"] == 1
    assert client.sent("futures_cancel_algo_order") == [{"symbol": "BTCUSDT", "algoId": 77}]
    names = [name for name, _ in client.calls]
    assert names.index("futures_create_order") < names.index("futures_cancel_algo_order")


def test_close_only_cancels_sell_stops_of_that_symbol(client: FakeFuturesClient) -> None:
    _held(client)
    _held(client, "DOGEUSDT", 4000.0)
    client.algo_orders[-1]["algoId"] = 78
    client.algo_orders.append({"algoId": 79, "orderType": "TAKE_PROFIT_MARKET", "symbol": "BTCUSDT", "side": "SELL",
                               "triggerPrice": "99000"})
    _executor(client).execute_order_intent(_close(), CYCLE)
    assert sorted(order["algoId"] for order in client.algo_orders) == [78, 79]


@pytest.mark.parametrize(
    ("symbol", "quantity", "sent"),
    [("BTCUSDT", 0.012, "0.012"), ("BTCUSDT", 1.2, "1.2"), ("DOGEUSDT", 4000.0, "4000"), ("DOGEUSDT", 12264.0, "12264")],
)
def test_every_close_carries_reduce_only(client: FakeFuturesClient, symbol: str, quantity: float, sent: str) -> None:
    _held(client, symbol, quantity)
    _executor(client).execute_order_intent(_close(symbol), CYCLE)
    sells = _sells(client)
    assert sells and all(order["reduceOnly"] == "true" for order in sells)
    # Just as the exchange gives it, without padding decimals that exceed the contract's precision.
    assert sells[0]["quantity"] == sent


def test_dry_run_close_quantity_has_no_padding(client: FakeFuturesClient, tmp_path: Path) -> None:
    executor = _executor(client, mode=DRY_RUN, paper=_paper(tmp_path))
    executor.execute_order_intent(_open("DOGEUSDT", 4000.7, 0.2), CYCLE)
    assert executor.execute_order_intent(_close("DOGEUSDT"), CYCLE)["payloads"][0]["quantity"] == "4000"


def test_close_without_position_sells_nothing_but_clears_stops(client: FakeFuturesClient) -> None:
    client.algo_orders.append({"algoId": 5, "orderType": "STOP_MARKET", "symbol": "BTCUSDT", "side": "SELL",
                               "triggerPrice": "74000"})
    result = _executor(client).execute_order_intent(_close(), CYCLE)
    assert result["status"] == NO_POSITION and result["cancelled_stops"] == 1
    assert client.sent("futures_create_order") == [] and client.algo_orders == []


def test_close_never_sells_a_short_position(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [-0.5, 78_000.0]
    result = _executor(client).execute_order_intent(_close(), CYCLE)
    assert result["status"] == NO_POSITION
    assert client.sent("futures_create_order") == []


def test_if_the_sale_fails_the_stop_stays(client: FakeFuturesClient) -> None:
    _held(client)
    client.fail["futures_create_order"] = [api_error(-4131, "PERCENT_PRICE filter limit")]
    with pytest.raises(OrderRejectedError):
        _executor(client).execute_order_intent(_close(), CYCLE)
    assert len(client.algo_orders) == 1 and client.sent("futures_cancel_algo_order") == []
    assert client.positions["BTCUSDT"][0] == pytest.approx(0.012)


def test_a_sale_whose_answer_is_lost_is_not_sent_twice(client: FakeFuturesClient) -> None:
    _held(client)
    client.fail_after["futures_create_order"] = [network_error()]
    result = _executor(client).execute_order_intent(_close(), CYCLE)
    assert result["status"] == FILLED and len(client.sent("futures_create_order")) == 1
    assert client.positions == {} and client.algo_orders == []


def test_a_stop_that_cannot_be_cancelled_does_not_undo_the_close(client: FakeFuturesClient, log: list[str]) -> None:
    _held(client)
    client.fail["futures_cancel_algo_order"] = [api_error(-2011, "Unknown order sent.")]
    result = _executor(client).execute_order_intent(_close(), CYCLE)
    assert result["status"] == FILLED and result["cancelled_stops"] == 0
    assert any("some stop was left uncancelled" in line for line in log)


# --- Other intents -----------------------------------------------------------


def test_hold_does_nothing(client: FakeFuturesClient) -> None:
    result = _executor(client).execute_order_intent(OrderIntent("BTCUSDT", HOLD, 0.0, MARKET, None, "NO_BREAKOUT"), CYCLE)
    assert (result["status"], result["reason"]) == (SKIPPED, "NO_BREAKOUT")
    assert client.calls == []


@pytest.mark.parametrize("intent", [None, "OPEN_LONG", {"symbol": "BTCUSDT", "action": "OPEN_LONG"}])
def test_only_order_intents_are_accepted(client: FakeFuturesClient, intent: Any) -> None:
    with pytest.raises(OrderRejectedError, match="OrderIntent"):
        _executor(client).execute_order_intent(intent, CYCLE)


def test_client_ids_fit_the_exchange_limit(client: FakeFuturesClient) -> None:
    client.symbols["1000000MOGUSDT"] = symbol_info("1000000MOGUSDT", step="1", min_qty="1", tick="0.0001")
    client.marks["1000000MOGUSDT"] = 2.0
    executor = _executor(client)
    executor.execute_order_intent(_open("1000000MOGUSDT", 100.0, 1.5), "2026-10-09-a-very-long-retry")
    identifier = client.sent("futures_create_order")[0]["newClientOrderId"]
    assert len(identifier) <= 36 and identifier.endswith("1000000MOGUSDT-o")


def test_without_cycle_the_ids_come_from_the_clock(client: FakeFuturesClient) -> None:
    _executor(client).execute_order_intent(_open())
    assert client.sent("futures_create_order")[0]["newClientOrderId"] == "tb-20261009000130-BTCUSDT-o"


# --- DRY_RUN -----------------------------------------------------------------


def _dry_payloads(log: list[str]) -> list[tuple[str, dict[str, Any]]]:
    found = []
    for line in log:
        if line.startswith("INFO|DRY_RUN | "):
            _, request, body = line.rstrip("\n").split(" | ", 2)
            found.append((request, json.loads(body)))
    return found


def test_dry_run_open_logs_the_exact_payloads_and_sends_nothing(
    client: FakeFuturesClient, tmp_path: Path, log: list[str]
) -> None:
    paper = _paper(tmp_path)
    executor = _executor(client, mode=DRY_RUN, paper=paper)
    result = executor.execute_order_intent(_open(volume=0.0129, stop=76_543.21), CYCLE)

    assert result["status"] == SIMULATED and result["mode"] == DRY_RUN
    assert result["quantity"] == pytest.approx(0.012) and result["stop_price"] == pytest.approx(76_543.2)
    assert _dry_payloads(log) == [
        ("POST /fapi/v1/order", {
            "symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.012",
            "newOrderRespType": "RESULT", "newClientOrderId": "tb-20261009-BTCUSDT-o",
        }),
        ("POST /fapi/v1/algoOrder", {
            "algoType": "CONDITIONAL", "symbol": "BTCUSDT", "side": "SELL", "type": "STOP_MARKET",
            "quantity": "0.012", "triggerPrice": "76543.2", "reduceOnly": "true",
            "workingType": "CONTRACT_PRICE", "clientAlgoId": "tb-20261009-BTCUSDT-s",
        }),
    ]
    assert [body for _, body in _dry_payloads(log)] == result["payloads"]
    private = {"futures_create_order", "futures_create_algo_order", "futures_account", "futures_change_leverage",
               "futures_position_information", "futures_get_position_mode", "futures_cancel_algo_order"}
    assert not private & {name for name, _ in client.calls}
    assert paper.positions["BTCUSDT"]["quantity"] == pytest.approx(0.012)
    assert paper.balance == pytest.approx(10_000.0 - 0.012 * 80_000.0 * 0.0005)


def test_dry_run_payloads_match_the_real_ones(client: FakeFuturesClient, tmp_path: Path) -> None:
    simulated = _executor(client, mode=DRY_RUN, paper=_paper(tmp_path)).execute_order_intent(_open(), CYCLE)
    real = _executor(client).execute_order_intent(_open(), CYCLE)
    assert simulated["payloads"] == real["payloads"]


def test_dry_run_close_logs_reduce_only_and_books_the_result(
    client: FakeFuturesClient, tmp_path: Path, log: list[str]
) -> None:
    paper = _paper(tmp_path)
    executor = _executor(client, mode=DRY_RUN, paper=paper)
    executor.execute_order_intent(_open(), CYCLE)
    client.marks["BTCUSDT"] = 82_000.0
    log.clear()
    result = executor.execute_order_intent(_close(), "2026-10-12")

    assert result["status"] == SIMULATED
    request, sell = _dry_payloads(log)[0]
    assert request == "POST /fapi/v1/order"
    assert sell["side"] == "SELL" and sell["reduceOnly"] == "true" and sell["quantity"] == "0.012"
    fees = 0.012 * (80_000.0 + 82_000.0) * 0.0005
    assert result["pnl"] == pytest.approx(0.012 * 2_000.0 - fees)
    assert paper.positions == {} and paper.balance == pytest.approx(10_000.0 + 24.0 - fees)
    assert client.sent("futures_create_order") == []


def test_dry_run_close_without_position(client: FakeFuturesClient, tmp_path: Path) -> None:
    result = _executor(client, mode=DRY_RUN, paper=_paper(tmp_path)).execute_order_intent(_close(), CYCLE)
    assert result["status"] == NO_POSITION


def test_dry_run_applies_the_same_minimums(client: FakeFuturesClient, tmp_path: Path) -> None:
    paper = _paper(tmp_path)
    executor = _executor(client, mode=DRY_RUN, paper=paper)
    assert executor.execute_order_intent(_open("DOGEUSDT", 19.9, 0.2), CYCLE)["reason"] == BELOW_MIN_NOTIONAL
    executor.execute_order_intent(_open(), CYCLE)
    assert executor.execute_order_intent(_open(), CYCLE)["reason"] == POSITION_EXISTS
    assert list(paper.positions) == ["BTCUSDT"]


def test_dry_run_account_and_positions_come_from_the_paper_account(client: FakeFuturesClient, tmp_path: Path) -> None:
    executor = _executor(client, mode=DRY_RUN, paper=_paper(tmp_path, fee_rate=0.0))
    executor.execute_order_intent(_open(volume=0.012, stop=76_000.0), CYCLE)
    client.marks["BTCUSDT"] = 81_000.0

    account = executor.get_account()
    assert account.wallet_balance == pytest.approx(10_000.0)
    assert account.unrealized_pnl == pytest.approx(12.0)
    assert account.available_balance == pytest.approx(10_000.0 + 12.0 - 0.012 * 81_000.0)
    (position,) = executor.get_open_positions()
    assert (position.symbol, position.direction) == ("BTCUSDT", 1)
    assert position.nominal_value == pytest.approx(0.012 * 81_000.0)
    assert position.risk_usd == pytest.approx(0.012 * 4_000.0)
    assert client.sent("futures_account") == []


def test_dry_run_account_and_positions_share_one_price_read(client: FakeFuturesClient, tmp_path: Path) -> None:
    executor = _executor(client, mode=DRY_RUN, paper=_paper(tmp_path, fee_rate=0.0))
    executor.execute_order_intent(_open(volume=0.012, stop=76_000.0), CYCLE)
    read = client.futures_mark_price

    def drifting(**parameters: Any) -> dict[str, Any]:  # the price moves between one read and the next
        answer = read(**parameters)
        client.marks[parameters["symbol"]] += 100.0
        return answer

    client.futures_mark_price = drifting  # type: ignore[method-assign]
    before = len(client.sent("futures_mark_price"))
    account, (position,) = executor.get_account_and_positions()

    assert len(client.sent("futures_mark_price")) == before + 1
    assert account.unrealized_pnl == pytest.approx(position.quantity * (position.mark_price - position.entry_price))
    assert account.available_balance == pytest.approx(
        account.wallet_balance + account.unrealized_pnl - position.quantity * position.mark_price)


def test_on_the_exchange_account_and_positions_are_the_usual_two_queries(client: FakeFuturesClient) -> None:
    executor = _executor(client, mode=TESTNET)
    account, positions = executor.get_account_and_positions()
    assert account == executor.get_account() and positions == executor.get_position_details()


# --- Paper account -----------------------------------------------------------


def _candle(day: int, open_: float, low: float) -> dict[str, float]:
    start = NOW_MS - 90_000 + day * DAY_MS
    return {"open_time": start, "close_time": start + DAY_MS - 1, "open": open_, "high": open_ + 5.0, "low": low,
            "close": open_}


def _frame(*candles: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(list(candles))


def test_paper_account_survives_a_restart(tmp_path: Path) -> None:
    first = _paper(tmp_path, fee_rate=0.001)
    first.open("BTCUSDT", 0.5, 100.0, 90.0, NOW_MS)
    again = PaperFuturesAccount(tmp_path / "paper.json", 999_999.0, 0.001)
    assert again.balance == pytest.approx(10_000.0 - 0.05)
    assert again.positions == first.positions
    assert not (tmp_path / "paper.tmp").exists()


def test_paper_account_refuses_an_unreadable_file(tmp_path: Path) -> None:
    (tmp_path / "paper.json").write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(BrokerUnavailableError, match="cannot be read"):
        _paper(tmp_path)
    (tmp_path / "paper.json").write_text('{"balance": 5}', encoding="utf-8")
    with pytest.raises(BrokerUnavailableError, match="cannot be read"):
        _paper(tmp_path)


@pytest.mark.parametrize(("capital", "fee"), [(0.0, 0.0005), (-1.0, 0.0005), (100.0, -0.1), (100.0, 0.5)])
def test_paper_account_validates_its_arguments(tmp_path: Path, capital: float, fee: float) -> None:
    with pytest.raises(ValueError):
        _paper(tmp_path, capital, fee)


def test_paper_account_holds_one_position_per_symbol(tmp_path: Path) -> None:
    paper = _paper(tmp_path)
    paper.open("BTCUSDT", 0.5, 100.0, 90.0, NOW_MS)
    with pytest.raises(OrderRejectedError, match="already has a position"):
        paper.open("BTCUSDT", 0.5, 100.0, 90.0, NOW_MS)
    assert paper.close("ETHUSDT", 100.0, "x", NOW_MS) is None


def test_paper_stop_is_hit_by_the_low_of_a_daily_candle(tmp_path: Path) -> None:
    paper = _paper(tmp_path, fee_rate=0.0)
    paper.open("BTCUSDT", 2.0, 100.0, 90.0, NOW_MS)
    stopped = paper.settle({"BTCUSDT": _frame(_candle(0, 100.0, 95.0), _candle(1, 96.0, 89.0))}, None, NOW_MS + 2 * DAY_MS)
    assert [trade["symbol"] for trade in stopped] == ["BTCUSDT"]
    assert stopped[0]["exit_price"] == 90.0 and stopped[0]["reason"] == "STOP"
    assert stopped[0]["pnl"] == pytest.approx(-20.0)
    assert paper.positions == {} and paper.balance == pytest.approx(9_980.0)
    assert paper.trades == stopped


def test_paper_stop_fills_at_the_open_when_the_candle_gaps_below(tmp_path: Path) -> None:
    paper = _paper(tmp_path, fee_rate=0.0)
    paper.open("BTCUSDT", 2.0, 100.0, 90.0, NOW_MS)
    stopped = paper.settle({"BTCUSDT": _frame(_candle(0, 100.0, 95.0), _candle(1, 85.0, 80.0))}, None, NOW_MS + 2 * DAY_MS)
    assert stopped[0]["exit_price"] == 85.0 and stopped[0]["pnl"] == pytest.approx(-30.0)


def test_paper_stop_on_the_entry_day_fills_at_the_stop(tmp_path: Path) -> None:
    paper = _paper(tmp_path, fee_rate=0.0)
    paper.open("BTCUSDT", 2.0, 100.0, 90.0, NOW_MS)
    # The candle of the day of the buy opened 90 s before buying: its open is not an exit price.
    stopped = paper.settle({"BTCUSDT": _frame(_candle(0, 88.0, 80.0))}, None, NOW_MS + DAY_MS)
    assert stopped[0]["exit_price"] == 90.0


def test_paper_ignores_candles_closed_before_the_entry(tmp_path: Path) -> None:
    paper = _paper(tmp_path, fee_rate=0.0)
    paper.open("BTCUSDT", 2.0, 100.0, 90.0, NOW_MS)
    assert paper.settle({"BTCUSDT": _frame(_candle(-2, 80.0, 70.0), _candle(-1, 80.0, 70.0))}, None, NOW_MS + 60_000) == []
    assert "BTCUSDT" in paper.positions


def test_paper_does_not_check_the_same_candle_twice(tmp_path: Path) -> None:
    paper = _paper(tmp_path, fee_rate=0.0)
    paper.open("BTCUSDT", 2.0, 100.0, 90.0, NOW_MS)
    candles = _frame(_candle(0, 100.0, 95.0))
    assert paper.settle({"BTCUSDT": candles}, None, NOW_MS + DAY_MS) == []
    assert paper.positions["BTCUSDT"]["checked_until"] == candles["close_time"].iloc[-1]
    # Even if that same candle arrived now with another low, it is already checked.
    assert paper.settle({"BTCUSDT": _frame(_candle(0, 100.0, 50.0))}, None, NOW_MS + DAY_MS) == []


def test_paper_keeps_a_position_whose_symbol_has_no_candles(tmp_path: Path) -> None:
    paper = _paper(tmp_path)
    paper.open("BTCUSDT", 2.0, 100.0, 90.0, NOW_MS)
    assert paper.settle({}, None, NOW_MS + DAY_MS) == []
    assert "BTCUSDT" in paper.positions


def test_paper_charges_funding_to_longs(tmp_path: Path) -> None:
    paper = _paper(tmp_path, fee_rate=0.0)
    paper.open("BTCUSDT", 2.0, 100.0, 90.0, NOW_MS)
    asked: list[tuple[str, int, int]] = []

    def funding(symbol: str, start: int, end: int) -> list[dict[str, Any]]:
        asked.append((symbol, start, end))
        return [{"fundingTime": start + 1, "fundingRate": "0.0001", "markPrice": "110.0"},
                {"fundingTime": start + 2, "fundingRate": "-0.00005", "markPrice": ""}]

    paper.settle({}, funding, NOW_MS + DAY_MS)
    assert asked == [("BTCUSDT", NOW_MS + 1, NOW_MS + DAY_MS)]
    charged = 2.0 * 110.0 * 0.0001 - 2.0 * 100.0 * 0.00005
    assert paper.balance == pytest.approx(10_000.0 - charged)
    assert paper.positions["BTCUSDT"]["funding"] == pytest.approx(charged)
    paper.settle({}, funding, NOW_MS + DAY_MS)
    assert len(asked) == 1  # that span is already charged
    trade = paper.close("BTCUSDT", 100.0, "x", NOW_MS + DAY_MS)
    assert trade["pnl"] == pytest.approx(-charged)


def test_paper_funding_stops_when_the_stop_is_hit(tmp_path: Path) -> None:
    paper = _paper(tmp_path, fee_rate=0.0)
    paper.open("BTCUSDT", 2.0, 100.0, 90.0, NOW_MS)
    asked: list[int] = []
    second_day = _candle(1, 96.0, 89.0)
    paper.settle({"BTCUSDT": _frame(_candle(0, 100.0, 95.0), second_day)},
                 lambda symbol, start, end: asked.append(end) or [], NOW_MS + 5 * DAY_MS)
    assert asked == [second_day["open_time"]]


def test_paper_funding_that_cannot_be_read_stays_pending(tmp_path: Path, log: list[str]) -> None:
    paper = _paper(tmp_path, fee_rate=0.0)
    paper.open("BTCUSDT", 2.0, 100.0, 90.0, NOW_MS)

    def down(symbol: str, start: int, end: int) -> list[dict[str, Any]]:
        raise BrokerError("no network")

    paper.settle({}, down, NOW_MS + DAY_MS)
    assert paper.positions["BTCUSDT"]["funded_until"] == NOW_MS and paper.balance == pytest.approx(10_000.0)
    assert any("paper funding pending" in line for line in log)


def test_settle_paper_uses_the_exchange_funding_and_only_in_dry_run(client: FakeFuturesClient, tmp_path: Path) -> None:
    paper = _paper(tmp_path, fee_rate=0.0)
    executor = _executor(client, mode=DRY_RUN, paper=paper)
    executor.execute_order_intent(_open(volume=0.012, stop=76_000.0), CYCLE)
    client.funding["BTCUSDT"] = [
        {"fundingTime": NOW_MS + 8 * 3_600_000, "fundingRate": "0.0001", "markPrice": "80000"},
        {"fundingTime": NOW_MS - 1, "fundingRate": "0.5", "markPrice": "80000"},  # before the buy
    ]
    client.now_ms += DAY_MS
    assert executor.settle_paper({}) == []
    assert paper.balance == pytest.approx(10_000.0 - 0.012 * 80_000.0 * 0.0001)
    assert _executor(client).settle_paper({}) == []


def test_an_order_refused_for_a_passing_reason_is_sent_again(client: FakeFuturesClient) -> None:
    _held(client)
    client.fail["futures_create_order"] = [api_error(-1021, "Timestamp for this request is outside of the recvWindow.")]
    result = _executor(client).execute_order_intent(_close(), CYCLE)
    assert result["status"] == FILLED and client.positions == {}
    first, second = client.sent("futures_create_order")
    assert first == second  # same client id: the exchange cannot take it twice


def test_a_refusal_that_keeps_coming_is_a_rejection(client: FakeFuturesClient) -> None:
    _held(client)
    client.fail["futures_create_order"] = [api_error(-1021, "recvWindow") for _ in range(10)]
    with pytest.raises(OrderRejectedError):
        _executor(client, retries=2).execute_order_intent(_close(), CYCLE)
    assert len(client.sent("futures_create_order")) == 3 and "BTCUSDT" in client.positions


def test_a_partial_close_keeps_the_stop_on_what_is_left(client: FakeFuturesClient) -> None:
    _held(client)
    sell = client.futures_create_order

    def half_filled(**parameters: Any) -> dict[str, Any]:
        order = sell(**{**parameters, "quantity": "0.005"})
        order["status"] = "EXPIRED"
        client.orders[parameters["newClientOrderId"]]["status"] = "EXPIRED"
        return order

    client.futures_create_order = half_filled  # type: ignore[method-assign]
    result = _executor(client).execute_order_intent(_close(), CYCLE)
    assert result["status"] == PARTIAL_CLOSE and result["quantity"] == pytest.approx(0.005)
    assert client.positions["BTCUSDT"][0] == pytest.approx(0.007)
    assert [order["algoId"] for order in client.algo_orders] == [77]  # still protected


def test_the_daily_refresh_picks_up_changed_rules_and_delistings(client: FakeFuturesClient) -> None:
    executor = _executor(client)
    assert set(executor.load_market_rules(["BTCUSDT", "DOGEUSDT"])) == {"BTCUSDT", "DOGEUSDT"}
    client.symbols["BTCUSDT"] = symbol_info("BTCUSDT", step="0.01", min_qty="0.01", tick="0.10", min_notional="100")
    client.symbols["DOGEUSDT"] = symbol_info("DOGEUSDT", status="SETTLING")
    assert executor.load_market_rules(["BTCUSDT", "DOGEUSDT"])["BTCUSDT"].step_size == Decimal("0.001")  # cached
    rules = executor.load_market_rules(["BTCUSDT", "DOGEUSDT"], refresh=True)
    assert list(rules) == ["BTCUSDT"] and rules["BTCUSDT"].step_size == Decimal("0.01")
    result = executor.execute_order_intent(_open("DOGEUSDT", 100.0, 0.2), CYCLE)
    assert (result["status"], result["reason"]) == (REJECTED, NO_RULES)  # no entries in a contract being delisted


def test_a_stop_placed_by_hand_counts_as_protection(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [0.012, 78_000.0]
    client.algo_orders.append({"algoId": 5, "orderType": "TRAILING_STOP_MARKET", "symbol": "BTCUSDT", "side": "SELL"})
    executor = _executor(client)
    executor.load_market_rules(["BTCUSDT"])
    assert executor.protect_open_positions({"BTCUSDT": 3_000.0}, CYCLE) == []
    assert client.sent("futures_create_algo_order") == [] and client.sent("futures_create_order") == []


def test_no_repair_is_attempted_on_a_hedge_mode_account(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [0.012, 78_000.0]
    client.dual_side = True
    with pytest.raises(BrokerUnavailableError, match="hedge"):
        _executor(client).protect_open_positions({"BTCUSDT": 3_000.0}, CYCLE)
    assert client.sent("futures_create_algo_order") == [] and client.sent("futures_create_order") == []


def test_a_restored_stop_goes_at_the_protective_distance_from_the_entry(client: FakeFuturesClient) -> None:
    client.positions["BTCUSDT"] = [0.012, 78_000.0]
    executor = _executor(client)
    executor.load_market_rules(["BTCUSDT"])
    ((result),) = executor.protect_open_positions({"BTCUSDT": 3_000.05}, CYCLE)
    assert (result["status"], result["stop_price"]) == ("STOP_RESTORED", 74_999.9)
    ((stop),) = client.algo_orders
    assert (stop["triggerPrice"], stop["quantity"], stop["clientAlgoId"]) == ("74999.9", "0.012", "tb-20261009-BTCUSDT-r")
