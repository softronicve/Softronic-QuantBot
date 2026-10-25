"""Tests of the daily basket cycle (``DailyRunner``)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from pydantic import ValidationError

from config.settings import DEFAULT_FUTURES_BASKET, MEASURED_FUTURES_BASKET, STRESS_TEST_FUTURES_BASKET, Settings
from src.data.candle_loader import DailyCandleLoader
from src.engine import daily_runner
from src.engine.daily_runner import (
    ENTRIES_DECIDED,
    INCOMPLETE,
    ALREADY_DISPATCHED,
    ALREADY_EXECUTED,
    CLOSE_FAILED,
    COMPLETED,
    COMPLETED_WITH_ERRORS,
    ERROR,
    NO_BALANCE,
    NO_DATA,
    TOO_LATE,
    WRONG_DATE,
    DailyRunner,
    build_runner,
    effective_mode,
)
from src.engine.notifier import TelegramNotifier
from src.engine.orchestrator import CLOSE_LONG, OPEN_LONG
from src.engine.state_store import DailyStateStore, StateError
from src.execution.errors import BrokerError, BrokerUnavailableError
from src.execution.binance_futures_executor import (
    CLOSED_UNPROTECTED,
    STOP_FAILED_CLOSED,
    STOP_RESTORED,
    DRY_RUN,
    FILLED,
    FuturesPosition,
    LIVE,
    SIMULATED,
    TESTNET,
    BinanceFuturesExecutor,
    PaperFuturesAccount,
)
from src.instance_lock import StateInUseError
from src.risk.portfolio_risk import PortfolioRiskManager
from tests.fake_futures import DAY_MS, NOW_MS, FakeFuturesClient, api_error, daily_klines, network_error, symbol_info

TODAY = "2026-10-09"
TOMORROW = "2026-10-10"
BREAKS = 104.0  # closes above the 20-candle high (~103)
QUIET = 100.0  # inside the range
FALLS = 96.0  # closes below the 10-candle low (~97)
# With a balance of 10 000 the 10 % cap is 1 000 USDT: 9.615 units at 104.
QUANTITY = 9.615


@pytest.fixture
def log() -> Any:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}")
    yield lines
    logger.remove(sink)


def _client(lasts: dict[str, float], wallet: float = 10_000.0) -> FakeFuturesClient:
    fake = FakeFuturesClient(
        {symbol: symbol_info(symbol, step="0.001", min_qty="0.001", tick="0.01") for symbol in lasts}, wallet
    )
    _prices(fake, lasts)
    return fake


def _prices(fake: FakeFuturesClient, lasts: dict[str, float]) -> None:
    for symbol, last in lasts.items():
        fake.klines[symbol] = daily_klines(last, now_ms=fake.now_ms)
        fake.marks[symbol] = last


def _next_day(fake: FakeFuturesClient, lasts: dict[str, float]) -> None:
    fake.now_ms += DAY_MS
    _prices(fake, lasts)


class Rig:
    """A daily cycle set up on the fake exchange."""

    def __init__(self, client: FakeFuturesClient, tmp_path: Path, mode: str = DRY_RUN,
                 basket: list[str] | None = None, **options: Any) -> None:
        self.client = client
        self.paper = PaperFuturesAccount(tmp_path / "paper.json", 10_000.0, 0.0005)
        self.executor = BinanceFuturesExecutor(
            mode=mode, client=client, paper=self.paper, sleep=lambda _: None, clock=lambda: client.now_ms / 1000,
        )
        self.store = DailyStateStore(tmp_path / "runtime_state.json")
        self.runner = DailyRunner(
            self.executor, DailyCandleLoader(self.executor, candles=150), self.store,
            list(client.symbols) if basket is None else basket, clock=lambda: client.now_ms / 1000, **options,
        )
        self.sent: list[tuple[str, str]] = []
        execute = self.executor.execute_order_intent

        def spy(intent: Any, cycle_id: str | None = None) -> dict[str, Any]:
            self.sent.append((intent.symbol, intent.action))
            return execute(intent, cycle_id)

        self.executor.execute_order_intent = spy  # type: ignore[method-assign]

    def run(self, target_date: str | None = None) -> dict[str, Any]:
        return self.runner.run_daily_cycle(target_date)


def _hold_on_exchange(client: FakeFuturesClient, symbol: str, quantity: float = 5.0, stop: str = "80.00") -> None:
    client.positions[symbol] = [quantity, 100.0]
    client.algo_orders.append({
        "algoId": 900 + len(client.algo_orders), "clientAlgoId": f"tb-20261001-{symbol}-s",
        "orderType": "STOP_MARKET", "symbol": symbol, "side": "SELL", "triggerPrice": stop,
    })


# --- Full cycle --------------------------------------------------------------


def test_full_cycle_in_dry_run(tmp_path: Path, log: list[str]) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": QUIET, "CCCUSDT": QUIET}), tmp_path)
    summary = rig.run()

    assert summary["status"] == COMPLETED and summary["mode"] == DRY_RUN and summary["cycle_date"] == TODAY
    assert summary["closes"] == [] and summary["holds"] == {"NO_BREAKOUT": 2}
    (opened,) = summary["opens"]
    assert (opened["symbol"], opened["status"]) == ("AAAUSDT", SIMULATED)
    assert opened["quantity"] == pytest.approx(QUANTITY) and opened["price"] == pytest.approx(104.0)
    assert 97.5 < opened["stop_price"] < 99.0

    position = rig.paper.positions["AAAUSDT"]
    assert position["quantity"] == pytest.approx(QUANTITY) and position["stop_price"] == opened["stop_price"]
    assert rig.store.is_cycle_already_executed(TODAY)
    assert rig.store.get_cycle(TODAY)["summary"]["opens"] == summary["opens"]
    assert "AAAUSDT:OPEN_LONG" in rig.store.dispatched(TODAY)
    # In DRY_RUN no order goes out and nothing private of the account is touched.
    assert rig.client.sent("futures_create_order") == [] and rig.client.sent("futures_create_algo_order") == []
    assert rig.client.sent("futures_account") == []
    assert sum(line.startswith("INFO|DRY_RUN | POST") for line in log) == 2
    assert any(line.startswith(f"INFO|Cycle {TODAY} finished (COMPLETED): 0 closes, 1 buys") for line in log)


def test_full_cycle_on_the_exchange(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": QUIET}), tmp_path, mode=TESTNET)
    summary = rig.run()

    assert summary["status"] == COMPLETED and summary["opens"][0]["status"] == FILLED
    assert rig.client.positions["AAAUSDT"][0] == pytest.approx(QUANTITY)
    (stop,) = rig.client.algo_orders
    assert stop["orderType"] == "STOP_MARKET" and stop["reduceOnly"] is True
    assert float(stop["quantity"]) == pytest.approx(QUANTITY)
    assert rig.client.leverage == {"AAAUSDT": 1}
    assert rig.paper.positions == {}  # outside DRY_RUN the paper account plays no part


def test_exits_are_dispatched_before_entries(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": FALLS, "CCCUSDT": BREAKS}), tmp_path)
    rig.paper.open("BBBUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    summary = rig.run()

    assert rig.sent == [("BBBUSDT", CLOSE_LONG), ("AAAUSDT", OPEN_LONG), ("CCCUSDT", OPEN_LONG)]
    assert [result["symbol"] for result in summary["closes"]] == ["BBBUSDT"]
    assert summary["positions_before"] == ["BBBUSDT"]
    assert sorted(rig.paper.positions) == ["AAAUSDT", "CCCUSDT"]
    assert rig.paper.trades[0]["reason"] == "EXIT_CHANNEL_BROKEN"


def test_exits_are_dispatched_before_entries_on_the_exchange(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS, "BBBUSDT": FALLS})
    _hold_on_exchange(client, "BBBUSDT")
    rig = Rig(client, tmp_path, mode=TESTNET)
    rig.run()

    orders = client.sent("futures_create_order")
    assert [(order["symbol"], order["side"]) for order in orders] == [("BBBUSDT", "SELL"), ("AAAUSDT", "BUY")]
    assert orders[0]["reduceOnly"] == "true"
    assert [order["symbol"] for order in client.algo_orders] == ["AAAUSDT"]  # the orphan stop of BBB was canceled


def test_position_sizes_use_the_wallet_balance_not_the_free_collateral(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": QUIET, "BBBUSDT": BREAKS}), tmp_path)
    rig.paper.open("AAAUSDT", 50.0, 100.0, 96.0, NOW_MS - DAY_MS)  # takes up half of the collateral
    summary = rig.run()
    assert "AAAUSDT" in rig.paper.positions and rig.executor.get_account().available_balance < 4_100.0
    # 10 % of the realized balance (~9 997) at 104; with the free collateral it would be about 4.8 units.
    assert summary["opens"][0]["quantity"] == pytest.approx(9.61, abs=0.01)


def test_the_open_risk_ceiling_holds_end_to_end(tmp_path: Path) -> None:
    coins = {f"{letter * 3}USDT": BREAKS for letter in "ABCDEFGH"}
    rig = Rig(_client(coins), tmp_path)
    summary = rig.run()
    # Each position risks ~0.56 % of the balance: seven fit in the 4 %, not eight.
    assert len(rig.paper.positions) == 7 and len(summary["opens"]) == 7
    assert summary["holds"] == {"MAX_HEAT_EXCEEDED": 1}
    risk = sum(p["quantity"] * (p["entry_price"] - p["stop_price"]) for p in rig.paper.positions.values())
    assert risk <= 0.04 * 10_000.0


# --- Idempotency -------------------------------------------------------------


def test_the_same_day_is_never_executed_twice(tmp_path: Path, log: list[str]) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": BREAKS}), tmp_path)
    first = rig.run()
    calls, positions, balance = len(rig.client.calls), rig.paper.positions, rig.paper.balance

    second = rig.run()
    assert first["status"] == COMPLETED and second == {"cycle_date": TODAY, "status": ALREADY_EXECUTED}
    assert len(rig.client.calls) == calls and len(rig.sent) == 2
    assert rig.paper.positions == positions and rig.paper.balance == balance
    assert any(line == f"INFO|Cycle {TODAY}: already run; not repeated\n" for line in log)


def test_a_restarted_process_does_not_repeat_the_day(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS})
    Rig(client, tmp_path).run()
    restarted = Rig(client, tmp_path)
    assert restarted.run()["status"] == ALREADY_EXECUTED
    assert restarted.sent == [] and list(restarted.paper.positions) == ["AAAUSDT"]


def test_the_next_day_runs_again_and_keeps_what_is_held(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": QUIET}), tmp_path)
    rig.run()
    _next_day(rig.client, {"AAAUSDT": BREAKS, "BBBUSDT": BREAKS})
    summary = rig.run()

    assert summary["cycle_date"] == TOMORROW and summary["status"] == COMPLETED
    assert summary["positions_before"] == ["AAAUSDT"] and summary["holds"] == {"TREND_INTACT": 1}
    assert [result["symbol"] for result in summary["opens"]] == ["BBBUSDT"]
    assert rig.sent == [("AAAUSDT", OPEN_LONG), ("BBBUSDT", OPEN_LONG)]
    assert rig.store.is_cycle_already_executed(TODAY) and rig.store.is_cycle_already_executed(TOMORROW)


def test_an_order_already_sent_is_not_repeated_after_a_crash(tmp_path: Path, log: list[str]) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": BREAKS}), tmp_path)
    # The process died after buying AAA and before finishing; then its stop fired and the account no longer has it.
    rig.store.record_dispatch(TODAY, "AAAUSDT", OPEN_LONG, {"symbol": "AAAUSDT", "status": FILLED, "quantity": 9.6})
    summary = rig.run()

    assert rig.sent == [("BBBUSDT", OPEN_LONG)]
    assert [(r["symbol"], r["status"]) for r in summary["opens"]] == [("AAAUSDT", ALREADY_DISPATCHED),
                                                                   ("BBBUSDT", SIMULATED)]
    assert summary["status"] == COMPLETED and list(rig.paper.positions) == ["BBBUSDT"]
    assert any("AAAUSDT: OPEN_LONG was already sent in this cycle" in line for line in log)


def test_an_order_that_was_not_sent_is_tried_again(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    rig.store.record_dispatch(TODAY, "AAAUSDT", OPEN_LONG, {"symbol": "AAAUSDT", "status": ERROR})
    assert rig.run()["opens"][0]["status"] == SIMULATED


# --- Cycle date --------------------------------------------------------------


@pytest.mark.parametrize("target", ["2026-10-08", TOMORROW, "2025-01-01"])
def test_only_todays_cycle_can_be_run(tmp_path: Path, log: list[str], target: str) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    assert rig.run(target) == {"cycle_date": target, "status": WRONG_DATE}
    assert rig.client.calls == [] and rig.paper.positions == {}
    assert not rig.store.is_cycle_already_executed(target)
    assert any(line.startswith(f"ERROR|Cycle {target}: not run") for line in log)


def test_todays_date_can_be_given_explicitly(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    assert rig.run(TODAY)["status"] == COMPLETED
    assert rig.run(TODAY)["status"] == ALREADY_EXECUTED


@pytest.mark.parametrize("target", ["today", "2026-10-9", "09/10/2026", ""])
def test_an_invalid_date_is_an_error(tmp_path: Path, target: str) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    with pytest.raises(StateError, match="YYYY-MM-DD"):
        rig.run(target)
    assert rig.client.calls == []


# --- Network failures --------------------------------------------------------


def test_transient_network_failures_are_ridden_out(tmp_path: Path, log: list[str]) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": QUIET}), tmp_path)
    rig.client.fail.update(
        futures_exchange_info=[api_error(-1001, "Internal error; unable to process your request.", 503)],
        futures_time=[network_error()],
        futures_klines=[network_error(), network_error()],
        futures_mark_price=[network_error()],
        futures_funding_rate=[network_error()],
    )
    summary = rig.run()
    assert summary["status"] == COMPLETED and summary["symbols_with_data"] == 2
    assert [result["status"] for result in summary["opens"]] == [SIMULATED]
    assert sum("retrying in" in line for line in log) == 5


def test_transient_network_failures_are_ridden_out_on_the_exchange(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS, "BBBUSDT": FALLS})
    _hold_on_exchange(client, "BBBUSDT")
    client.fail.update(
        futures_account=[network_error()],
        futures_position_information=[network_error()],
        futures_get_open_algo_orders=[api_error(-1001, "Disconnected", 503)],
        futures_change_leverage=[network_error()],
    )
    # The answer to the sell is lost and the buy does not get through on the first try.
    client.fail_after["futures_create_order"] = [network_error()]
    rig = Rig(client, tmp_path, mode=TESTNET)
    summary = rig.run()

    assert summary["status"] == COMPLETED
    assert "BBBUSDT" not in client.positions and client.positions["AAAUSDT"][0] == pytest.approx(QUANTITY)
    assert [(order["symbol"], order["side"]) for order in client.sent("futures_create_order")] == [
        ("BBBUSDT", "SELL"), ("AAAUSDT", "BUY"),
    ]
    assert [order["symbol"] for order in client.algo_orders] == ["AAAUSDT"]


def test_a_symbol_whose_candles_never_arrive_is_skipped(tmp_path: Path, log: list[str]) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": BREAKS}), tmp_path)
    rig.client.fail["futures_klines"] = [network_error() for _ in range(4)]  # uses up the retries of AAA
    summary = rig.run()
    assert summary["status"] == COMPLETED and summary["symbols_with_data"] == 1
    assert [result["symbol"] for result in summary["opens"]] == ["BBBUSDT"]
    assert list(rig.paper.positions) == ["BBBUSDT"]
    assert any(line.startswith("WARNING|AAAUSDT: no daily candles") for line in log)


def test_if_the_exchange_is_down_the_cycle_stays_pending(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    rig.client.fail["futures_time"] = [network_error() for _ in range(4)]
    with pytest.raises(BrokerError, match="could not reach"):
        rig.run()
    assert not rig.store.is_cycle_already_executed(TODAY) and rig.store.get_cycle(TODAY) is None
    assert rig.sent == []
    # When the network is back, the same day can be run.
    assert rig.run()["status"] == COMPLETED and rig.sent == [("AAAUSDT", OPEN_LONG)]


def test_a_held_position_without_candles_leaves_the_day_open_until_they_arrive(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": FALLS, "BBBUSDT": BREAKS}), tmp_path)
    rig.paper.open("AAAUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    rig.client.fail["futures_klines"] = [network_error() for _ in range(4)]
    summary = rig.run()
    assert summary == {"cycle_date": TODAY, "status": INCOMPLETE, "unresolved": [], "without_data": ["AAAUSDT"]}
    assert "AAAUSDT" in rig.paper.positions and not rig.store.is_cycle_already_executed(TODAY)
    assert INCOMPLETE in daily_runner.RETRY

    # The retry finds the candles: the exit is checked and nothing already sent is sent again.
    summary = rig.run()
    assert summary["status"] == COMPLETED and summary["without_data"] == []
    assert [(order["symbol"], order["status"]) for order in summary["closes"]] == [("AAAUSDT", SIMULATED)]
    assert [order["symbol"] for order in summary["opens"]] == ["BBBUSDT"]
    assert rig.sent.count(("BBBUSDT", OPEN_LONG)) == 1 and rig.store.is_cycle_already_executed(TODAY)


def test_a_stop_hit_in_the_first_attempt_is_still_in_the_day_summary(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": FALLS, "CCCUSDT": FALLS}), tmp_path)
    rig.paper.open("AAAUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    rig.paper.open("CCCUSDT", 5.0, 100.0, 99.0, NOW_MS - DAY_MS)  # today's low goes through 99
    rig.client.fail["futures_klines"] = [network_error() for _ in range(4)]  # AAAUSDT, loaded first
    assert rig.run()["status"] == INCOMPLETE and "CCCUSDT" not in rig.paper.positions

    summary = rig.run()
    assert summary["status"] == COMPLETED and summary["stopped"] == ["CCCUSDT"]


def test_past_the_entry_deadline_a_day_without_candles_is_closed_and_says_so(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": FALLS, "BBBUSDT": QUIET}), tmp_path, max_entry_delay_minutes=0.0)
    rig.paper.open("AAAUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    rig.client.fail["futures_klines"] = [network_error() for _ in range(4)]
    summary = rig.run()
    assert summary["status"] == COMPLETED and summary["without_data"] == ["AAAUSDT"]
    assert summary["closes"] == [] and "AAAUSDT" in rig.paper.positions
    assert summary["holds"] == {"NO_DATA": 1, "NO_BREAKOUT": 1} and rig.store.is_cycle_already_executed(TODAY)


# --- Cycles that never start -------------------------------------------------


def test_without_valid_candles_nothing_is_traded_and_the_day_stays_pending(tmp_path: Path, log: list[str]) -> None:
    client = _client({"AAAUSDT": BREAKS})
    client.klines["AAAUSDT"] = daily_klines(BREAKS, now_ms=NOW_MS - 3 * DAY_MS)
    rig = Rig(client, tmp_path)
    assert rig.run() == {"cycle_date": TODAY, "status": NO_DATA}
    assert not rig.store.is_cycle_already_executed(TODAY) and rig.sent == []
    assert any("no symbol has valid daily candles" in line for line in log)


def test_an_empty_account_does_not_trade(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}, wallet=0.0), tmp_path, mode=TESTNET)
    assert rig.run()["status"] == NO_BALANCE
    assert rig.sent == [] and not rig.store.is_cycle_already_executed(TODAY)


def test_an_empty_basket_is_rejected(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS})
    client.symbols.clear()
    with pytest.raises(ValueError, match="basket"):
        Rig(client, tmp_path)


# --- Orders that fail --------------------------------------------------------


def test_a_failed_close_blocks_the_entries_and_the_day_is_retried(tmp_path: Path, log: list[str]) -> None:
    client = _client({"AAAUSDT": BREAKS, "BBBUSDT": FALLS})
    _hold_on_exchange(client, "BBBUSDT")
    client.fail["futures_create_order"] = [api_error(-4131, "PERCENT_PRICE filter limit")]
    rig = Rig(client, tmp_path, mode=TESTNET)
    summary = rig.run()

    assert summary == {"cycle_date": TODAY, "status": INCOMPLETE, "unresolved": ["BBBUSDT"], "without_data": []}
    assert rig.sent == [("BBBUSDT", CLOSE_LONG)]  # the entry waits for the close
    assert list(client.positions) == ["BBBUSDT"] and len(client.algo_orders) == 1  # still under its stop
    assert not rig.store.is_cycle_already_executed(TODAY)
    assert rig.store.dispatched(TODAY)["BBBUSDT:CLOSE_LONG"]["status"] == ERROR
    assert any(line.startswith("ERROR|BBBUSDT: CLOSE_LONG was not executed") for line in log)

    summary = rig.run()  # five minutes later the exchange takes the close
    assert [(order["symbol"], order["status"]) for order in summary["closes"]] == [("BBBUSDT", FILLED)]
    # As in a cycle run in one go, a day whose close failed buys nothing.
    assert [(order["symbol"], order["status"]) for order in summary["opens"]] == [("AAAUSDT", CLOSE_FAILED)]
    assert summary["status"] == COMPLETED_WITH_ERRORS and client.positions == {}
    assert rig.sent == [("BBBUSDT", CLOSE_LONG), ("BBBUSDT", CLOSE_LONG)]


def test_a_retry_buys_nothing_the_first_attempt_did_not(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    rig.store.mark_cycle_incomplete(TODAY, None)  # an earlier attempt decided the entries and saw no breakout
    summary = rig.run()
    assert [(order["symbol"], order["status"]) for order in summary["opens"]] == [("AAAUSDT", ENTRIES_DECIDED)]
    assert rig.sent == [] and rig.paper.positions == {}


def test_a_held_contract_on_its_way_out_keeps_its_exit_checked(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": FALLS, "BBBUSDT": QUIET}), tmp_path)
    rig.paper.open("AAAUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    rig.executor.load_market_rules(["AAAUSDT", "BBBUSDT"])  # known from an earlier day
    rig.client.symbols["AAAUSDT"] = symbol_info("AAAUSDT", status="SETTLING")
    summary = rig.run()
    assert summary["untradable"] == ["AAAUSDT"]
    assert [(order["symbol"], order["status"]) for order in summary["closes"]] == [("AAAUSDT", SIMULATED)]
    assert rig.paper.positions == {}


def test_a_close_filled_in_part_is_finished_on_the_retry_and_reported_whole(tmp_path: Path) -> None:
    client = _client({"BBBUSDT": FALLS})
    _hold_on_exchange(client, "BBBUSDT", quantity=4.0)
    sell = client.futures_create_order
    halves = [True]

    def half_once(**parameters: Any) -> dict[str, Any]:
        if halves and parameters["side"] == "SELL":
            halves.pop()
            order = sell(**{**parameters, "quantity": "1"})
            client.orders[parameters["newClientOrderId"]]["status"] = order["status"] = "EXPIRED"
            return order
        return sell(**parameters)

    client.futures_create_order = half_once  # type: ignore[method-assign]
    rig = Rig(client, tmp_path, mode=TESTNET)
    assert rig.run()["status"] == INCOMPLETE and client.positions["BBBUSDT"][0] == 3.0
    summary = rig.run()

    ((close),) = summary["closes"]
    assert (close["status"], close["quantity"]) == (FILLED, 4.0) and summary["status"] == COMPLETED
    first, rest = client.sent("futures_create_order")
    assert first["newClientOrderId"] == "tb-20261009-BBBUSDT-c" and rest["newClientOrderId"] == "tb-202610092-BBBUSDT-c"
    assert client.positions == {}


def test_an_entry_that_went_wrong_is_not_reported_as_done_when_repeated(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    rig.store.record_dispatch(TODAY, "AAAUSDT", OPEN_LONG,
                              {"symbol": "AAAUSDT", "action": OPEN_LONG, "status": STOP_FAILED_CLOSED})
    summary = rig.run()
    assert [(order["symbol"], order["status"]) for order in summary["opens"]] == [("AAAUSDT", STOP_FAILED_CLOSED)]
    assert summary["status"] == COMPLETED_WITH_ERRORS and rig.sent == []


def test_past_the_entry_deadline_a_failed_close_ends_the_day_with_errors(tmp_path: Path, log: list[str]) -> None:
    client = _client({"AAAUSDT": BREAKS, "BBBUSDT": FALLS})
    _hold_on_exchange(client, "BBBUSDT")
    client.fail["futures_create_order"] = [api_error(-4131, "PERCENT_PRICE filter limit")]
    rig = Rig(client, tmp_path, mode=TESTNET, max_entry_delay_minutes=0.0)
    summary = rig.run()

    assert summary["status"] == COMPLETED_WITH_ERRORS
    assert summary["closes"][0]["status"] == ERROR and "PERCENT_PRICE" in summary["closes"][0]["reason"]
    assert [(r["symbol"], r["status"]) for r in summary["opens"]] == [("AAAUSDT", CLOSE_FAILED)]
    assert list(client.positions) == ["BBBUSDT"] and len(client.algo_orders) == 1  # still has its stop
    assert rig.store.is_cycle_already_executed(TODAY)
    assert any(line.startswith(f"WARNING|Cycle {TODAY} finished (COMPLETED_WITH_ERRORS)") for line in log)


def test_a_position_that_vanished_before_its_close_does_not_block_entries(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS, "BBBUSDT": FALLS})
    _hold_on_exchange(client, "BBBUSDT")
    rig = Rig(client, tmp_path, mode=TESTNET)
    reconcile = rig.executor.get_open_positions

    def stop_fires_after_reading(fallback: Any = None) -> Any:
        positions = reconcile(fallback)
        del client.positions["BBBUSDT"]
        return positions

    rig.executor.get_open_positions = stop_fires_after_reading  # type: ignore[method-assign]
    summary = rig.run()
    assert summary["status"] == COMPLETED and summary["closes"][0]["status"] == "NO_POSITION"
    assert summary["opens"][0]["status"] == FILLED and client.sent("futures_create_order")[0]["side"] == "BUY"


def test_one_failed_entry_does_not_stop_the_others(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS, "BBBUSDT": BREAKS})
    client.fail["futures_create_order"] = [api_error(-2019, "Margin is insufficient.")]
    rig = Rig(client, tmp_path, mode=TESTNET)
    summary = rig.run()

    assert [(r["symbol"], r["status"]) for r in summary["opens"]] == [("AAAUSDT", ERROR), ("BBBUSDT", FILLED)]
    assert summary["status"] == COMPLETED_WITH_ERRORS and list(client.positions) == ["BBBUSDT"]
    assert rig.store.dispatched(TODAY)["AAAUSDT:OPEN_LONG"]["status"] == ERROR


def test_an_order_below_the_minimum_notional_is_reported_not_sent(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS}, wallet=300.0)
    client.symbols["AAAUSDT"] = symbol_info("AAAUSDT", step="0.001", min_qty="0.001", tick="0.01", min_notional="50")
    rig = Rig(client, tmp_path, mode=TESTNET)
    summary = rig.run()
    assert summary["opens"][0]["status"] == "REJECTED" and summary["opens"][0]["reason"] == "BELOW_MIN_NOTIONAL"
    assert summary["status"] == COMPLETED_WITH_ERRORS and client.sent("futures_create_order") == []


def test_a_position_found_without_stop_gets_one_back(tmp_path: Path, log: list[str]) -> None:
    client = _client({"AAAUSDT": QUIET, "BBBUSDT": QUIET})
    client.positions["AAAUSDT"] = [5.0, 100.0]  # e.g. the process died between the fill and its stop
    rig = Rig(client, tmp_path, mode=TESTNET)
    seen: list[Any] = []
    reconcile = rig.executor.get_open_positions
    rig.executor.get_open_positions = lambda fallback=None: seen.extend(reconcile(fallback)) or seen  # type: ignore[method-assign]
    summary = rig.run()

    assert summary["status"] == COMPLETED
    ((restored),) = summary["protected"]
    assert (restored["symbol"], restored["status"]) == ("AAAUSDT", STOP_RESTORED)
    (stop,) = client.algo_orders
    # Entry minus 2 ATR of a range whose ATR is about 2.9.
    assert stop["side"] == "SELL" and float(stop["quantity"]) == 5.0
    assert float(stop["triggerPrice"]) == pytest.approx(94.2, abs=0.6)
    assert stop["clientAlgoId"] == f"tb-{TODAY.replace('-', '')}-AAAUSDT-r"
    assert seen[0].risk_usd == pytest.approx(5.0 * (100.0 - float(stop["triggerPrice"])))
    assert rig.store.dispatched(TODAY)["AAAUSDT:PROTECT"]["status"] == STOP_RESTORED
    assert any("AAAUSDT: 5 units without a stop on the exchange" in line for line in log)

    rig.store = DailyStateStore(tmp_path / "other.json")  # a second pass finds the stop and leaves it alone
    assert rig.executor.protect_open_positions({"AAAUSDT": 5.8}, TODAY) == []


def test_a_position_below_its_restored_stop_level_is_closed(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": QUIET})
    client.positions["AAAUSDT"] = [5.0, 120.0]  # entry 120, price 100: 2 ATR below the entry is above the price
    rig = Rig(client, tmp_path, mode=TESTNET)
    summary = rig.run()
    ((closed),) = summary["protected"]
    assert (closed["symbol"], closed["status"]) == ("AAAUSDT", CLOSED_UNPROTECTED)
    assert client.positions == {} and client.algo_orders == []


# --- Paper account between cycles --------------------------------------------


def test_a_simulated_stop_is_applied_at_the_next_cycle(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": QUIET}), tmp_path)
    stop = rig.run()["opens"][0]["stop_price"]
    _next_day(rig.client, {"AAAUSDT": 90.0, "BBBUSDT": QUIET})  # the day's low (89) breaks through the stop (~98)
    summary = rig.run()

    assert summary["stopped"] == ["AAAUSDT"] and summary["positions_before"] == []
    assert rig.paper.positions == {} and rig.sent == [("AAAUSDT", OPEN_LONG)]
    (trade,) = rig.paper.trades
    assert trade["reason"] == "STOP" and trade["exit_price"] == pytest.approx(stop)
    assert trade["pnl"] < 0.0 and rig.paper.balance == pytest.approx(10_000.0 + trade["pnl"])


# --- Positions in symbols no longer in the basket -----------------------------


def test_a_position_outside_the_basket_is_closed_when_its_channel_breaks(tmp_path: Path, log: list[str]) -> None:
    rig = Rig(_client({"AAAUSDT": QUIET, "BBBUSDT": FALLS}), tmp_path, basket=["AAAUSDT"])
    rig.paper.open("BBBUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    summary = rig.run()

    assert rig.sent == [("BBBUSDT", CLOSE_LONG)] and summary["positions_before"] == ["BBBUSDT"]
    assert rig.paper.positions == {} and rig.paper.trades[0]["reason"] == "EXIT_CHANNEL_BROKEN"
    assert summary["symbols_with_data"] == 2
    assert any(line.startswith("WARNING|") and "outside the basket (BBBUSDT)" in line for line in log)


def test_a_position_outside_the_basket_is_closed_on_the_exchange(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": QUIET, "BBBUSDT": FALLS})
    _hold_on_exchange(client, "BBBUSDT")
    Rig(client, tmp_path, mode=TESTNET, basket=["AAAUSDT"]).run()

    (order,) = client.sent("futures_create_order")
    assert (order["symbol"], order["side"], order["reduceOnly"]) == ("BBBUSDT", "SELL", "true")


def test_a_position_outside_the_basket_keeps_its_simulated_stop(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": QUIET, "BBBUSDT": 90.0}), tmp_path, basket=["AAAUSDT"])
    rig.paper.open("BBBUSDT", 5.0, 100.0, 95.0, NOW_MS - DAY_MS)  # the day's low (89) breaks through the stop
    summary = rig.run()

    assert summary["stopped"] == ["BBBUSDT"] and summary["positions_before"] == []
    assert rig.paper.positions == {} and rig.sent == []
    assert summary["symbols_with_data"] == 1  # with no position left, it stops counting


def test_a_symbol_outside_the_basket_is_held_but_never_bought(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": QUIET, "BBBUSDT": BREAKS}), tmp_path, basket=["AAAUSDT"])
    rig.paper.open("BBBUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    assert rig.run()["positions_before"] == ["BBBUSDT"] and rig.sent == []  # breaks out upward: it is kept, not added to

    _next_day(rig.client, {"AAAUSDT": QUIET, "BBBUSDT": FALLS})
    rig.run()
    assert rig.sent == [("BBBUSDT", CLOSE_LONG)] and rig.paper.positions == {}

    # Once the position is closed, the symbol is no longer loaded: even if it breaks out, it is not bought.
    _next_day(rig.client, {"AAAUSDT": QUIET, "BBBUSDT": BREAKS})
    before = len(rig.client.sent("futures_klines"))
    summary = rig.run()
    assert rig.sent == [("BBBUSDT", CLOSE_LONG)] and summary["opens"] == []
    assert [call["symbol"] for call in rig.client.sent("futures_klines")[before:]] == ["AAAUSDT"]


def test_a_symbol_outside_the_basket_stopped_out_on_a_breakout_day_is_not_bought_back(tmp_path: Path) -> None:
    # That day the low hits the stop (103) and the close (104) is above the 20-candle high.
    def cycle(basket: list[str], root: Path) -> tuple[Rig, dict[str, Any]]:
        rig = Rig(_client({"AAAUSDT": QUIET, "BBBUSDT": BREAKS}), root, basket=basket)
        rig.paper.open("BBBUSDT", 5.0, 100.0, 103.0, NOW_MS - DAY_MS)
        return rig, rig.run()

    outside, summary = cycle(["AAAUSDT"], tmp_path / "outside")
    assert summary["stopped"] == ["BBBUSDT"] and outside.sent == [] and outside.paper.positions == {}
    # With BBB in the basket the same candle is a buy: being outside the basket is what prevents it.
    inside, summary = cycle(["AAAUSDT", "BBBUSDT"], tmp_path / "inside")
    assert summary["stopped"] == ["BBBUSDT"] and inside.sent == [("BBBUSDT", OPEN_LONG)]


def test_when_the_risk_runs_out_the_strongest_breakouts_win_wherever_they_are_in_the_basket(tmp_path: Path) -> None:
    # The basket goes from the weakest breakout to the strongest, and the open risk only fits two.
    client = _client({"AAAUSDT": 103.5, "BBBUSDT": 105.0, "CCCUSDT": 108.0})
    rig = Rig(client, tmp_path, risk_manager=PortfolioRiskManager(max_portfolio_heat_pct=1.2))
    summary = rig.run()

    assert rig.sent == [("CCCUSDT", OPEN_LONG), ("BBBUSDT", OPEN_LONG)]
    assert sorted(rig.paper.positions) == ["BBBUSDT", "CCCUSDT"]
    assert summary["holds"] == {"MAX_HEAT_EXCEEDED": 1}


def test_simulated_funding_is_charged_between_cycles(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    rig.run()
    balance = rig.paper.balance
    rig.client.funding["AAAUSDT"] = [
        {"fundingTime": NOW_MS + hours * 3_600_000, "fundingRate": "0.0001", "markPrice": "104.0"}
        for hours in (8, 16, 25)
    ]
    _next_day(rig.client, {"AAAUSDT": BREAKS})
    rig.run()
    # Two payments fall between one cycle and the next; the third comes later.
    assert rig.paper.balance == pytest.approx(balance - 2 * QUANTITY * 104.0 * 0.0001)


# --- Delay -------------------------------------------------------------------


def test_a_late_cycle_closes_but_does_not_buy(tmp_path: Path, log: list[str]) -> None:
    client = _client({"AAAUSDT": BREAKS, "BBBUSDT": FALLS})
    rig = Rig(client, tmp_path, max_entry_delay_minutes=360.0)
    rig.paper.open("BBBUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    client.now_ms += 7 * 3_600_000
    summary = rig.run()

    assert summary["late_minutes"] == pytest.approx(420.0)
    assert [(r["symbol"], r["status"]) for r in summary["opens"]] == [("AAAUSDT", TOO_LATE)]
    assert summary["closes"][0]["status"] == SIMULATED and rig.paper.positions == {}
    assert summary["status"] == COMPLETED_WITH_ERRORS
    assert any("starting 420 minutes late" in line for line in log)
    assert any("over the maximum of 360" in line for line in log)


def test_a_cycle_within_the_allowed_delay_still_buys(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS})
    rig = Rig(client, tmp_path, max_entry_delay_minutes=360.0)
    client.now_ms += 5 * 3_600_000
    assert rig.run()["opens"][0]["status"] == SIMULATED


def test_a_late_cycle_with_nothing_to_buy_is_a_clean_cycle(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": QUIET})
    rig = Rig(client, tmp_path)
    client.now_ms += 20 * 3_600_000
    assert rig.run()["status"] == COMPLETED


# --- Scheduling --------------------------------------------------------------


def test_the_cycle_is_scheduled_ninety_seconds_after_midnight_utc(tmp_path: Path) -> None:
    runner = Rig(_client({"AAAUSDT": QUIET}), tmp_path).runner
    assert runner.scheduled_for(date(2026, 10, 9)) == datetime(2026, 10, 9, 0, 1, 30, tzinfo=timezone.utc)
    before = datetime(2026, 10, 9, 0, 1, 29, tzinfo=timezone.utc)
    assert runner.next_run_at(before) == datetime(2026, 10, 9, 0, 1, 30, tzinfo=timezone.utc)
    for now in (datetime(2026, 10, 9, 0, 1, 30, tzinfo=timezone.utc), datetime(2026, 10, 9, 23, 59, tzinfo=timezone.utc)):
        assert runner.next_run_at(now) == datetime(2026, 10, 10, 0, 1, 30, tzinfo=timezone.utc)


def test_the_delay_after_midnight_is_configurable(tmp_path: Path) -> None:
    runner = Rig(_client({"AAAUSDT": QUIET}), tmp_path, run_delay_seconds=600.0).runner
    assert runner.scheduled_for(date(2026, 10, 9)) == datetime(2026, 10, 9, 0, 10, tzinfo=timezone.utc)


class Stop:
    """Stand-in for ``threading.Event``: each wait is recorded and moves the clock forward."""

    def __init__(self, client: FakeFuturesClient, rounds: int) -> None:
        self.client, self.rounds, self.waits = client, rounds, []

    def is_set(self) -> bool:
        return self.rounds <= 0

    def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        self.rounds -= 1
        self.client.now_ms += int(seconds * 1000)
        return self.rounds <= 0


def test_run_forever_executes_the_pending_cycle_and_sleeps_until_tomorrow(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    stop = Stop(rig.client, rounds=1)
    rig.runner.run_forever(stop)  # type: ignore[arg-type]
    assert rig.store.is_cycle_already_executed(TODAY) and rig.sent == [("AAAUSDT", OPEN_LONG)]
    assert stop.waits == [pytest.approx(86_400.0)]


def test_run_forever_runs_one_cycle_per_day(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": QUIET}), tmp_path)
    stop = Stop(rig.client, rounds=1)
    rig.runner.run_forever(stop)  # type: ignore[arg-type]
    _prices(rig.client, {"AAAUSDT": BREAKS})  # the clock is already on the next day
    stop.rounds = 1
    rig.runner.run_forever(stop)  # type: ignore[arg-type]
    assert rig.store.is_cycle_already_executed(TOMORROW) and rig.sent == [("AAAUSDT", OPEN_LONG)]


def test_run_forever_waits_for_the_scheduled_time(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS})
    client.now_ms -= 60_000  # 00:00:30, one minute before the time
    rig = Rig(client, tmp_path)
    stop = Stop(client, rounds=1)
    rig.runner.run_forever(stop)  # type: ignore[arg-type]
    assert rig.sent == [] and stop.waits == [pytest.approx(60.0)]


def test_run_forever_retries_while_the_exchange_is_down(tmp_path: Path, log: list[str]) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    rig.client.fail["futures_exchange_info"] = [network_error() for _ in range(4)]
    stop = Stop(rig.client, rounds=2)
    rig.runner.run_forever(stop, retry_seconds=300.0)  # type: ignore[arg-type]
    assert stop.waits[0] == 300.0 and stop.waits[1] == pytest.approx(86_400.0 - 300.0)
    assert rig.store.is_cycle_already_executed(TODAY)
    assert any(line.startswith("WARNING|The cycle could not run") for line in log)


def test_run_forever_does_not_repeat_an_executed_day(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path)
    rig.run()
    calls = len(rig.client.calls)
    rig.runner.run_forever(Stop(rig.client, rounds=1))  # type: ignore[arg-type]
    assert len(rig.client.calls) == calls


def test_run_forever_stops_on_a_problem_that_waiting_cannot_fix(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS})
    client.dual_side = True
    rig = Rig(client, tmp_path, mode=TESTNET)
    stop = Stop(client, rounds=5)
    with pytest.raises(BrokerUnavailableError, match="hedge mode"):
        rig.runner.run_forever(stop)  # type: ignore[arg-type]
    assert stop.waits == [] and rig.sent == []


# --- Alerts --------------------------------------------------------------------


class Alerts:
    """A fake notifier: it keeps what it is asked to send."""

    def __init__(self, broken: bool = False) -> None:
        self.sent: list[tuple[str, str | None]] = []
        self.broken = broken

    def send(self, text: str, *, once_key: str | None = None) -> bool:
        if self.broken:
            raise RuntimeError("Telegram does not answer")
        if once_key is not None and any(key == once_key for _, key in self.sent):
            return False
        self.sent.append((text, once_key))
        return True


def test_a_cycle_that_buys_sends_its_alert(tmp_path: Path) -> None:
    alerts = Alerts()
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": QUIET}), tmp_path, notifier=alerts)
    rig.run()

    ((text, key),) = alerts.sent
    lines = text.splitlines()
    assert key is None and lines[0] == f"Softronic QuantBot · DRY_RUN · cycle {TODAY}"
    assert lines[1] == "BUYS" and lines[2].startswith("• AAA: 9.615 at 104.0000 · stop ")
    # The account is the one after buying: balance minus the fee, one position.
    assert lines[-5].startswith("Account: 9,999.50 USDT") and lines[-3] == "Positions (1):"
    # Each position with where it will be sold, and the end-of-cycle stop check.
    assert lines[-2].startswith("• AAA: entry 104.0000 · stop 98.1900 · exit channel ")
    assert lines[-1] == "Stops: OK (1 of 1 positions protected)"


def test_the_alert_tells_sales_and_stops(tmp_path: Path) -> None:
    alerts = Alerts()
    rig = Rig(_client({"AAAUSDT": BREAKS, "BBBUSDT": FALLS}), tmp_path, notifier=alerts)
    rig.paper.open("BBBUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    rig.run()
    _next_day(rig.client, {"AAAUSDT": 90.0, "BBBUSDT": QUIET})  # the day's low breaks through the stop of AAA
    rig.run()

    first, second = (text for text, _ in alerts.sent)
    assert "SELLS\n• BBB: at 96.0000 on exit channel · −20.49 USDT" in first and "BUYS\n• AAA" in first
    assert "STOPS\n• AAA: at " in second and "Positions (0): none" in second


def test_a_quiet_cycle_sends_its_line_only_if_asked(tmp_path: Path) -> None:
    alerts = Alerts()
    Rig(_client({"AAAUSDT": QUIET}), tmp_path / "with", notifier=alerts).run()
    assert "No changes: no buys or sells." in alerts.sent[0][0]

    silent = Alerts()
    rig = Rig(_client({"AAAUSDT": QUIET}), tmp_path / "without", notifier=silent, notify_quiet_days=False)
    rig.run()
    assert silent.sent == []
    _next_day(rig.client, {"AAAUSDT": BREAKS})
    rig.run()
    assert len(silent.sent) == 1 and "BUYS" in silent.sent[0][0]  # with news it always alerts


def test_an_alert_that_fails_does_not_spoil_the_cycle(tmp_path: Path, log: list[str]) -> None:
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path, notifier=Alerts(broken=True))
    assert rig.run()["status"] == COMPLETED and rig.store.is_cycle_already_executed(TODAY)
    assert any(line.startswith("ERROR|") and "alert not sent" in line for line in log)


def test_a_cycle_that_cannot_run_alerts_once_and_then_reports_normally(tmp_path: Path) -> None:
    alerts = Alerts()
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path, notifier=alerts)
    rig.client.fail["futures_exchange_info"] = [network_error() for _ in range(8)]  # two failed attempts
    rig.runner.run_forever(Stop(rig.client, rounds=3), retry_seconds=300.0)  # type: ignore[arg-type]

    (failure, key), (normal, _) = alerts.sent
    assert key == f"DRY_RUN:{TODAY}:retry"
    assert "ATTENTION: the cycle could not run." in failure and "Retrying every 5 min" in failure
    assert "BUYS" in normal and rig.store.is_cycle_already_executed(TODAY)


def test_a_cycle_without_data_alerts_too(tmp_path: Path) -> None:
    alerts = Alerts()
    rig = Rig(_client({"AAAUSDT": BREAKS}), tmp_path, notifier=alerts)
    rig.client.klines["AAAUSDT"] = []
    rig.runner.run_forever(Stop(rig.client, rounds=2), retry_seconds=300.0)  # type: ignore[arg-type]
    ((text, key),) = alerts.sent
    assert "No symbol has valid daily candles." in text and key == f"DRY_RUN:{TODAY}:retry"


def test_a_problem_that_stops_the_process_alerts_before_stopping(tmp_path: Path) -> None:
    client = _client({"AAAUSDT": BREAKS})
    client.dual_side = True
    alerts = Alerts()
    rig = Rig(client, tmp_path, mode=TESTNET, notifier=alerts)
    with pytest.raises(BrokerUnavailableError):
        rig.runner.run_forever(Stop(client, rounds=5))  # type: ignore[arg-type]
    ((text, key),) = alerts.sent
    assert "hedge mode" in text and "The process has stopped" in text and key == f"TESTNET:{TODAY}:halt"


def test_without_report_and_without_alerts_the_account_is_not_read_again(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": QUIET}), tmp_path)
    read: list[int] = []
    rig.executor.get_account_and_positions = lambda: read.append(1)  # type: ignore[method-assign,assignment,return-value]
    assert rig.run()["status"] == COMPLETED and read == []


def test_build_runner_turns_the_alerts_on_only_with_token_and_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    client = _client({"AAAUSDT": QUIET})
    monkeypatch.setattr("src.execution.binance_futures_executor.Client", lambda *args, **kwargs: client)
    runner, lock = build_runner(_settings(tmp_path / "without", futures={"symbols": "AAAUSDT", "candles": 150}))
    lock.release()
    assert runner._notifier is None and any("Telegram alerts: disabled" in line for line in log)

    telegram = {"bot_token": "123:bot-secret", "chat_id": "555", "quiet_days": False}
    runner, lock = build_runner(
        _settings(tmp_path / "with", futures={"symbols": "AAAUSDT", "candles": 150}, telegram=telegram))
    lock.release()
    assert isinstance(runner._notifier, TelegramNotifier) and runner._notify_quiet_days is False
    assert any("Telegram alerts: enabled" in line for line in log) and not any("bot-secret" in line for line in log)


# --- Configuration and startup -----------------------------------------------


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(_env_file=None, state_dir=tmp_path, **overrides)


def test_futures_settings_are_safe_by_default(tmp_path: Path) -> None:
    futures = _settings(tmp_path).futures
    assert futures.execution_mode == "DRY_RUN" and futures.live_trading_acknowledged is False
    # By default, the twelve measured coins.
    assert futures.symbol_list == DEFAULT_FUTURES_BASKET.split(",") == MEASURED_FUTURES_BASKET.split(",")
    assert len(set(futures.symbol_list)) == 12
    # The 37 stress-test contracts, kept apart and with none of the twelve.
    stress = STRESS_TEST_FUTURES_BASKET.split(",")
    assert len(set(stress)) == 37 and not set(stress) & set(futures.symbol_list)
    assert {"1000PEPEUSDT", "RENDERUSDT", "AUSDT", "SUSDT"} <= set(stress)
    # The names the exchange does not list as a contract: they would be skipped every day.
    assert not {"PEPEUSDT", "SHIBUSDT", "FLOKIUSDT", "BONKUSDT", "RNDRUSDT", "EOSUSDT", "FTMUSDT", "MKRUSDT"} & set(stress)
    wide = _settings(tmp_path, futures={"symbols": MEASURED_FUTURES_BASKET + "," + STRESS_TEST_FUTURES_BASKET}).futures
    assert len(wide.symbol_list) == 49
    assert (futures.candles, futures.leverage, futures.run_delay_seconds) == (500, 1, 90.0)
    assert (futures.risk_pct, futures.max_position_fraction, futures.max_heat_pct) == (1.0, 0.1, 4.0)
    assert not futures.has_credentials


def test_futures_settings_come_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_FUTURES__EXECUTION_MODE", "TESTNET")
    monkeypatch.setenv("BOT_FUTURES__SYMBOLS", " btcusdt, ethusdt ,")
    monkeypatch.setenv("BOT_FUTURES__API_KEY", "the-key")
    monkeypatch.setenv("BOT_FUTURES__API_SECRET", "the-secret")
    futures = _settings(tmp_path).futures
    assert futures.execution_mode == "TESTNET" and futures.symbol_list == ["BTCUSDT", "ETHUSDT"]
    assert futures.has_credentials and "the-secret" not in repr(futures)


@pytest.mark.parametrize("symbols", ["", " , ", "BTCUSDT,BTCUSDT", "BTCUSD", "USDT", "BTC-USDT"])
def test_invalid_baskets_are_rejected(tmp_path: Path, symbols: str) -> None:
    with pytest.raises(ValidationError):
        _settings(tmp_path, futures={"symbols": symbols})


@pytest.mark.parametrize(
    "field", [{"execution_mode": "REAL"}, {"leverage": 0}, {"leverage": 20}, {"candles": 30}, {"paper_fee_rate": -0.1}]
)
def test_invalid_futures_settings_are_rejected(tmp_path: Path, field: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _settings(tmp_path, futures=field)


def test_live_needs_dry_run_off_as_well(tmp_path: Path, log: list[str]) -> None:
    live = {"execution_mode": "LIVE", "live_trading_acknowledged": True}
    assert effective_mode(_settings(tmp_path, futures=live)) == DRY_RUN
    assert any("BOT_DRY_RUN=true" in line for line in log)
    real = _settings(tmp_path, futures=live, dry_run=False, broker_provider="mt5", symbol="XAUUSD", use_ml_filter=False)
    assert effective_mode(real) == LIVE
    assert effective_mode(_settings(tmp_path, futures={"execution_mode": "TESTNET"})) == TESTNET


def test_build_runner_wires_a_dry_run_cycle_and_locks_its_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client({"AAAUSDT": QUIET, "BBBUSDT": QUIET})
    monkeypatch.setattr("src.execution.binance_futures_executor.Client", lambda *args, **kwargs: client)
    settings = _settings(tmp_path, futures={"symbols": "AAAUSDT,BBBUSDT", "candles": 150}, paper_capital=5_000.0)

    runner, lock = build_runner(settings)
    try:
        assert runner.run_daily_cycle()["status"] == COMPLETED
        assert (tmp_path / "runtime_state_dry_run.json").exists() and (tmp_path / "paper_futures.json").exists()
        with pytest.raises(StateInUseError, match="Another daily cycle"):
            build_runner(settings)
    finally:
        lock.release()


def test_build_runner_takes_the_risk_of_the_basket_from_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client({"AAAUSDT": QUIET})
    monkeypatch.setattr("src.execution.binance_futures_executor.Client", lambda *args, **kwargs: client)
    risk = {"risk_pct": 2.0, "max_position_fraction": 0.2, "max_heat_pct": 8.0}
    runner, lock = build_runner(_settings(tmp_path, futures={"symbols": "AAAUSDT", "candles": 150, **risk}))
    try:
        assert (runner._strategy.risk_pct, runner._strategy.max_leverage) == (2.0, 0.2)
        assert runner._risk_manager.max_portfolio_heat_pct == 8.0
    finally:
        lock.release()
    with pytest.raises(ValidationError):
        _settings(tmp_path, futures={"risk_pct": 0.0})


def test_build_runner_with_live_but_no_acknowledgement_stays_simulated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client({"AAAUSDT": BREAKS})
    monkeypatch.setattr("src.execution.binance_futures_executor.Client", lambda *args, **kwargs: client)
    settings = _settings(
        tmp_path, dry_run=False, broker_provider="mt5", symbol="XAUUSD", use_ml_filter=False,
        futures={"symbols": "AAAUSDT", "candles": 150, "execution_mode": "LIVE", "api_key": "k", "api_secret": "s"},
    )
    runner, lock = build_runner(settings)
    try:
        summary = runner.run_daily_cycle()
        assert summary["mode"] == DRY_RUN and client.sent("futures_create_order") == []
        assert (tmp_path / "runtime_state_dry_run.json").exists()
    finally:
        lock.release()


def test_main_with_panel_serves_it_from_the_same_process_and_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client({"AAAUSDT": QUIET})
    events: list[str] = []

    class Panel:
        def shutdown(self) -> None:
            events.append("dashboard stopped")

        def server_close(self) -> None:
            events.append("dashboard closed")

    def serve(settings: Settings, account: str) -> Panel:
        events.append(f"dashboard for {account}")
        return Panel()

    settings = _settings(tmp_path, futures={"symbols": "AAAUSDT", "candles": 150})
    monkeypatch.setattr("src.execution.binance_futures_executor.Client", lambda *args, **kwargs: client)
    monkeypatch.setattr("src.ui.dashboard.serve_in_background", serve)
    monkeypatch.setattr(daily_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(daily_runner, "configure_logging", lambda _: None)
    monkeypatch.setattr(daily_runner.signal, "signal", lambda *_: None)
    monkeypatch.setattr(DailyRunner, "run_forever", lambda self, stop: events.append("cycle running"))

    assert daily_runner.main(["--panel"]) == 0
    assert events == ["dashboard for dry_run", "cycle running", "dashboard stopped", "dashboard closed"]
    # The lock was released: it can be started again.
    events.clear()
    assert daily_runner.main([]) == 0 and events == ["cycle running"]


def test_main_once_does_not_start_the_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client({"AAAUSDT": QUIET})
    settings = _settings(tmp_path, futures={"symbols": "AAAUSDT", "candles": 150})
    monkeypatch.setattr("src.execution.binance_futures_executor.Client", lambda *args, **kwargs: client)
    monkeypatch.setattr("src.ui.dashboard.serve_in_background", lambda *_: pytest.fail("the dashboard must not start"))
    monkeypatch.setattr(daily_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(daily_runner, "configure_logging", lambda _: None)
    assert daily_runner.main(["--once", "--panel"]) == 0
    assert (tmp_path / "reports" / "dry_run").is_dir()


def test_main_reports_a_configuration_problem_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    settings = _settings(tmp_path, futures={"execution_mode": "TESTNET"})
    monkeypatch.setattr(daily_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(daily_runner, "configure_logging", lambda _: None)
    assert daily_runner.main(["--once"]) == 1
    assert any(line.startswith("ERROR|TESTNET mode requires BOT_FUTURES__API_KEY") for line in log)


def test_the_entry_time_of_a_position_comes_from_the_paper_account_or_the_cycle_that_bought_it(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": QUIET}), tmp_path)
    rig.paper.open("AAAUSDT", 5.0, 100.0, 80.0, NOW_MS - DAY_MS)
    held = [FuturesPosition("AAAUSDT", 5.0, 100.0, 100.0, 80.0)]
    assert rig.runner._open_times(held) == {"AAAUSDT": NOW_MS - DAY_MS}

    exchange = Rig(_client({"AAAUSDT": QUIET}), tmp_path / "x", mode=TESTNET)
    exchange.store.record_dispatch("2026-10-05", "AAAUSDT", OPEN_LONG,
                                   {"symbol": "AAAUSDT", "action": OPEN_LONG, "status": FILLED})
    expected = int(exchange.runner.scheduled_for(date(2026, 10, 5)).timestamp() * 1000)
    assert exchange.runner._open_times(held) == {"AAAUSDT": expected}
    assert exchange.runner._open_times([FuturesPosition("ZZZUSDT", 1.0, 1.0, 1.0, None)]) == {}
