"""Tests of the pre-trade slippage collar, the end-of-cycle stop check and the optional post-cycle work."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config.settings import Settings
from src.engine import post_cycle
from src.engine.daily_reporter import render_markdown, stop_check_text
from src.engine.daily_runner import COMPLETED, _auditor
from src.engine.orchestrator import MARKET, OPEN_LONG, OrderIntent
from src.engine.post_cycle import LlmAuditor, ping_healthcheck
from src.execution.binance_futures_executor import (
    DRY_RUN,
    FILLED,
    REJECTED,
    SIMULATED,
    SLIPPAGE_COLLAR,
    STOP_RESTORED,
    TESTNET,
    BinanceFuturesExecutor,
    PaperFuturesAccount,
)
from tests.fake_futures import FakeFuturesClient, symbol_info
from tests.test_daily_reporter import _reporter, _snapshot
from tests.test_daily_runner import QUIET, Rig, _client, log  # noqa: F401 (log is a fixture)

CYCLE = "2026-10-09"


def _executor(mode: str, tmp_path: Path, mark: float, **options: Any) -> tuple[BinanceFuturesExecutor, FakeFuturesClient]:
    client = FakeFuturesClient({"AAAUSDT": symbol_info("AAAUSDT", step="0.001", min_qty="0.001", tick="0.01")})
    client.marks["AAAUSDT"] = mark
    executor = BinanceFuturesExecutor(
        mode=mode, client=client, sleep=lambda _: None, clock=lambda: client.now_ms / 1000,
        paper=PaperFuturesAccount(tmp_path / "paper.json", 10_000.0), max_slippage_pct=1.5, **options,
    )
    return executor, client


def _entry(signal: float = 100.0) -> OrderIntent:
    return OrderIntent("AAAUSDT", OPEN_LONG, 5.0, MARKET, 90.0, "BREAKOUT", signal_price=signal)


# --- Slippage collar ----------------------------------------------------------


def test_an_entry_too_far_from_its_signal_close_is_not_sent_on_the_exchange(tmp_path: Path) -> None:
    executor, client = _executor(TESTNET, tmp_path, mark=101.6)  # +1.6 % over the close that gave the signal
    result = executor.execute_order_intent(_entry(), CYCLE)
    assert (result["status"], result["reason"]) == (REJECTED, SLIPPAGE_COLLAR)
    assert client.sent("futures_create_order") == []


def test_an_entry_within_the_collar_goes_out_and_records_its_slippage(tmp_path: Path) -> None:
    executor, client = _executor(TESTNET, tmp_path, mark=101.0)
    result = executor.execute_order_intent(_entry(), CYCLE)
    assert result["status"] == FILLED and result["slippage_pct"] == pytest.approx(1.0)
    assert result["signal_price"] == 100.0 and len(client.sent("futures_create_order")) == 1


def test_dry_run_only_records_what_the_collar_would_have_done(tmp_path: Path, log: list[str]) -> None:
    executor, _ = _executor(DRY_RUN, tmp_path, mark=102.0)
    result = executor.execute_order_intent(_entry(), CYCLE)
    # The paper account keeps trading as it always has: the collar is a TESTNET/LIVE guard.
    assert result["status"] == SIMULATED and result["collar_exceeded"] is True
    assert result["slippage_pct"] == pytest.approx(2.0)
    assert any("AAAUSDT: price 102" in line and "+2.00 %" in line for line in log)


def test_without_a_signal_close_there_is_no_collar(tmp_path: Path) -> None:
    executor, _ = _executor(TESTNET, tmp_path, mark=150.0)
    intent = OrderIntent("AAAUSDT", OPEN_LONG, 5.0, MARKET, 90.0, "BREAKOUT")
    assert executor.execute_order_intent(intent, CYCLE)["status"] == FILLED


def test_the_signal_close_does_not_change_which_order_it_is() -> None:
    assert _entry(100.0) == _entry(103.0) == OrderIntent("AAAUSDT", OPEN_LONG, 5.0, MARKET, 90.0, "BREAKOUT")


def test_the_report_measures_entry_slippage_in_atr_and_flags_it(tmp_path: Path) -> None:
    cycle = _snapshot(results=[{"symbol": "BBBUSDT", "action": OPEN_LONG, "status": "SIMULATED", "quantity": 9.615,
                                "price": 104.52, "collar_exceeded": True}])
    (note,) = _reporter(tmp_path).build(cycle)["observations"]["execution"]
    row_atr = next(row for row in _reporter(tmp_path).build(cycle)["basket"] if row["symbol"] == "BBBUSDT")["atr"]
    assert note["slippage_pct"] == pytest.approx(0.5) and note["slippage_warning"] is True
    assert note["slippage_atr"] == pytest.approx(0.52 / row_atr) and note["collar_exceeded"] is True
    markdown = render_markdown(_reporter(tmp_path).build(cycle))
    assert "WARNING: slippage above 0.3 %" in markdown and "would not have been sent" in markdown


# --- Stops at the end of the cycle -------------------------------------------


def test_every_simulated_position_ends_the_cycle_with_its_stop(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": QUIET}), tmp_path)
    rig.paper.open("AAAUSDT", 5.0, 100.0, 80.0, 0)
    summary = rig.run()
    assert summary["stop_check"] == {"positions": 1, "protected": 1, "repaired": [], "uncovered": [],
                                     "unprotected": []}


def test_a_stop_missing_at_the_end_of_the_cycle_is_put_back_and_reported(tmp_path: Path) -> None:
    rig = Rig(_client({"AAAUSDT": QUIET}), tmp_path, mode=TESTNET)
    rig.client.positions["AAAUSDT"] = [5.0, 100.0]
    verify = rig.executor.verify_stops
    protect = rig.executor.protect_open_positions
    rig.executor.protect_open_positions = lambda distances, cycle: []  # the start-of-cycle repair misses it
    rig.executor.verify_stops = lambda distances, cycle: (  # type: ignore[method-assign]
        setattr(rig.executor, "protect_open_positions", protect) or verify(distances, cycle))
    check = rig.run()["stop_check"]
    assert check["protected"] == check["positions"] == 1 and check["unprotected"] == []
    assert [item["status"] for item in check["repaired"]] == [STOP_RESTORED]
    assert stop_check_text(check) == "OK (1 of 1 positions protected); repaired: AAAUSDT STOP_RESTORED"


def test_a_stop_smaller_than_the_position_is_flagged(tmp_path: Path) -> None:
    executor, client = _executor(TESTNET, tmp_path, mark=100.0)
    client.positions["AAAUSDT"] = [5.0, 100.0]
    client.algo_orders.append({"algoId": 1, "orderType": "STOP_MARKET", "symbol": "AAAUSDT", "side": "SELL",
                               "quantity": "3", "triggerPrice": "90"})
    check = executor.verify_stops({}, CYCLE)
    assert check["uncovered"] == ["AAAUSDT"] and check["protected"] == 1
    assert stop_check_text(check).startswith("ATTENTION: AAAUSDT with a partial stop")
    # A stop that closes the whole position says no quantity: it covers everything.
    client.algo_orders[0].pop("quantity")
    assert executor.verify_stops({}, CYCLE)["uncovered"] == []


def test_the_stop_check_is_in_the_report_summary(tmp_path: Path) -> None:
    check = {"positions": 2, "protected": 2, "repaired": [], "uncovered": [], "unprotected": []}
    report = _reporter(tmp_path).build(_snapshot(stop_check=check))
    assert "| Stops checked / reconciled | OK (2 of 2 positions protected) |" in render_markdown(report)
    assert "| Stops checked / reconciled | not checked |" in render_markdown(
        _reporter(tmp_path).build(_snapshot()))


# --- Healthcheck ping ----------------------------------------------------------


class _Answer:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *_: object) -> None: ...


def test_the_ping_is_a_get_and_never_raises(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> None:
    asked: list[str] = []
    monkeypatch.setattr(post_cycle.urllib.request, "urlopen", lambda url, timeout: asked.append(url) or _Answer())
    assert ping_healthcheck("https://hc-ping.example/secret-token") is True and asked == ["https://hc-ping.example/secret-token"]

    def down(url: str, timeout: float) -> Any:
        raise TimeoutError

    monkeypatch.setattr(post_cycle.urllib.request, "urlopen", down)
    assert ping_healthcheck("https://hc-ping.example/secret-token") is False
    assert ping_healthcheck(None) is False and ping_healthcheck("file:///etc/passwd") is False
    assert not any("secret-token" in line for line in log)


def test_the_cycle_pings_only_when_it_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pinged: list[str | None] = []
    monkeypatch.setattr("src.engine.daily_runner.ping_healthcheck", pinged.append)
    rig = Rig(_client({"AAAUSDT": QUIET}), tmp_path, healthcheck_url="https://hc.example/x")
    assert rig.run()["status"] == COMPLETED and pinged == ["https://hc.example/x"]
    rig.run()  # already done today: no new cycle, no new ping
    assert pinged == ["https://hc.example/x"]


# --- LLM auditor ---------------------------------------------------------------


class _Claude:
    """A stand-in for the Anthropic client: answers each block, or refuses, or fails."""

    def __init__(self, behaviour: str = "answer") -> None:
        self.behaviour = behaviour
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.behaviour == "fail":
            raise RuntimeError("network down")
        if self.behaviour == "refuse":
            return SimpleNamespace(stop_reason="refusal", content=[], model=request["model"])
        text = SimpleNamespace(type="text", text=f"Answer {len(self.requests)}")
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="thinking"), text],
                               model=request["model"])


def _written_report(tmp_path: Path) -> tuple[Path, Path]:
    return _reporter(tmp_path).write(_snapshot())


def test_the_auditor_answers_the_three_blocks_and_appends_section_six(tmp_path: Path) -> None:
    markdown_path, json_path = _written_report(tmp_path)
    claude = _Claude()
    assert LlmAuditor(api_key=None, model="claude-opus-5-5", effort="medium", client=claude).audit(
        json_path, markdown_path) is True

    assert len(claude.requests) == 3
    request = claude.requests[0]
    assert request["model"] == "claude-opus-5-5" and request["output_config"] == {"effort": "medium"}
    assert request["fallbacks"] == "default" and request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["messages"][0]["content"].startswith("Cycle ")
    report = json.loads(json_path.read_text())
    assert [answer["answer"] for answer in report["llm_audit"]["answers"]] == [
        "Answer 1", "Answer 2", "Answer 3"]
    markdown = markdown_path.read_text()
    assert "## 6. Automatic review by the LLM auditor" in markdown and "Answer 3" in markdown
    assert render_markdown(report) == markdown


def test_a_refused_block_is_said_and_a_failure_leaves_the_report_alone(tmp_path: Path) -> None:
    markdown_path, json_path = _written_report(tmp_path)
    auditor = LlmAuditor(api_key=None, model="m", effort="low", client=_Claude("refuse"))
    assert auditor.audit(json_path, markdown_path) is True
    assert "The model declined to answer this block." in markdown_path.read_text()

    markdown_path, json_path = _written_report(tmp_path / "other")
    before = markdown_path.read_text()
    assert LlmAuditor(api_key=None, model="m", effort="low", client=_Claude("fail")).audit(
        json_path, markdown_path) is False
    assert markdown_path.read_text() == before


def test_the_auditor_is_off_by_default_and_never_costs_tokens_unless_enabled() -> None:
    settings = Settings(_env_file=None)
    assert settings.audit_llm.enabled is False and _auditor(settings) is None
    assert settings.healthcheck_ping_url is None
    enabled = Settings(_env_file=None, audit_llm={"enabled": True, "api_key": "sk-test"})
    assert isinstance(_auditor(enabled), LlmAuditor)


def test_the_runner_hands_the_written_report_to_the_auditor(tmp_path: Path) -> None:
    from src.engine.daily_reporter import DailyReporter
    from src.risk.portfolio_risk import PortfolioRiskManager
    from src.strategies.trend_breakout import TrendBreakoutStrategy

    claude = _Claude()
    rig = Rig(_client({"AAAUSDT": QUIET}), tmp_path,
              reporter=DailyReporter(tmp_path / "reports", TrendBreakoutStrategy(), PortfolioRiskManager()),
              auditor=LlmAuditor(api_key=None, model="m", effort="low", client=claude))
    rig.run()
    assert len(claude.requests) == 3
    assert "## 6. Automatic review" in next((tmp_path / "reports").glob("*.md")).read_text()
