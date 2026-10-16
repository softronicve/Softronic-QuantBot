"""Tests of the Telegram alerts of the daily cycle (``src/engine/notifier.py``)."""

from __future__ import annotations

import urllib.error
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from config.settings import Settings
from src.engine import notifier
from src.engine.notifier import TelegramNotifier, build_notifier, cycle_message, failure_message
from src.execution.binance_futures_executor import FuturesAccount, FuturesPosition

TOKEN = "123456:the-bot-secret"


class Telegram:
    """A fake Telegram: it keeps what it is asked for and answers what it is told to."""

    def __init__(self, answer: Any = None, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict[str, str], float]] = []
        self.answer = {"ok": True} if answer is None else answer
        self.error = error

    def __call__(self, url: str, fields: Mapping[str, str], timeout: float) -> Any:
        self.calls.append((url, dict(fields), timeout))
        if self.error is not None:
            raise self.error
        return self.answer


@pytest.fixture
def log() -> Any:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}")
    yield lines
    logger.remove(sink)


def _notifier(telegram: Telegram, memory: Path | None = None) -> TelegramNotifier:
    return TelegramNotifier(TOKEN, "-100200300", memory=memory, transport=telegram)


# --- Sending -----------------------------------------------------------------


def test_a_message_goes_to_the_configured_chat() -> None:
    telegram = Telegram()
    assert _notifier(telegram).send("hello") is True
    (url, fields, timeout), = telegram.calls
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendMessage" and timeout == 5.0
    assert fields == {"chat_id": "-100200300", "text": "hello", "disable_web_page_preview": "true"}


def test_a_long_message_is_cut_to_what_telegram_accepts() -> None:
    telegram = Telegram()
    _notifier(telegram).send("x" * 5000)
    assert len(telegram.calls[0][1]["text"]) == 4096


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError(f"https://api.telegram.org/bot{TOKEN}/sendMessage", 401, "Unauthorized", None, None),  # type: ignore[arg-type]
    urllib.error.URLError(TimeoutError("timed out")),
    TimeoutError("timed out"),
    RuntimeError(f"something odd with https://api.telegram.org/bot{TOKEN}/sendMessage"),
])
def test_a_failed_send_never_raises_and_never_logs_the_token(error: Exception, log: list[str]) -> None:
    assert _notifier(Telegram(error=error)).send("hello") is False
    assert len(log) == 1 and log[0].startswith("WARNING|Telegram alert not sent")
    assert TOKEN not in log[0] and "secret" not in log[0]


def test_a_refusal_from_telegram_is_reported(log: list[str]) -> None:
    telegram = Telegram(answer={"ok": False, "description": "Bad Request: chat not found"})
    assert _notifier(telegram).send("hello") is False
    assert log == ["WARNING|Telegram alert rejected: Bad Request: chat not found\n"]
    assert _notifier(Telegram(answer=[])).send("hello") is False


@pytest.mark.parametrize(("token", "chat"), [("", "1"), (TOKEN, "")])
def test_token_and_chat_are_required(token: str, chat: str) -> None:
    with pytest.raises(ValueError, match="bot token and the chat_id"):
        TelegramNotifier(token, chat)


def test_a_one_time_alert_is_not_repeated_even_after_a_restart(tmp_path: Path) -> None:
    telegram, memory = Telegram(), tmp_path / "state" / "telegram_sent.json"
    first = _notifier(telegram, memory)
    assert first.send("the cycle fails", once_key="DRY_RUN:2026-10-13:retry") is True
    assert first.send("the cycle fails", once_key="DRY_RUN:2026-10-13:retry") is False
    # Another process with the same file does not repeat it either; another key does go out.
    restarted = _notifier(telegram, memory)
    assert restarted.send("the cycle fails", once_key="DRY_RUN:2026-10-13:retry") is False
    assert restarted.send("the cycle fails", once_key="DRY_RUN:2026-10-14:retry") is True
    assert restarted.send("a normal alert") is True and len(telegram.calls) == 3


def test_a_one_time_alert_that_did_not_go_out_is_tried_again(tmp_path: Path) -> None:
    telegram = Telegram(error=TimeoutError())
    alerts = _notifier(telegram, tmp_path / "telegram_sent.json")
    assert alerts.send("failure", once_key="k") is False
    telegram.error = None
    assert alerts.send("failure", once_key="k") is True and alerts.send("failure", once_key="k") is False


def test_an_unreadable_memory_does_not_stop_the_alerts(tmp_path: Path) -> None:
    memory = tmp_path / "telegram_sent.json"
    memory.write_text("{", encoding="utf-8")
    telegram = Telegram()
    assert _notifier(telegram, memory).send("failure", once_key="k") is True
    assert _notifier(telegram, memory).send("failure", once_key="k") is False  # the file was repaired
    assert _notifier(telegram).send("failure", once_key="k") is True  # without a file there is no memory


def test_the_memory_keeps_only_the_recent_keys(tmp_path: Path) -> None:
    alerts = _notifier(Telegram(), tmp_path / "telegram_sent.json")
    for day in range(notifier.REMEMBERED_KEYS + 5):
        alerts.send("failure", once_key=f"k{day}")
    assert alerts.send("failure", once_key="k0") is True and alerts.send("failure", once_key="k104") is False


# --- Chats -------------------------------------------------------------------


def test_chats_lists_who_wrote_to_the_bot() -> None:
    updates = {"ok": True, "result": [
        {"message": {"chat": {"id": 555, "type": "private", "first_name": "Ana", "last_name": "Pérez"}}},
        {"message": {"chat": {"id": 555, "type": "private", "first_name": "Ana", "last_name": "Pérez"}}},
        {"channel_post": {"chat": {"id": -100777, "type": "channel", "title": "Basket bot"}}},
        {"edited_message": {}},
    ]}
    telegram = Telegram(answer=updates)
    assert TelegramNotifier(TOKEN, "0", transport=telegram).chats() == [
        {"chat_id": "555", "type": "private", "name": "Ana Pérez"},
        {"chat_id": "-100777", "type": "channel", "name": "Basket bot"},
    ]
    assert telegram.calls[0][0].endswith("/getUpdates")
    with pytest.raises(ValueError, match="Unauthorized"):
        TelegramNotifier(TOKEN, "0", transport=Telegram(answer={"ok": False, "description": "Unauthorized"})).chats()


# --- Messages ----------------------------------------------------------------


def _summary(**fields: Any) -> dict[str, Any]:
    return {"cycle_date": "2026-10-13", "mode": "DRY_RUN", "status": "COMPLETED", "late_minutes": 0.0,
            "opens": [], "closes": [], **fields}


ACCOUNT = FuturesAccount(wallet_balance=9_998.77, available_balance=7_500.0, unrealized_pnl=-150.97)
POSITIONS = [FuturesPosition("SUSDT", 12_785.0, 0.061114, 0.0578, 0.05305),
             FuturesPosition("NEARUSDT", 169.0, 5.9316, 5.558, 5.383)]


def test_the_message_of_an_eventful_cycle() -> None:
    summary = _summary(
        opens=[{"symbol": "NEARUSDT", "status": "SIMULATED", "quantity": 169.0, "price": 5.93161486, "stop_price": 5.383},
               {"symbol": "BTCUSDT", "status": "FILLED", "quantity": 0.012, "price": 83_723.0, "stop_price": 79_174.4}],
        closes=[{"symbol": "APTUSDT", "status": "SIMULATED", "reason": "EXIT_CHANNEL_BROKEN", "price": 0.8414,
                 "pnl": -45.71}],
    )
    stopped = [{"symbol": "1000PEPEUSDT", "exit_price": 0.003637, "pnl": -92.72, "reason": "STOP"}]
    text, eventful = cycle_message(summary, stopped, ACCOUNT, POSITIONS)
    assert eventful is True
    assert text.splitlines() == [
        "Softronic QuantBot · DRY_RUN · cycle 2026-10-13",
        "BUYS",
        "• NEAR: 169 at 5.9316 · stop 5.3830",
        "• BTC: 0.012 at 83,723.00 · stop 79,174.40",
        "SELLS",
        "• APT: at 0.841400 on exit channel · −45.71 USDT",
        "STOPS",
        "• 1000PEPE: at 0.003637 · −92.72 USDT",
        "Account: 9,847.80 USDT (unrealized PnL −150.97)",
        "Positions (2): S, NEAR",
    ]


def test_the_message_of_a_quiet_cycle() -> None:
    text, eventful = cycle_message(_summary(), [], ACCOUNT, [])
    assert eventful is False
    assert text.splitlines() == [
        "Softronic QuantBot · DRY_RUN · cycle 2026-10-13",
        "No changes: no buys or sells.",
        "Account: 9,847.80 USDT (unrealized PnL −150.97)",
        "Positions (0): none",
    ]
    # If the account could not be read, the alert goes out without it.
    assert cycle_message(_summary(), [], None, None)[0].splitlines()[-1] == "No changes: no buys or sells."


def test_the_message_flags_what_needs_a_look() -> None:
    summary = _summary(
        status="COMPLETED_WITH_ERRORS", late_minutes=400.0,
        opens=[{"symbol": "AAAUSDT", "status": "UNPROTECTED", "quantity": 5.0, "price": 104.0},
               {"symbol": "BBBUSDT", "status": "STOP_FAILED_CLOSED"},
               {"symbol": "CCCUSDT", "status": "TOO_LATE", "reason": "TOO_LATE"},
               {"symbol": "DDDUSDT", "status": "REJECTED", "reason": "BELOW_MIN_NOTIONAL"},
               {"symbol": "EEEUSDT", "status": "ERROR", "reason": "the exchange did not answer"},
               {"symbol": "FFFUSDT", "status": "ALREADY_DISPATCHED", "quantity": 2.0, "price": 10.0, "stop_price": 9.0}],
        closes=[{"symbol": "GGGUSDT", "status": "NO_POSITION"}],
    )
    text, eventful = cycle_message(summary, [], None, None)
    assert eventful is True
    assert text.splitlines() == [
        "Softronic QuantBot · DRY_RUN · cycle 2026-10-13",
        "Started 400 min late.",
        "BUYS",
        "• FFF: 2 at 10.0000 · stop 9.0000",  # it was sent before a restart: it counts as done
        "ATTENTION",
        "• AAA: BOUGHT WITHOUT A STOP on the exchange: check it now",
        "• BBB: bought, the stop could not be placed and it was closed",
        "• CCC: not bought: the cycle started too late",
        "• DDD: not executed, REJECTED (BELOW_MIN_NOTIONAL)",
        "• EEE: not executed, ERROR (the exchange did not answer)",
    ]


def test_failure_messages() -> None:
    retrying = failure_message("DRY_RUN", "2026-10-13", "No symbol has valid daily candles.", 300.0)
    assert retrying.splitlines() == [
        "Softronic QuantBot · DRY_RUN · cycle 2026-10-13",
        "ATTENTION: the cycle could not run. No symbol has valid daily candles.",
        "Retrying every 5 min; if it succeeds the normal alert will follow.",
    ]
    stopped = failure_message("LIVE", "2026-10-13", "The state is damaged.", None)
    assert stopped.splitlines()[-1] == "The process has stopped: it needs to be checked."


# --- Configuration -----------------------------------------------------------


def test_without_token_or_chat_there_is_no_notifier(tmp_path: Path) -> None:
    assert build_notifier(Settings(_env_file=None, state_dir=tmp_path)) is None
    assert build_notifier(Settings(_env_file=None, state_dir=tmp_path, telegram={"bot_token": TOKEN})) is None
    assert build_notifier(Settings(_env_file=None, state_dir=tmp_path, telegram={"bot_token": "", "chat_id": " "})) is None

    settings = Settings(_env_file=None, state_dir=tmp_path, telegram={"bot_token": TOKEN, "chat_id": -100200300})
    assert settings.telegram.chat_id == "-100200300" and settings.telegram.quiet_days is True
    assert TOKEN not in repr(settings.telegram)
    built = build_notifier(settings)
    assert isinstance(built, TelegramNotifier) and built._memory == tmp_path / "telegram_sent.json"


def test_the_command_line_explains_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(notifier, "configure_logging", lambda _: None)
    monkeypatch.setattr(notifier, "get_settings", lambda: Settings(_env_file=None, state_dir=tmp_path))
    assert notifier.main(["--test"]) == 1 and "BOT_TELEGRAM__BOT_TOKEN" in capsys.readouterr().out

    only_token = Settings(_env_file=None, state_dir=tmp_path, telegram={"bot_token": TOKEN})
    monkeypatch.setattr(notifier, "get_settings", lambda: only_token)
    assert notifier.main(["--test"]) == 1 and "BOT_TELEGRAM__CHAT_ID" in capsys.readouterr().out

    telegram = Telegram(answer={"ok": True, "result": [{"message": {"chat": {"id": 555, "type": "private", "first_name": "Ana"}}}]})
    monkeypatch.setattr(notifier, "_call", telegram)
    assert notifier.main(["--chats"]) == 0
    out = capsys.readouterr().out
    assert "BOT_TELEGRAM__CHAT_ID=555" in out and TOKEN not in out

    both = Settings(_env_file=None, state_dir=tmp_path, telegram={"bot_token": TOKEN, "chat_id": "555"})
    monkeypatch.setattr(notifier, "get_settings", lambda: both)
    assert notifier.main(["--test"]) == 0 and "Message sent" in capsys.readouterr().out
    assert telegram.calls[-1][1]["chat_id"] == "555"


def test_repairs_and_missing_candles_ask_for_attention() -> None:
    summary = _summary(
        protected=[{"symbol": "OPUSDT", "status": "STOP_RESTORED", "stop_price": 0.12},
                   {"symbol": "SUSDT", "status": "UNPROTECTED", "reason": "no candles"}],
        closes=[{"symbol": "APTUSDT", "status": "PARTIAL_CLOSE", "reason": "3 left unsold"}],
        without_data=["NEARUSDT"],
    )
    text, eventful = cycle_message(summary, [], None, None)
    assert eventful
    assert text.splitlines()[1:] == [
        "ATTENTION",
        "• OP: had no stop on the exchange; it has been placed again",
        "• S: HAS NO STOP on the exchange and it could not be fixed: check it now",
        "• APT: the sell filled only in part; the rest stays open with its stop",
        "• NEAR: no candles today; its exit could not be checked",
    ]
