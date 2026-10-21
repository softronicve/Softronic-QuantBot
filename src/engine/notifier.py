"""Telegram alerts of the daily basket cycle.

The cycle sends an alert about what it did (buys, sells, stops, orders that did
not go out) and about when it could not run. An alert never stops the cycle: if
Telegram does not answer, it is written to the log and the cycle goes on.

The bot token is only read from the settings (``BOT_TELEGRAM__BOT_TOKEN``):
it is not written to the log or to any error message.

    python -m src.engine.notifier --chats   # chats that have written to the bot, to find the chat_id
    python -m src.engine.notifier --test    # sends a test message
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from config.settings import Settings, get_settings
from src.atomic_file import write_atomically
from src.engine.daily_reporter import stop_check_text
from src.console import configure_logging
from src.execution.binance_futures_executor import (
    CLOSED_UNPROTECTED,
    DONE,
    NO_POSITION,
    PARTIAL_CLOSE,
    STOP_FAILED_CLOSED,
    STOP_RESTORED,
    UNPROTECTED,
    FuturesAccount,
    FuturesPosition,
)

API = "https://api.telegram.org"
MAX_TEXT = 4096
REMEMBERED_KEYS = 100
# Close reasons the bot writes, in words.
REASONS = {"EXIT_CHANNEL_BROKEN": "exit channel", "STOP": "stop"}
# An order already sent before a restart of the same cycle counts as done.
ALREADY_DISPATCHED = "ALREADY_DISPATCHED"
EXECUTED = DONE | {ALREADY_DISPATCHED}
# What calls for a look at the account, in words; the rest is shown with its status.
PROBLEMS = {
    UNPROTECTED: "BOUGHT WITHOUT A STOP on the exchange: check it now",
    STOP_FAILED_CLOSED: "bought, the stop could not be placed and it was closed",
    "TOO_LATE": "not bought: the cycle started too late",
    "CLOSE_FAILED": "not bought: a sell failed first",
    PARTIAL_CLOSE: "the sell filled only in part; the rest stays open with its stop",
    "ENTRIES_DECIDED": "not bought: the signal showed up when the cycle ran again, after the day's buys were settled",
}
# A position found without a stop on the exchange, and what was done about it.
REPAIRS = {
    STOP_RESTORED: "had no stop on the exchange; it has been placed again",
    CLOSED_UNPROTECTED: "had no stop and could not be protected; it has been closed",
    UNPROTECTED: "HAS NO STOP on the exchange and it could not be fixed: check it now",
}

Transport = Callable[[str, Mapping[str, str], float], Any]


class Notifier(Protocol):
    def send(self, text: str, *, once_key: str | None = None) -> bool: ...


def _call(url: str, fields: Mapping[str, str], timeout: float) -> Any:
    body = urllib.parse.urlencode(fields).encode("utf-8")
    with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=timeout) as response:
        return json.loads(response.read())


class TelegramNotifier:
    """Sends messages to a Telegram chat with a bot of your own."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        memory: Path | None = None,
        timeout: float = 5.0,
        transport: Transport | None = None,
    ) -> None:
        """
        Args:
            token: bot token (from @BotFather).
            chat_id: chat the messages go to.
            memory: file that remembers which one-time alerts already went
                out, so they are not repeated if the process restarts.
            timeout: longest wait per message.
            transport: how the API is called; for tests.
        """
        if not token or not chat_id:
            raise ValueError("Telegram needs the bot token and the chat_id")
        self._token, self._chat_id = token, str(chat_id)
        self._memory = memory
        self._timeout = timeout
        self._transport = transport

    def send(self, text: str, *, once_key: str | None = None) -> bool:
        """Send ``text``. Never raises: returns whether the message went out.

        With ``once_key`` the alert goes out only once however often it is
        asked for (a cycle retried every few minutes, a process that restarts).
        """
        if once_key is not None and once_key in self._sent_keys():
            return False
        try:
            answer = self._request("sendMessage", {
                "chat_id": self._chat_id, "text": text[:MAX_TEXT], "disable_web_page_preview": "true",
            })
        except Exception as error:  # an alert that fails must never bring the cycle down
            logger.warning("Telegram alert not sent: {}", _describe(error))
            return False
        if not (isinstance(answer, dict) and answer.get("ok")):
            logger.warning("Telegram alert rejected: {}", _refusal(answer))
            return False
        if once_key is not None:
            self._remember(once_key)
        return True

    def chats(self) -> list[dict[str, Any]]:
        """Chats that have written to the bot recently (``getUpdates``), to find out the ``chat_id``.

        Raises:
            OSError, ValueError: if Telegram does not answer or rejects the token.
        """
        answer = self._request("getUpdates", {"timeout": "0"})
        if not (isinstance(answer, dict) and answer.get("ok")):
            raise ValueError(f"Telegram rejected the query: {_refusal(answer)}")
        found: dict[str, dict[str, Any]] = {}
        for update in answer.get("result", []):
            chat = (update.get("message") or update.get("channel_post") or {}).get("chat") or {}
            if "id" in chat:
                name = chat.get("title") or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")]))
                found[str(chat["id"])] = {"chat_id": str(chat["id"]), "type": chat.get("type", ""), "name": name}
        return list(found.values())

    def _request(self, method: str, fields: Mapping[str, str]) -> Any:
        return (self._transport or _call)(f"{API}/bot{self._token}/{method}", fields, self._timeout)

    def _sent_keys(self) -> list[str]:
        if self._memory is None:
            return []
        try:
            keys = json.loads(self._memory.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [key for key in keys if isinstance(key, str)] if isinstance(keys, list) else []

    def _remember(self, key: str) -> None:
        if self._memory is None:
            return
        keys = [*self._sent_keys(), key][-REMEMBERED_KEYS:]
        try:
            write_atomically(self._memory, json.dumps(keys))
        except OSError as error:
            logger.warning("Could not record the alert sent ({}); it might be repeated", error)


def _describe(error: Exception) -> str:
    """The failure without the request address, which carries the token."""
    if isinstance(error, urllib.error.HTTPError):
        return f"Telegram answered {error.code}"
    if isinstance(error, urllib.error.URLError):
        return f"no connection to Telegram ({type(error.reason).__name__})"
    return type(error).__name__


def _refusal(answer: Any) -> str:
    return str(answer.get("description", "no detail")) if isinstance(answer, dict) else "unexpected answer"


def build_notifier(settings: Settings) -> TelegramNotifier | None:
    """The notifier the settings ask for, or ``None`` if the token or the chat is missing."""
    telegram = settings.telegram
    if telegram.bot_token is None or telegram.chat_id is None:
        return None
    return TelegramNotifier(
        telegram.bot_token.get_secret_value(), telegram.chat_id, memory=settings.state_dir / "telegram_sent.json"
    )


# --- Messages ----------------------------------------------------------------


def _number(value: float, decimals: int = 2) -> str:
    """A number the English way: 9,847.80."""
    return f"{value:,.{decimals}f}"


def _price(value: float) -> str:
    magnitude = abs(value)
    return _number(value, 2 if magnitude >= 1000 else 4 if magnitude >= 1 else 6)


def _quantity(value: float) -> str:
    return f"{value:g}"


def _signed(value: float) -> str:
    return f"{'+' if value >= 0 else '−'}{_number(abs(value))}"


def _coin(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def _header(mode: str, cycle_date: str) -> str:
    return f"Softronic QuantBot · {mode} · cycle {cycle_date}"


def cycle_message(
    summary: Mapping[str, Any],
    stopped: Sequence[Mapping[str, Any]],
    account: FuturesAccount | None,
    positions: Sequence[FuturesPosition] | None,
    exit_channels: Mapping[str, float] | None = None,
) -> tuple[str, bool]:
    """The alert of a finished cycle and whether anything happened in it (buys, sells, stops or failures).

    Args:
        summary: summary of the cycle, as ``DailyRunner`` saves it.
        stopped: trades closed by the simulated stop since the previous cycle.
        account: the account after the orders; ``None`` if it could not be read.
        positions: the positions after the orders.
        exit_channels: exit channel (lowest low of the exit period) of each held symbol, shown next to its stop.
    """
    opens, closes = list(summary.get("opens", [])), list(summary.get("closes", []))
    bought = [order for order in opens if order.get("status") in EXECUTED]
    sold = [order for order in closes if order.get("status") in EXECUTED]
    # A sell of a position that was no longer there is not a failure: there was nothing to sell.
    failed = [order for order in (*closes, *opens) if order.get("status") not in EXECUTED | {NO_POSITION}]
    lines = [_header(str(summary["mode"]), str(summary["cycle_date"]))]
    late = float(summary.get("late_minutes") or 0.0)
    if late > 5.0:
        lines.append(f"Started {late:.0f} min late.")
    if bought:
        lines.append("BUYS")
        lines += [f"• {_coin(order['symbol'])}: {_quantity(order['quantity'])} at {_price(order['price'])} · stop "
                  f"{_price(order['stop_price'])}" for order in bought]
    if sold:
        lines.append("SELLS")
        lines += [f"• {_coin(order['symbol'])}: at {_price(order['price'])} on {_reason(order)}{_result(order)}"
                  for order in sold]
    if stopped:
        lines.append("STOPS")
        lines += [f"• {_coin(trade['symbol'])}: at {_price(trade['exit_price'])}{_result(trade)}" for trade in stopped]
    repaired = list(summary.get("protected", []))
    blind = list(summary.get("without_data", []))
    retired = list(summary.get("untradable", []))
    if failed or repaired or blind or retired:
        lines.append("ATTENTION")
        lines += [f"• {_coin(order['symbol'])}: {REPAIRS.get(str(order.get('status')), _problem(order))}"
                  for order in repaired]
        lines += [f"• {_coin(order['symbol'])}: {_problem(order)}" for order in failed]
        lines += [f"• {_coin(symbol)}: no candles today; its exit could not be checked" for symbol in blind]
        lines += [f"• {_coin(symbol)}: Binance no longer trades it (delisting); its exit is still checked with the "
                  "last known rules" for symbol in retired]
    eventful = bool(bought or sold or stopped or failed or repaired or blind or retired)
    if not eventful:
        lines.append("No changes: no buys or sells.")
    if account is not None and positions is not None:
        equity = account.wallet_balance + account.unrealized_pnl
        heat = sum(position.quantity * max(position.entry_price - position.stop_price, 0.0)
                   for position in positions if position.stop_price is not None and position.quantity > 0.0)
        heat_pct = heat / account.wallet_balance * 100.0 if account.wallet_balance > 0.0 else 0.0
        lines.append(f"Account: {_number(equity)} USDT (unrealized PnL {_signed(account.unrealized_pnl)})")
        lines.append(f"Balance {_number(account.wallet_balance)} USDT · open risk {_number(heat_pct)} %")
        lines.append(f"Positions ({len(positions)}){':' if positions else ': none'}")
        channels = exit_channels or {}
        lines += [
            f"• {_coin(position.symbol)}: entry {_price(position.entry_price)} · stop "
            f"{_price(position.stop_price) if position.stop_price is not None else 'NONE'}"
            + (f" · exit channel {_price(channels[position.symbol])}" if position.symbol in channels else "")
            + f" · {_signed(position.quantity * (position.mark_price - position.entry_price))} USDT"
            for position in positions
        ]
    if summary.get("stop_check") is not None:
        lines.append(f"Stops: {stop_check_text(summary['stop_check'])}")
    return "\n".join(lines), eventful


def _problem(order: Mapping[str, Any]) -> str:
    status, reason = order.get("status"), order.get("reason")
    if status in PROBLEMS:
        return PROBLEMS[str(status)]
    return f"not executed, {status} ({reason})" if reason not in (None, status) else f"not executed, {status}"


def _reason(order: Mapping[str, Any]) -> str:
    # The sells the cycle decides are always on the exit channel; the stop is handled apart.
    reason = order.get("reason")
    return REASONS.get(str(reason), str(reason)) if reason else REASONS["EXIT_CHANNEL_BROKEN"]


def _result(trade: Mapping[str, Any]) -> str:
    pnl = trade.get("pnl")
    return f" · {_signed(float(pnl))} USDT" if isinstance(pnl, (int, float)) else ""


def failure_message(mode: str, cycle_date: str, problem: str, retry_seconds: float | None) -> str:
    """The alert of a cycle that could not run."""
    ending = (f"Retrying every {retry_seconds / 60:g} min; if it succeeds the normal alert will follow."
              if retry_seconds else "The process has stopped: it needs to be checked.")
    return "\n".join([_header(mode, cycle_date), f"ATTENTION: the cycle could not run. {problem}", ending])


# --- Command line -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telegram alerts of the daily cycle")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--chats", action="store_true", help="list the chats that have written to the bot")
    action.add_argument("--test", action="store_true", help="send a test message")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    token = settings.telegram.bot_token
    if token is None:
        print("BOT_TELEGRAM__BOT_TOKEN is missing from the .env file.")
        return 1
    if args.chats:
        try:
            chats = TelegramNotifier(token.get_secret_value(), "0").chats()
        except (OSError, ValueError) as error:
            print(f"Could not query Telegram: {_describe(error) if isinstance(error, OSError) else error}")
            return 1
        if not chats:
            print("No chat has written to the bot yet: send it a message from Telegram and try again.")
            return 1
        for chat in chats:
            print(f"BOT_TELEGRAM__CHAT_ID={chat['chat_id']}    ({chat['type']}: {chat['name']})")
        return 0
    notifier = build_notifier(settings)
    if notifier is None:
        print("BOT_TELEGRAM__CHAT_ID is missing from the .env file (find it with --chats).")
        return 1
    sent = notifier.send("Softronic QuantBot: test message. The daily cycle alerts will arrive in this chat.")
    print("Message sent." if sent else "The message did not go out; see the warning in the log.")
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
