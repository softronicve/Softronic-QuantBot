"""Basket dashboard: the portfolio, the radar of its coins and the detail of each decision.

It is read-only. What it shows comes from the files the daily cycle leaves on
disk: the last report (``reports/``), the candle cache, the cycles that have
run (``runtime_state_<account>.json``) and, in ``DRY_RUN``, the paper
account. Those files are written atomically, so reading them never
slows down or interferes with the cycle.

Decisions and positions are those of the last cycle: on daily candles they only
change once a day. The only thing asked for outside is the current price of the
open positions (``src/ui/quotes.py``: public candles, no credentials), to show
how each one is doing; if it does not arrive, the price of the last cycle is shown.

Usage::

    python -m src.ui.dashboard             # http://127.0.0.1:8770
    python -m src.ui.dashboard --port 8771
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from loguru import logger

from config.settings import Settings, get_settings
from src.console import configure_logging
from src.engine.daily_reporter import MARKET_CACHE, REPORT_PREFIX, report_paths
from src.engine.daily_runner import cycle_time, effective_mode, next_cycle_time, reports_dir, state_path
from src.engine.state_store import COMPLETED, DailyStateStore, StateError
from src.ui.analytics import by_symbol, execution_quality, executions_of, trade_statistics
from src.ui.context import MarketContext
from src.ui.floor import quant_floor
from src.ui.quotes import (
    PUBLIC_API,
    PUBLIC_STREAM,
    UPSTREAM_ERRORS,
    LiveBySymbol,
    LiveCandles,
    LiveFeed,
    LivePrices,
    LiveQuotes,
    Quote,
    derivatives,
    funding_rates,
    market_stats,
    order_book,
    UnsupportedRequest,
)

STATIC_DIR = Path(__file__).parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}
JSON = "application/json; charset=utf-8"
TEXT = "text/plain; charset=utf-8"
# Names the panel answers to. A page on another site that rebinds its own domain to 127.0.0.1 (DNS
# rebinding) still sends that domain as Host, so checking it keeps such a page from reading the account.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
WILDCARD_ADDRESSES = frozenset({"", "0.0.0.0", "::"})
# An idle or half-sent request does not keep a server thread forever.
REQUEST_TIMEOUT_SECONDS = 30.0
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; connect-src 'self'{stream}; img-src 'self' data:; "
    "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"
)
# Other markets listed on Binance Futures as perpetual contracts: they come from the same price request.
CROSS_ASSETS = (
    ("SPYUSDT", "S&P 500 (SPY)"), ("QQQUSDT", "Nasdaq 100 (QQQ)"), ("XAUUSDT", "Gold"), ("CLUSDT", "WTI crude oil"),
)
HISTORY_CYCLES = 30
RECENT_TRADES = 50
# What would happen to a position if today's candle closed at the current price.
HOLDING = "HOLD"
STOP_TOUCHED = "STOP_TOUCHED"
BELOW_EXIT_CHANNEL = "BELOW_EXIT_CHANNEL"
# What the account's performance is compared with.
BENCHMARK_SYMBOL = "BTCUSDT"
# What changes between cycles: it is what the page asks for again every few seconds.
LIVE_KEYS = ("meta", "positions", "live", "performance", "radar")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class ReportNotFoundError(LookupError):
    """There is no report for the requested date."""


class BasketPanel:
    """Reads from disk what the daily cycle left and puts it together in a single snapshot."""

    def __init__(
        self,
        state_file: Path,
        reports: Path,
        paper_file: Path | None = None,
        *,
        run_delay_seconds: float = 90.0,
        clock: Callable[[], datetime] | None = None,
        quotes: LiveQuotes | None = None,
        prices: LivePrices | None = None,
        stats: LiveFeed | None = None,
        funding: LiveFeed | None = None,
        stream: str | None = None,
    ) -> None:
        """
        Args:
            state_file: cycles that have run (``runtime_state_<account>.json``).
            reports: folder of the daily reports and of the candle cache.
            paper_file: paper account, to list its closed trades;
                ``None`` outside ``DRY_RUN``.
            run_delay_seconds: seconds after 00:00 UTC at which the cycle
                runs; the time of the next one is announced with them.
            clock: clock (UTC) of the dashboard; the system one if not given.
            quotes: current price of the open positions; without it the one
                of the last cycle is shown.
            prices: current price of the whole basket, for the radar; without
                it the radar only shows the last cycle.
            stats: 24-hour change and volume of each contract.
            funding: funding rate of each contract.
            stream: where the browser can listen to the live order book and
                tape; ``None`` if it is not allowed out.
        """
        self._state_file = state_file
        self._reports = reports
        self._paper_file = paper_file
        self._run_delay = timedelta(seconds=run_delay_seconds)
        self._now = clock or (lambda: datetime.now(timezone.utc))
        self._quotes = quotes
        self._prices = prices
        self._stats, self._funding = stats, funding
        self._stream = stream
        # What each report already read adds to the curve and to the analytics, with the modification time
        # it was read with.
        self._digests: dict[str, tuple[int, dict[str, Any]]] = {}
        self._coins: list[str] = []

    def coins(self) -> list[str]:
        """Tickers of the basket coins of the last report (``BTC``, ``1000PEPE``...)."""
        return list(self._coins)

    def report_dates(self) -> list[str]:
        """Dates with a report, from the most recent to the oldest."""
        if not self._reports.is_dir():
            return []
        dates = [path.stem.removeprefix(REPORT_PREFIX) for path in self._reports.glob(f"{REPORT_PREFIX}*.json")]
        return sorted((day for day in dates if _DATE.fullmatch(day)), reverse=True)

    def snapshot(self) -> dict[str, Any]:
        """Everything the page needs, in a dictionary ready for JSON."""
        notices: list[str] = []
        dates = self.report_dates()
        report = self._read(report_paths(self._reports, dates[0])[1], notices) if dates else None
        market = self._read(self._reports / MARKET_CACHE, notices) or {}
        if report is not None and market.get("cycle_date") != report.get("cycle_date"):
            # The cache is written right after the report: they only differ for an instant.
            notices.append("The chart candles are from a different cycle than the report.")
        now = self._now()
        cycles = self._cycles(notices)
        paper = self._read(self._paper_file, notices) if self._paper_file is not None else None
        paper = paper if isinstance(paper, dict) else {}
        held = [position["symbol"] for position in (report or {}).get("positions") or []]
        # With positions BTC is requested too, to carry the benchmark of the curve up to now.
        quotes, read_at = (
            self._quotes.get([*held, BENCHMARK_SYMBOL]) if held and self._quotes is not None else ({}, None)
        )
        positions, live = self._positions(report, market.get("symbols", {}), paper, quotes, read_at)
        prices, prices_at = self._prices.get() if report is not None and self._prices is not None else ({}, None)
        stats = self._stats.get()[0] if prices and self._stats is not None else {}
        benchmark = quotes.get(BENCHMARK_SYMBOL)
        digests = self._report_digests(dates)
        self._coins = [row["symbol"].removesuffix("USDT") for row in (report or {}).get("basket", [])]
        performance = _performance(
            [digest["point"] for digest in digests if digest["point"] is not None],
            live["equity"] if live["live"] else None,
            benchmark.price if benchmark is not None else None,
            int(now.timestamp()),
        )
        radar = _radar(
            report, market.get("symbols", {}), prices, prices_at, set(held), stats,
            self._funding.get()[0] if prices and self._funding is not None else {},
        )
        history = self._history(cycles)
        return {
            "meta": {
                "generated_at": int(now.timestamp() * 1000),
                "next_cycle": self._next_cycle(cycles, now),
                "notices": notices,
                "report_dates": dates,
                "stream": self._stream,
            },
            "report": report,
            "positions": positions,
            "live": live,
            "performance": performance,
            "radar": radar,
            "analytics": {
                "trades": trade_statistics(paper.get("trades") if isinstance(paper.get("trades"), list) else []),
                "symbols": by_symbol(paper.get("trades") if isinstance(paper.get("trades"), list) else [], positions),
                "execution": execution_quality([note for digest in digests for note in digest["executions"]]),
            },
            "market": market.get("symbols", {}),
            "history": history,
            "floor": quant_floor(report, positions, radar, history),
            "trades": self._trades(paper),
        }

    def report_file(self, kind: str, cycle_date: str | None = None) -> tuple[bytes, str]:
        """The report of a date (the last one if not given), as ``md`` or ``json``.

        Raises:
            ReportNotFoundError: if there is no report for that date.
        """
        dates = self.report_dates()
        day = cycle_date if cycle_date is not None else (dates[0] if dates else "")
        # The date has to be one of those that exist: a path is never built from what arrives in the URL.
        if day not in dates:
            # Without repeating what was asked: the message travels in the response.
            raise ReportNotFoundError("no report for that date" if dates else "there is no report yet")
        path = report_paths(self._reports, day)[0 if kind == "md" else 1]
        try:
            return path.read_bytes(), path.name
        except OSError as error:
            raise ReportNotFoundError(f"could not read {path.name}: {error}") from error

    def _cycles(self, notices: list[str]) -> list[dict[str, Any]]:
        """The last recorded cycles; none if the state does not exist or cannot be read (and then a notice is added)."""
        if not self._state_file.exists():
            return []
        try:
            return DailyStateStore(self._state_file).recent_cycles(HISTORY_CYCLES)
        except StateError as error:
            notices.append(str(error))
            return []

    def _next_cycle(self, cycles: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
        """When the next cycle is due.

        If today's is past its time and is not recorded as completed, the next
        one is that one and it is marked as pending: the daily cycle runs it or
        retries it as soon as it can.
        """
        today = cycle_time(now.date(), self._run_delay)
        done = any(
            cycle["cycle_date"] == now.date().isoformat() and cycle.get("status") == COMPLETED for cycle in cycles
        )
        pending = now >= today and not done
        moment = today if pending else next_cycle_time(now, self._run_delay)
        return {"at": int(moment.timestamp() * 1000), "pending": pending}

    @staticmethod
    def _history(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "cycle_date": cycle["cycle_date"],
                "status": (cycle.get("summary") or {}).get("status", cycle.get("status")),
                "completed_at": cycle.get("completed_at"),
                "late_minutes": (cycle.get("summary") or {}).get("late_minutes"),
                "opens": (cycle.get("summary") or {}).get("opens", []),
                "closes": (cycle.get("summary") or {}).get("closes", []),
                "holds": (cycle.get("summary") or {}).get("holds", {}),
                "stopped": (cycle.get("summary") or {}).get("stopped", []),
            }
            for cycle in cycles
        ]

    def _report_digests(self, dates: list[str]) -> list[dict[str, Any]]:
        """What each daily report adds, from the oldest to the most recent. Each report is read only once."""
        digests = []
        for day in sorted(dates):
            path = report_paths(self._reports, day)[1]
            try:
                stamp = path.stat().st_mtime_ns
            except OSError:
                continue
            known = self._digests.get(day)
            if known is None or known[0] != stamp:
                report = self._read(path, [])
                known = (stamp, {"point": _equity_point(day, report),
                                 "executions": executions_of(report) if isinstance(report, dict) else []})
                self._digests[day] = known
            digests.append(known[1])
        return digests

    def _positions(
        self, report: dict[str, Any] | None, market: dict[str, Any], paper: dict[str, Any],
        quotes: dict[str, Quote], read_at: int | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Each open position with its current price, its result and the two levels at which it would be sold.

        Positions are those of the last report; the price, the current one if
        it could be read. The exit channel is the one the next cycle will use:
        the low of the last closed candles.
        """
        held = (report or {}).get("positions") or []
        if not held:
            return [], {"at": None, "live": False, "open_pnl_usd": 0.0, "equity": None}
        assert report is not None
        exit_period = int(report.get("parameters", {}).get("exit_period", 0))
        basket = {row["symbol"]: row for row in report.get("basket", [])}
        simulated = paper.get("positions") if isinstance(paper.get("positions"), dict) else {}
        rows = [
            _position_now(position, quotes.get(position["symbol"]), _next_exit_channel(
                market.get(position["symbol"]), exit_period, basket.get(position["symbol"])
            ), simulated.get(position["symbol"]) or {})
            for position in held
        ]
        open_pnl = sum(row["pnl_usd"] for row in rows)
        live = {
            "at": read_at,
            "live": all(row["price_is_live"] for row in rows),
            "open_pnl_usd": open_pnl,
            "equity": float(report["account"]["wallet_balance"]) + open_pnl,
        }
        return rows, live

    @staticmethod
    def _trades(paper: dict[str, Any]) -> list[dict[str, Any]]:
        trades = paper.get("trades")
        return list(reversed(trades[-RECENT_TRADES:])) if isinstance(trades, list) else []

    @staticmethod
    def _read(path: Path, notices: list[str]) -> Any:
        """A JSON file; ``None`` if it does not exist or cannot be read (and then a notice is added)."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            notices.append(f"Could not read {path.name}: {error}")
            return None


def _radar(
    report: dict[str, Any] | None, market: dict[str, Any], prices: dict[str, float], read_at: int | None,
    held: set[str], stats: dict[str, dict[str, float]], funding: dict[str, float],
) -> dict[str, Any]:
    """Where each basket coin is now relative to the levels the next cycle will use.

    The next cycle will compare today's close with the high of the last
    closed candles (to buy) and with their low (to sell). Here the current
    price is compared with them: it is a preview, not a decision. What
    counts is the close of the candle, and a buy also depends on the free risk.
    """
    rows: dict[str, dict[str, Any]] = {}
    parameters = (report or {}).get("parameters") or {}
    entry_period, exit_period = int(parameters.get("entry_period", 0)), int(parameters.get("exit_period", 0))
    for symbol in (row["symbol"] for row in (report or {}).get("basket", [])):
        candles, price = market.get(symbol) or {}, prices.get(symbol)
        highs, lows, closes = candles.get("high") or [], candles.get("low") or [], candles.get("close") or []
        if price is None or not price > 0.0 or not closes or entry_period <= 0 or len(highs) < entry_period:
            continue
        entry = float(max(highs[-entry_period:]))
        exit_channel = float(min(lows[-exit_period:])) if 0 < exit_period <= len(lows) else None
        rows[symbol] = {
            "price": price,
            "change_pct": (price / closes[-1] - 1.0) * 100.0,
            "entry_channel": entry,
            "proximity_pct": (price / entry - 1.0) * 100.0,
            "exit_channel": exit_channel,
            # Without a position, above the high; with one, below the low.
            "breaking": symbol not in held and price > entry,
            "exiting": symbol in held and exit_channel is not None and price < exit_channel,
            # From the market, not from the bot: change and volume (USDT) over the last 24 hours and funding.
            "change_24h_pct": (stats.get(symbol) or {}).get("change_24h_pct"),
            "volume_24h_usd": (stats.get(symbol) or {}).get("volume_24h_usd"),
            "funding_pct": funding.get(symbol),
        }
    cross = [
        {"symbol": symbol, "name": name, "price": prices[symbol],
         "change_24h_pct": (stats.get(symbol) or {}).get("change_24h_pct")}
        for symbol, name in CROSS_ASSETS if prices.get(symbol)
    ]
    return {"at": read_at if rows else None, "rows": rows, "cross": cross if rows else []}


def _equity_point(day: str, report: Any) -> dict[str, Any] | None:
    """What a daily report adds to the curve: the account after the cycle, how much stayed invested and the BTC close."""
    account = report.get("account") if isinstance(report, dict) else None
    if not isinstance(account, dict):
        return None
    equity, nominal = account.get("equity"), account.get("nominal_usd", 0.0)
    if not isinstance(equity, (int, float)) or not equity > 0.0 or not isinstance(nominal, (int, float)):
        return None
    closes = [row.get("close") for row in report.get("basket", []) if row.get("symbol") == BENCHMARK_SYMBOL]
    btc = closes[0] if closes and isinstance(closes[0], (int, float)) and closes[0] > 0.0 else None
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return {"date": day, "time": int(start.timestamp()), "equity": float(equity),
            "exposure": float(nominal) / float(equity), "btc": btc}


def _performance(
    cycles: list[dict[str, Any]], live_equity: float | None, live_btc: float | None, now: int
) -> dict[str, Any]:
    """The account curve, its drawdown from the peak and the BTC benchmark with the same exposure.

    There is one point per daily cycle and, with open positions and a current
    price, one more with the account as it is now. The benchmark is what holding
    in BTC, each day, the same fraction of the account the bot left invested in
    that cycle would have given; it carries no fees. A day without a BTC price does not move it.
    """
    steps = [dict(cycle) for cycle in cycles]
    if steps and live_equity is not None and now > steps[-1]["time"]:
        steps.append({"date": None, "time": now, "equity": float(live_equity), "exposure": steps[-1]["exposure"],
                      "btc": live_btc})
    points: list[dict[str, Any]] = []
    benchmark = peak = 0.0
    last_btc: float | None = None
    for index, step in enumerate(steps):
        if index == 0:
            benchmark = step["equity"]
        elif step["btc"] is not None and last_btc is not None:
            benchmark *= 1.0 + steps[index - 1]["exposure"] * (step["btc"] / last_btc - 1.0)
        last_btc = step["btc"] if step["btc"] is not None else last_btc
        peak = max(peak, step["equity"])
        points.append({
            "date": step["date"], "time": step["time"], "equity": step["equity"], "benchmark": benchmark,
            "drawdown_pct": (step["equity"] / peak - 1.0) * 100.0, "exposure_pct": step["exposure"] * 100.0,
        })
    if not points:
        return {"points": [], "stats": None}
    first, last = points[0], points[-1]
    return {
        "points": points,
        "stats": {
            "since": first["date"],
            "cycles": len(cycles),
            "start_equity": first["equity"],
            "equity": last["equity"],
            "is_live": last["date"] is None,
            "return_pct": (last["equity"] / first["equity"] - 1.0) * 100.0,
            "benchmark_return_pct": (last["benchmark"] / first["equity"] - 1.0) * 100.0,
            "drawdown_pct": last["drawdown_pct"],
            "max_drawdown_pct": min(point["drawdown_pct"] for point in points),
            "mean_exposure_pct": sum(cycle["exposure"] for cycle in cycles) / len(cycles) * 100.0,
        },
    }


def _next_exit_channel(candles: dict[str, Any] | None, period: int, row: dict[str, Any] | None) -> float | None:
    """Level the next cycle will compare today's close with: the low of the last ``period`` closed candles.

    If there are no candles in the cache the channel of the last cycle is used,
    which is the same except for the most recent candle.
    """
    lows = (candles or {}).get("low") or []
    if period > 0 and len(lows) >= period:
        return float(min(lows[-period:]))
    previous = (row or {}).get("low_exit")
    return float(previous) if isinstance(previous, (int, float)) else None


def _position_now(
    position: dict[str, Any], quote: Quote | None, exit_channel: float | None, simulated: dict[str, Any]
) -> dict[str, Any]:
    quantity, entry = float(position["quantity"]), float(position["entry_price"])
    stop = position.get("stop_price")
    stop = float(stop) if isinstance(stop, (int, float)) else None
    price = quote.price if quote is not None else float(position["mark_price"])
    # The simulated stop is checked in each cycle against the lows of the candles it has not seen yet;
    # without that mark (TESTNET and LIVE) the stop is on the exchange and today's candle is enough.
    checked = simulated.get("checked_until")
    if quote is None:
        lows = []
    elif checked is None:
        lows = [quote.candles[-1][2]]
    else:
        lows = [low for _, close_time, low in quote.candles if close_time > checked]
    low = min(lows) if lows else None
    touched = stop is not None and (price <= stop or (low is not None and low <= stop))
    below = exit_channel is not None and price < exit_channel
    fees = simulated.get("fees")
    # Once the stop is touched, the result no longer depends on the price: the position exits at the stop
    # (in DRY_RUN it is recorded that way in the next cycle; on the exchange the order will already have sold).
    worth = stop if touched and stop is not None else price
    return {
        "symbol": position["symbol"],
        "quantity": quantity,
        "entry_price": entry,
        "opened_at": simulated.get("opened_at"),
        "cost_usd": quantity * entry,
        "price": price,
        "price_is_live": quote is not None,
        "valued_at_stop": worth != price,
        "value_usd": quantity * worth,
        "pnl_usd": quantity * (worth - entry),
        "pnl_pct": (worth / entry - 1.0) * 100.0 if entry > 0.0 else 0.0,
        # Fees and funding already deducted from the balance; they are not part of the open result.
        "costs_usd": float(fees) + float(simulated.get("funding", 0.0)) if isinstance(fees, (int, float)) else None,
        "stop_price": stop,
        "stop_distance_pct": (stop / price - 1.0) * 100.0 if stop is not None and price > 0.0 else None,
        "result_at_stop_usd": quantity * (stop - entry) if stop is not None else None,
        "exit_channel": exit_channel,
        "exit_distance_pct": (exit_channel / price - 1.0) * 100.0 if exit_channel is not None and price > 0.0 else None,
        "low_since_check": low,
        "status": STOP_TOUCHED if touched else BELOW_EXIT_CHANNEL if below else HOLDING,
    }


class BasketDashboardServer(ThreadingHTTPServer):
    """Serves the page, its files and the dashboard snapshot."""

    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], panel: BasketPanel, cache_seconds: float = 5.0,
        candles: LiveCandles | None = None,
        derivatives: LiveBySymbol | None = None,
        book: LiveBySymbol | None = None,
        context: MarketContext | None = None,
        stream: str | None = None,
    ) -> None:
        """
        Args:
            address: where it listens.
            panel: what is shown.
            cache_seconds: how often the dashboard snapshot is rebuilt.
            candles: candles of other intervals for the chart; without them only those of the cycle are drawn.
            derivatives: market positioning of each contract.
            book: order book and tape of each contract on request.
            context: what is outside Binance (market, sentiment, headlines, calendar).
            stream: the only outside site the page may connect to, to listen to the live order book.
        """
        super().__init__(address, _Handler)
        self.panel = panel
        self.candles = candles
        self.by_symbol = {"/api/derivatives": derivatives, "/api/book": book}
        self.context = context
        self.csp = CONTENT_SECURITY_POLICY.format(stream=f" {stream}" if stream else "")
        bound = address[0]
        self.allowed_hosts = LOOPBACK_HOSTS | ({bound} if bound not in WILDCARD_ADDRESSES else set())
        self._cache_seconds = cache_seconds
        self._rebuild = threading.Lock()
        self._cached: tuple[bytes, bytes, bytes] | None = None
        self._cached_at = 0.0
        # Only what is in this table is served: no path can get out of it.
        self.static: dict[str, tuple[bytes, str]] = {
            path.name: (path.read_bytes(), CONTENT_TYPES[path.suffix])
            for path in STATIC_DIR.iterdir()
            if path.suffix in CONTENT_TYPES
        }

    def floor(self) -> bytes:
        """Only the quant floor, from the same cached snapshot."""
        self.snapshot()
        assert self._cached is not None
        return self._cached[2]

    def snapshot(self, live_only: bool = False) -> bytes:
        """The dashboard snapshot, rebuilt at most every ``cache_seconds``.

        With ``live_only`` it carries only what changes between cycles (``LIVE_KEYS``):
        it is what the page asks for every few seconds, without the candles or the report.

        Composing it asks Binance for prices, so it can take seconds. Only one
        request rebuilds it; the others keep getting the previous one meanwhile
        instead of queueing behind that request. Only the very first request waits.
        """
        cached = self._cached
        if cached is None or time.monotonic() - self._cached_at >= self._cache_seconds:
            if self._rebuild.acquire(blocking=cached is None):
                try:
                    if self._cached is cached:  # nobody rebuilt it while this request waited
                        self._cached = self._compose()
                        self._cached_at = time.monotonic()
                except Exception:
                    if self._cached is None:
                        raise
                    # Tried again after another cache period, not on every request.
                    self._cached_at = time.monotonic()
                    logger.exception("Could not rebuild the basket dashboard snapshot; serving the previous one")
                finally:
                    self._rebuild.release()
        assert self._cached is not None
        return self._cached[1 if live_only else 0]

    def _compose(self) -> tuple[bytes, bytes, bytes]:
        full = self.panel.snapshot()
        return (
            json.dumps(full, allow_nan=False).encode("utf-8"),
            json.dumps({key: full[key] for key in LIVE_KEYS if key in full}, allow_nan=False).encode("utf-8"),
            json.dumps(full.get("floor") or {}, allow_nan=False).encode("utf-8"),
        )


class _Handler(BaseHTTPRequestHandler):
    server: BasketDashboardServer
    timeout = REQUEST_TIMEOUT_SECONDS

    def do_GET(self) -> None:  # noqa: N802 (name imposed by http.server)
        if not self._host_allowed():
            self._send(HTTPStatus.FORBIDDEN, b"host not allowed", TEXT)
            return
        url = urlsplit(self.path)
        path = url.path
        query = parse_qs(url.query)
        if path in ("/api/snapshot", "/api/live", "/api/floor"):
            try:
                body = self.server.floor() if path == "/api/floor" else self.server.snapshot(live_only=path == "/api/live")
            except Exception:
                logger.exception("Could not build the basket dashboard snapshot")
                # The detail goes to the log only: it can carry file names and paths.
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, b'{"error": "could not build the dashboard"}', JSON)
            else:
                self._send(HTTPStatus.OK, body, JSON)
        elif path == "/api/candles":
            symbol, interval = query.get("symbol", [""])[0], query.get("interval", [""])[0]
            self._feed(
                None if self.server.candles is None else lambda: self.server.candles.get(symbol, interval),
                wrap=lambda candles: {"symbol": symbol, "interval": interval, **candles},
                failure="could not read the candles", log=logger.warning,
            )
        elif path in self.server.by_symbol:
            feed, symbol = self.server.by_symbol[path], query.get("symbol", [""])[0]
            self._feed(None if feed is None else lambda: feed.get(symbol), failure="could not read", log=logger.debug)
        elif path == "/api/context":
            self._context()
        elif path in ("/api/report.md", "/api/report.json"):
            kind = path.rsplit(".", 1)[1]
            try:
                body, filename = self.server.panel.report_file(kind, query.get("date", [None])[0])
            except ReportNotFoundError as error:
                self._send(HTTPStatus.NOT_FOUND, str(error).encode("utf-8"), TEXT)
            else:
                content_type = "text/markdown; charset=utf-8" if kind == "md" else JSON
                self._send(HTTPStatus.OK, body, content_type, download_as=filename)
        elif path == "/healthz":
            self._send(HTTPStatus.OK, b"ok", TEXT)
        else:
            name = "index.html" if path == "/" else path.removeprefix("/static/")
            asset = self.server.static.get(name) if path == "/" or path.startswith("/static/") else None
            if asset is None:
                self._send(HTTPStatus.NOT_FOUND, b"not found", TEXT)
            else:
                self._send(HTTPStatus.OK, asset[0], asset[1])

    def _host_allowed(self) -> bool:
        """Whether the ``Host`` header names this machine. Requests without one (HTTP/1.0) are let through."""
        host = self.headers.get("Host")
        if host is None:
            return True
        host = host.strip().lower().replace(".:", ":").rstrip(".")  # "localhost." is the same name
        if host.startswith("["):  # [::1]:8770
            name = host[1:host.find("]")]
        elif host.count(":") == 1:  # localhost:8770
            name = host.split(":")[0]
        else:  # no port, or a bare IPv6 address
            name = host
        return name in self.server.allowed_hosts

    def _feed(
        self,
        read: Callable[[], Any] | None,
        *,
        failure: str,
        log: Callable[..., None],
        wrap: Callable[[Any], Any] = lambda value: value,
    ) -> None:
        """Public Binance data read on request. What is asked is validated before anything goes out."""
        if read is None:
            self._send(HTTPStatus.NOT_FOUND, b"not found", TEXT)
            return
        try:
            value = read()
        except UnsupportedRequest as error:
            self._send(HTTPStatus.BAD_REQUEST, str(error).encode("utf-8"), TEXT)
        except UPSTREAM_ERRORS as error:
            # Only the type: the message can carry the address that was asked.
            log("Basket dashboard: {} ({})", failure, type(error).__name__)
            self._send(HTTPStatus.BAD_GATEWAY, failure.encode("utf-8"), TEXT)
        else:
            self._send(HTTPStatus.OK, json.dumps(wrap(value), allow_nan=False).encode("utf-8"), JSON)

    def _context(self) -> None:
        """What is outside Binance. With that part turned off the answer is empty, not an error."""
        try:
            context = self.server.context.get() if self.server.context is not None else {}
            body = json.dumps(context, allow_nan=False).encode("utf-8")
        except Exception:
            logger.exception("Could not build the market context of the dashboard")
            body = b"{}"
        self._send(HTTPStatus.OK, body, JSON)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str, download_as: str | None = None) -> None:
        try:
            self._write(status, body, content_type, download_as)
        except ConnectionError:
            # The browser closed the tab or moved on before the answer arrived: nothing went wrong here.
            logger.debug("Basket dashboard: the client left before {} was sent", self.path)

    def _write(self, status: HTTPStatus, body: bytes, content_type: str, download_as: str | None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if download_as is not None:
            self.send_header("Content-Disposition", f'attachment; filename="{download_as}"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", self.server.csp)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("HTTP {}", format % args)


def build_server(
    settings: Settings, account: str, host: str | None = None, port: int | None = None
) -> BasketDashboardServer:
    """The dashboard server for an account (``dry_run``, ``testnet`` or ``live``), already bound to its port."""
    api = PUBLIC_API[account]
    external = settings.futures.panel_external_data
    stream = PUBLIC_STREAM[account] if external else None
    panel = BasketPanel(
        state_path(settings, account), reports_dir(settings, account),
        settings.state_dir / "paper_futures.json" if account == "dry_run" else None,
        run_delay_seconds=settings.futures.run_delay_seconds,
        quotes=LiveQuotes(api),
        prices=LivePrices(api),
        stats=market_stats(api),
        funding=funding_rates(api),
        stream=stream,
    )
    return BasketDashboardServer(
        (host or settings.dashboard.host, port or settings.futures.dashboard_port), panel,
        candles=LiveCandles(api),
        derivatives=derivatives(api),
        book=order_book(api),
        context=MarketContext(panel.coins) if external else None,
        stream=stream,
    )


def serve_in_background(settings: Settings, account: str) -> BasketDashboardServer | None:
    """Starts the dashboard in a thread of the calling process; the daily cycle uses it with ``--panel``.

    The dashboard is secondary: if it cannot open its port an error is logged and
    ``None`` is returned, so the cycle goes on without it.
    """
    try:
        server = build_server(settings, account)
    except OSError as error:
        logger.error("The basket dashboard could not start ({}); the daily cycle goes on without it", error)
        return None
    threading.Thread(target=server.serve_forever, name="basket-dashboard", daemon=True).start()
    logger.info("Basket dashboard ({}) listening on http://{}:{}", account, *server.server_address[:2])
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Basket dashboard (read-only)")
    parser.add_argument("--port", type=int, help="port; overrides BOT_FUTURES__DASHBOARD_PORT")
    parser.add_argument("--host", help="interface it listens on; overrides BOT_DASHBOARD__HOST")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)
    account = effective_mode(settings).lower()
    if account == "live" and not settings.futures.live_trading_acknowledged:
        account = "dry_run"  # the cycle runs in DRY_RUN without the acknowledgment: the dashboard looks at that same account
    server = build_server(settings, account, args.host, args.port)
    address = server.server_address[:2]

    def shutdown(*_: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    logger.info("Basket dashboard ({}) listening on http://{}:{}", account, *address)
    server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
