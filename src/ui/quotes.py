"""Current prices for the basket dashboard.

It is the only thing the dashboard asks for outside the disk, from the public
Binance USDT-M Futures API and without credentials: the last price of every
contract (a single request, for the radar) and the daily candles of the symbols
with a position (to know whether the day's low has touched their stop). It is
for watching the market between one cycle and the next; it takes no part in any
decision of the bot, which keeps trading once a day with its own readings.

Each request is kept for ``ttl_seconds`` and a failure does not break the
dashboard: it shows the last known and, if there is nothing, the last cycle's.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

PUBLIC_API = {
    "dry_run": "https://fapi.binance.com",
    "live": "https://fapi.binance.com",
    "testnet": "https://testnet.binancefuture.com",
}
# Where the browser listens to the live order book and tape (public streams, no credentials).
PUBLIC_STREAM = {
    "dry_run": "wss://fstream.binance.com",
    "live": "wss://fstream.binance.com",
    "testnet": "wss://fstream.binancefuture.com",
}
RECENT_CANDLES = 3
# What the dashboard chart may ask for: only USDT contracts and these intervals.
INTERVALS = ("5m", "15m", "1h", "4h", "1d")
SYMBOL = re.compile(r"[A-Z0-9]{1,20}USDT")


@dataclass(frozen=True, slots=True)
class Quote:
    """Last price of a symbol and the lows of its latest daily candles, today's included."""

    price: float
    # (open time in ms, close time in ms, low) of each candle, from the oldest to today's.
    candles: tuple[tuple[int, int, float], ...]


def fetch_quote(base_url: str, symbol: str, timeout: float) -> Quote:
    """Reads the latest daily candles of ``symbol`` from the public API."""
    query = urllib.parse.urlencode({"symbol": symbol, "interval": "1d", "limit": RECENT_CANDLES})
    with urllib.request.urlopen(f"{base_url}/fapi/v1/klines?{query}", timeout=timeout) as response:
        rows = json.loads(response.read())
    return Quote(float(rows[-1][4]), tuple((int(row[0]), int(row[6]), float(row[3])) for row in rows))


def fetch_prices(base_url: str, timeout: float) -> dict[str, float]:
    """Last price of every contract, in a single request to the public API."""
    with urllib.request.urlopen(f"{base_url}/fapi/v1/ticker/price", timeout=timeout) as response:
        rows = json.loads(response.read())
    return {str(row["symbol"]): float(row["price"]) for row in rows}


def fetch_stats(base_url: str, timeout: float) -> dict[str, dict[str, float]]:
    """Change and USDT volume over the last 24 hours for every contract, in one request."""
    with urllib.request.urlopen(f"{base_url}/fapi/v1/ticker/24hr", timeout=timeout) as response:
        rows = json.loads(response.read())
    return {
        str(row["symbol"]): {"change_24h_pct": float(row["priceChangePercent"]), "volume_24h_usd": float(row["quoteVolume"])}
        for row in rows
    }


def fetch_funding(base_url: str, timeout: float) -> dict[str, float]:
    """Last funding rate of every contract, in % per period (8 hours for almost all of them)."""
    with urllib.request.urlopen(f"{base_url}/fapi/v1/premiumIndex", timeout=timeout) as response:
        rows = json.loads(response.read())
    return {str(row["symbol"]): float(row["lastFundingRate"]) * 100.0 for row in rows if row.get("lastFundingRate")}


def fetch_candles(base_url: str, symbol: str, interval: str, limit: int, timeout: float) -> dict[str, list[float]]:
    """The latest candles of ``symbol`` at ``interval``, the one in progress included. Times are in seconds."""
    query = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    with urllib.request.urlopen(f"{base_url}/fapi/v1/klines?{query}", timeout=timeout) as response:
        rows = json.loads(response.read())
    return {
        "time": [int(row[0]) // 1000 for row in rows],
        "open": [float(row[1]) for row in rows],
        "high": [float(row[2]) for row in rows],
        "low": [float(row[3]) for row in rows],
        "close": [float(row[4]) for row in rows],
        "volume": [float(row[7]) for row in rows],
    }


class LiveFeed:
    """A market-wide figure requested in one go and reread at most every ``ttl_seconds``."""

    def __init__(
        self,
        fetch: Callable[[], dict[str, Any]],
        *,
        what: str,
        ttl_seconds: float,
        retry_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        """
        Args:
            fetch: the request; returns the figure by symbol.
            what: what the figure is called in the log warnings.
            ttl_seconds: how long a reading is valid.
            retry_seconds: wait after a failed read; the same as
                ``ttl_seconds`` if not given. It is for data with a long
                period: without it, a failure would leave them empty all that time.
            clock: monotonic clock, for expiry.
            wall_clock: wall clock, to say when it was read.
        """
        self._fetch, self._what = fetch, what
        self._ttl = ttl_seconds
        self._retry = ttl_seconds if retry_seconds is None else retry_seconds
        self._clock, self._wall_clock = clock, wall_clock
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {}
        self._read_at: int | None = None
        self._checked_at: float | None = None
        self._failing = False

    def get(self) -> tuple[dict[str, Any], int | None]:
        """The figure and the instant (ms) it was read; the last known one if the request fails."""
        with self._lock:
            now = self._clock()
            wait = self._retry if self._failing else self._ttl
            if self._checked_at is None or now - self._checked_at >= wait:
                self._checked_at = now
                try:
                    self._values = self._fetch()
                except (OSError, ValueError, LookupError, TypeError) as error:
                    if not self._failing:
                        logger.warning("Basket dashboard: could not read {} ({}); showing the last known "
                                       "values", self._what, error)
                    self._failing = True
                else:
                    self._read_at = int(self._wall_clock() * 1000)
                    if self._failing:
                        logger.info("Basket dashboard: reading {} again", self._what)
                    self._failing = False
            return dict(self._values), self._read_at


class LivePrices(LiveFeed):
    """The last price of every contract."""

    def __init__(
        self,
        base_url: str,
        *,
        ttl_seconds: float = 10.0,
        timeout: float = 3.0,
        fetch: Callable[[], dict[str, float]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(fetch or (lambda: fetch_prices(base_url, timeout)), what="the current prices",
                         ttl_seconds=ttl_seconds, clock=clock, wall_clock=wall_clock)


def market_stats(base_url: str, *, ttl_seconds: float = 30.0, timeout: float = 4.0) -> LiveFeed:
    """24-hour change and volume of every contract. It is the most expensive request: it is reread every half minute."""
    return LiveFeed(lambda: fetch_stats(base_url, timeout), what="the 24-hour figures", ttl_seconds=ttl_seconds)


def funding_rates(base_url: str, *, ttl_seconds: float = 60.0, timeout: float = 4.0) -> LiveFeed:
    """Funding rate of every contract; it changes every 8 hours, so rereading it every minute is enough."""
    return LiveFeed(lambda: fetch_funding(base_url, timeout), what="the funding rates",
                    ttl_seconds=ttl_seconds)


def fetch_derivatives(base_url: str, symbol: str, timeout: float) -> dict[str, Any]:
    """How the market of a contract is positioned: open interest, long accounts and who is the aggressor.

    It is three requests to the public futures statistics, with hourly
    data. It also brings the circulating supply, which the market cap comes from.
    """
    def read(path: str, limit: int) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode({"symbol": symbol, "period": "1h", "limit": limit})
        with urllib.request.urlopen(f"{base_url}/futures/data/{path}?{query}", timeout=timeout) as response:
            return list(json.loads(response.read()))

    interest = read("openInterestHist", 25)
    accounts = read("globalLongShortAccountRatio", 1)
    takers = read("takerlongshortRatio", 1)
    latest, earliest = float(interest[-1]["sumOpenInterestValue"]), float(interest[0]["sumOpenInterestValue"])
    supply = interest[-1].get("CMCCirculatingSupply")
    return {
        "open_interest_usd": latest,
        "open_interest_change_24h_pct": (latest / earliest - 1.0) * 100.0 if earliest > 0.0 else None,
        "open_interest_history": [float(row["sumOpenInterestValue"]) for row in interest],
        "long_accounts_pct": float(accounts[-1]["longAccount"]) * 100.0,
        "long_short_ratio": float(accounts[-1]["longShortRatio"]),
        "taker_buy_sell_ratio": float(takers[-1]["buySellRatio"]),
        "circulating_supply": float(supply) if supply else None,
    }


def fetch_book(base_url: str, symbol: str, timeout: float, levels: int = 20, trades: int = 40) -> dict[str, Any]:
    """The order book and the latest trades of a contract, for when the browser cannot listen to them live."""
    def read(path: str, limit: int) -> Any:
        query = urllib.parse.urlencode({"symbol": symbol, "limit": limit})
        with urllib.request.urlopen(f"{base_url}/fapi/v1/{path}?{query}", timeout=timeout) as response:
            return json.loads(response.read())

    depth = read("depth", levels)
    return {
        "bids": [[float(price), float(quantity)] for price, quantity in depth["bids"]],
        "asks": [[float(price), float(quantity)] for price, quantity in depth["asks"]],
        "trades": [{"time": int(row["time"]), "price": float(row["price"]), "quantity": float(row["qty"]),
                    "sell": bool(row["isBuyerMaker"])} for row in read("trades", trades)],
    }


class LiveBySymbol:
    """A per-contract figure requested separately; each contract is reread at most every ``ttl_seconds``."""

    def __init__(
        self,
        fetch: Callable[[str], dict[str, Any]],
        *,
        ttl_seconds: float,
        max_entries: int = 64,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._ttl, self._max_entries = ttl_seconds, max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def get(self, symbol: str) -> dict[str, Any]:
        """The figure of the contract.

        Raises:
            ValueError: if it is not a USDT contract.
            OSError, LookupError, TypeError: if the API does not answer or answers something else.
        """
        if not SYMBOL.fullmatch(symbol):
            raise ValueError("unsupported symbol")
        with self._lock:
            now = self._clock()
            known = self._cache.get(symbol)
            if known is not None and now - known[0] < self._ttl:
                return known[1]
            value = self._fetch(symbol)
            self._cache.pop(symbol, None)
            self._cache[symbol] = (now, value)
            while len(self._cache) > self._max_entries:
                self._cache.pop(next(iter(self._cache)))
            return value


def derivatives(base_url: str, *, ttl_seconds: float = 60.0, timeout: float = 4.0) -> LiveBySymbol:
    """Market positioning of each contract; the data is hourly, rereading it every minute is enough."""
    return LiveBySymbol(lambda symbol: fetch_derivatives(base_url, symbol, timeout), ttl_seconds=ttl_seconds)


def order_book(base_url: str, *, ttl_seconds: float = 2.0, timeout: float = 3.0) -> LiveBySymbol:
    """Order book and tape of each contract on request; the normal path is for the browser to listen to them live."""
    return LiveBySymbol(lambda symbol: fetch_book(base_url, symbol, timeout), ttl_seconds=ttl_seconds)


class LiveCandles:
    """Candles of a symbol at an interval, for the chart; each pair is reread at most every ``ttl_seconds``."""

    def __init__(
        self,
        base_url: str,
        *,
        ttl_seconds: float = 10.0,
        timeout: float = 4.0,
        limit: int = 300,
        max_entries: int = 64,
        fetch: Callable[[str, str], dict[str, list[float]]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch or (lambda symbol, interval: fetch_candles(base_url, symbol, interval, limit, timeout))
        self._ttl, self._max_entries = ttl_seconds, max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], tuple[float, dict[str, list[float]]]] = {}

    def get(self, symbol: str, interval: str) -> dict[str, list[float]]:
        """The candles asked for.

        Raises:
            ValueError: if the symbol or the interval is not one of those served.
            OSError, LookupError, TypeError: if the API does not answer or answers something else.
        """
        if interval not in INTERVALS or not SYMBOL.fullmatch(symbol):
            raise ValueError("unsupported symbol or interval")
        key = (symbol, interval)
        with self._lock:
            now = self._clock()
            known = self._cache.get(key)
            if known is not None and now - known[0] < self._ttl:
                return known[1]
            candles = self._fetch(symbol, interval)
            self._cache.pop(key, None)
            self._cache[key] = (now, candles)
            while len(self._cache) > self._max_entries:
                self._cache.pop(next(iter(self._cache)))
            return candles


class LiveQuotes:
    """Quotes of a few symbols, reread at most every ``ttl_seconds``."""

    def __init__(
        self,
        base_url: str,
        *,
        ttl_seconds: float = 10.0,
        timeout: float = 3.0,
        fetch: Callable[[str], Quote] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        """
        Args:
            base_url: root of the public API (see ``PUBLIC_API``).
            ttl_seconds: how long a reading is valid, also if it failed.
            timeout: maximum wait per symbol.
            fetch: how a symbol is read; by default, from ``base_url``.
            clock: monotonic clock, for expiry.
            wall_clock: wall clock, to say when it was read.
        """
        self._fetch = fetch or (lambda symbol: fetch_quote(base_url, symbol, timeout))
        self._ttl = ttl_seconds
        self._clock, self._wall_clock = clock, wall_clock
        self._lock = threading.Lock()
        self._quotes: dict[str, Quote] = {}
        self._read_at: dict[str, int] = {}
        self._checked_at: float | None = None
        self._failing = False

    def get(self, symbols: Sequence[str]) -> tuple[dict[str, Quote], int | None]:
        """Quote of each symbol that could be read and the instant (ms) of the oldest reading.

        A symbol that could never be read is left out. If the request
        fails, what was known before is returned.
        """
        wanted = list(dict.fromkeys(symbols))
        with self._lock:
            now = self._clock()
            stale = self._checked_at is None or now - self._checked_at >= self._ttl
            missing = any(symbol not in self._quotes for symbol in wanted)
            # Within the period a new request is only made for a new symbol, and not while the API is failing.
            if wanted and (stale or (missing and not self._failing)):
                self._refresh(wanted)
                self._checked_at = now
            quotes = {symbol: self._quotes[symbol] for symbol in wanted if symbol in self._quotes}
            return quotes, min((self._read_at[symbol] for symbol in quotes), default=None)

    def _refresh(self, symbols: list[str]) -> None:
        for symbol in symbols:
            try:
                self._quotes[symbol] = self._fetch(symbol)
            except (OSError, ValueError, LookupError, TypeError) as error:
                # Requests stop at the first failure: if the API does not answer, each symbol would cost another wait.
                if not self._failing:
                    logger.warning("Basket dashboard: could not read the current price of {} ({}); showing the "
                                   "last known one", symbol, error)
                self._failing = True
                return
            self._read_at[symbol] = int(self._wall_clock() * 1000)
        if self._failing:
            logger.info("Basket dashboard: reading the current price again")
        self._failing = False
