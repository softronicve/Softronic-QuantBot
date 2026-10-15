"""Market context for the basket dashboard: what happens outside the portfolio.

Market cap and dominance (CoinGecko), fear and greed (alternative.me), the
dollar and rates (Yahoo Finance), headlines (RSS feeds of crypto outlets) and
the week's economic calendar (ForexFactory). They are public sources, with no
key, that the dashboard reads at long intervals and that take no part in any
decision of the bot. They are turned off with ``BOT_FUTURES__PANEL_EXTERNAL_DATA=false``.

Nothing that arrives is trusted: responses have a size cap, XML with
entity declarations is rejected without parsing it, only the text and an
``https`` link are kept from each headline and, if a source fails, the
dashboard shows the last it got from it.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree

from src.ui.quotes import LiveFeed

USER_AGENT = "Mozilla/5.0 (compatible; softronic-quantbot-dashboard)"
MAX_BYTES = 2_000_000
MAX_NEWS = 40
# Headlines taken from each outlet, so the most prolific one does not crowd out the rest.
MAX_PER_SOURCE = 12
# A failing source is asked again after three minutes, not when its normal period expires.
RETRY_SECONDS = 180.0
MAX_TITLE = 220

GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=30"
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d"
# What is not listed on Binance: the dollar index and the US 10-year bond.
YAHOO_SYMBOLS = (("DXY", "US Dollar Index (DXY)", "DX-Y.NYB"), ("US10Y", "US 10-year yield", "^TNX"))
NEWS_FEEDS = (
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt", "https://decrypt.co/feed"),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/feed"),
)
# Names headlines use for the coins; the rest are looked up by their ticker.
COIN_NAMES = {
    "BTC": ("bitcoin",), "ETH": ("ethereum", "ether"), "SOL": ("solana",), "BNB": ("binance coin",),
    "XRP": ("ripple",), "ADA": ("cardano",), "AVAX": ("avalanche",), "DOT": ("polkadot",), "LINK": ("chainlink",),
    "LTC": ("litecoin",), "DOGE": ("dogecoin",), "TRX": ("tron",), "NEAR": ("near protocol",), "ATOM": ("cosmos",),
    "UNI": ("uniswap",), "BCH": ("bitcoin cash",), "ETC": ("ethereum classic",), "FIL": ("filecoin",),
    "ICP": ("internet computer",), "VET": ("vechain",), "SAND": ("the sandbox",), "MANA": ("decentraland",),
    "AAVE": ("aave",), "INJ": ("injective",), "RENDER": ("render",), "OP": ("optimism",), "ARB": ("arbitrum",),
    "APT": ("aptos",), "SUI": ("sui",), "TIA": ("celestia",), "STX": ("stacks",), "KAS": ("kaspa",),
    "LDO": ("lido",), "IMX": ("immutable",), "PEPE": ("pepe",), "SHIB": ("shiba",), "WIF": ("dogwifhat",),
    "FLOKI": ("floki",), "BONK": ("bonk",), "GRT": ("the graph",), "RUNE": ("thorchain",), "S": ("sonic",),
}
_DECLARATION = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


def http_get(url: str, timeout: float) -> bytes:
    """The body of ``url``, up to ``MAX_BYTES``. A larger response is rejected."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("response too large")
    return body


def fetch_global(get: Callable[[str], bytes]) -> dict[str, Any]:
    """Market cap and volume of the whole crypto market and the weight of BTC and ETH."""
    data = json.loads(get(GLOBAL_URL))["data"]
    return {
        "market_cap_usd": float(data["total_market_cap"]["usd"]),
        "volume_24h_usd": float(data["total_volume"]["usd"]),
        "market_cap_change_24h_pct": float(data["market_cap_change_percentage_24h_usd"]),
        "btc_dominance_pct": float(data["market_cap_percentage"]["btc"]),
        "eth_dominance_pct": float(data["market_cap_percentage"]["eth"]),
    }


def fetch_fear_greed(get: Callable[[str], bytes]) -> dict[str, Any]:
    """Today's fear and greed index (0 to 100) and its last 30 days, from the oldest to today's."""
    rows = json.loads(get(FEAR_GREED_URL))["data"]
    values = [int(row["value"]) for row in rows][::-1]
    label = str(rows[0]["value_classification"])
    return {"value": values[-1], "label": label, "history": values}


def fetch_macro(get: Callable[[str], bytes]) -> dict[str, Any]:
    """Last price and change against the previous close of each symbol in ``YAHOO_SYMBOLS``.

    A symbol that fails is skipped: it is enough for one to arrive.
    """
    quotes: dict[str, Any] = {}
    failure: Exception | None = None
    for key, name, symbol in YAHOO_SYMBOLS:
        try:
            meta = json.loads(get(YAHOO_URL.format(symbol=urllib.parse.quote(symbol))))["chart"]["result"][0]["meta"]
            price, previous = float(meta["regularMarketPrice"]), float(meta["chartPreviousClose"])
        except (OSError, ValueError, LookupError, TypeError) as error:
            failure = error
            continue
        quotes[key] = {"name": name, "price": price, "change_pct": (price / previous - 1.0) * 100.0 if previous else None}
    if not quotes and failure is not None:
        raise failure
    return quotes


def _text(value: str | None, limit: int = MAX_TITLE) -> str:
    """Plain text of a field of an RSS feed: no tags, on one line and with a length cap."""
    return _SPACES.sub(" ", _TAGS.sub(" ", value or "")).strip()[:limit]


def coins_in(title: str, coins: Sequence[str]) -> list[str]:
    """The basket coins a headline mentions, by ticker or by name."""
    lowered = title.lower()
    found = []
    for coin in coins:
        base = coin.removeprefix("1000")
        named = any(re.search(rf"\b{re.escape(name)}\b", lowered) for name in COIN_NAMES.get(base, ()))
        # A one- or two-letter ticker (S, OP, A) matches anything: only the name counts.
        by_ticker = len(base) >= 3 and re.search(rf"\b{re.escape(base)}\b", title)
        if named or by_ticker:
            found.append(coin)
    return found


def parse_feed(source: str, body: bytes) -> list[dict[str, Any]]:
    """The headlines of an RSS feed. XML with entity declarations is not parsed."""
    if _DECLARATION.search(body):
        raise ValueError("feed with unsupported declarations")
    items = []
    for item in ElementTree.fromstring(body).iter("item"):
        heading = item.find("title")
        # The headline may carry markup, as elements or as escaped text: only the text is kept.
        title = _text("".join(heading.itertext()) if heading is not None else "")
        link = (item.findtext("link") or "").strip()
        if not title or not link.startswith("https://") or len(link) > 500:
            continue
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        items.append({"title": title, "url": link, "source": source,
                      "published_at": int(published.timestamp() * 1000)})
    return items


def fetch_news(get: Callable[[str], bytes]) -> dict[str, Any]:
    """The most recent headlines of ``NEWS_FEEDS``, together and from newest to oldest."""
    items: list[dict[str, Any]] = []
    failure: Exception | None = None
    for source, url in NEWS_FEEDS:
        try:
            recent = sorted(parse_feed(source, get(url)), key=lambda item: -item["published_at"])
            items += recent[:MAX_PER_SOURCE]
        except (OSError, ValueError, LookupError, TypeError, ElementTree.ParseError) as error:
            failure = error
    if not items and failure is not None:
        raise ValueError(f"no news feed answered ({type(failure).__name__})") from failure
    # The same headline can come from two feeds of the same outlet.
    unique = {item["url"]: item for item in items}
    return {"items": sorted(unique.values(), key=lambda item: -item["published_at"])[:MAX_NEWS]}


def fetch_calendar(get: Callable[[str], bytes]) -> dict[str, Any]:
    """The week's economic releases that move markets: the high impact ones and the medium ones from the US."""
    events = []
    for row in json.loads(get(CALENDAR_URL)):
        impact, country = str(row.get("impact", "")), str(row.get("country", ""))
        if impact != "High" and not (impact == "Medium" and country == "USD"):
            continue
        try:
            moment = datetime.fromisoformat(str(row["date"]))
        except (KeyError, ValueError):
            continue
        events.append({
            "title": _text(str(row.get("title", "")), 120), "country": country[:8], "impact": impact,
            "at": int(moment.timestamp() * 1000), "forecast": _text(str(row.get("forecast", "")), 24),
            "previous": _text(str(row.get("previous", "")), 24),
        })
    return {"events": sorted(events, key=lambda event: event["at"])}


class MarketContext:
    """The five readings from outside Binance, each with its own period. One that fails does not drag the others down."""

    def __init__(self, coins: Callable[[], Sequence[str]], *, timeout: float = 5.0,
                 get: Callable[[str], bytes] | None = None) -> None:
        """
        Args:
            coins: tickers of the basket coins, to tag the headlines each
                time they are asked for (the basket can change between two reads).
            timeout: maximum wait per request.
            get: how a URL is fetched; for tests.
        """
        fetch = get or (lambda url: http_get(url, timeout))
        self._coins = coins

        def guarded(read: Callable[[], dict[str, Any]]) -> Callable[[], dict[str, Any]]:
            # What comes from outside can have any shape: any failure while reading it is "could not be read".
            def safely() -> dict[str, Any]:
                try:
                    return read()
                except Exception as error:
                    raise ValueError(type(error).__name__) from error
            return safely

        def feed(read: Callable[[], dict[str, Any]], what: str, ttl_seconds: float) -> LiveFeed:
            return LiveFeed(guarded(read), what=what, ttl_seconds=ttl_seconds, retry_seconds=RETRY_SECONDS)

        self._feeds = {
            "global": feed(lambda: fetch_global(fetch), "the market figures", 300.0),
            "fear_greed": feed(lambda: fetch_fear_greed(fetch), "the fear and greed index", 1_800.0),
            "macro": feed(lambda: fetch_macro(fetch), "the dollar and rates", 300.0),
            "news": feed(lambda: fetch_news(fetch), "the headlines", 600.0),
            "calendar": feed(lambda: fetch_calendar(fetch), "the economic calendar", 3_600.0),
        }

    def get(self) -> dict[str, Any]:
        """What was last known from each source; empty where it could never be read."""
        context: dict[str, Any] = {}
        for name, feed in self._feeds.items():
            value, read_at = feed.get()
            context[name] = {**value, "at": read_at} if value else None
        if context["news"]:
            coins = list(self._coins())
            context["news"]["items"] = [
                {**item, "coins": coins_in(item["title"], coins)} for item in context["news"]["items"]
            ]
        return context
