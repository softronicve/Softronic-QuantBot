"""Tests for the market context of the dashboard (``src/ui/context.py``): outside sources, read with distrust."""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest
from loguru import logger

from src.ui import context
from src.ui.context import (
    CALENDAR_URL,
    FEAR_GREED_URL,
    GLOBAL_URL,
    MarketContext,
    coins_in,
    fetch_calendar,
    fetch_fear_greed,
    fetch_global,
    fetch_macro,
    fetch_news,
    parse_feed,
)

GLOBAL = {"data": {"total_market_cap": {"usd": 2.8e12}, "total_volume": {"usd": 7.7e10},
                   "market_cap_change_percentage_24h_usd": -3.0, "market_cap_percentage": {"btc": 59.1, "eth": 10.9}}}
FEAR_GREED = {"data": [{"value": "69", "value_classification": "Greed"}, {"value": "61", "value_classification": "Greed"},
                       {"value": "40", "value_classification": "Fear"}]}
CALENDAR = [
    {"title": "Core CPI m/m", "country": "USD", "date": "2026-10-14T08:30:00-04:00", "impact": "High", "forecast": "0.2%", "previous": "0.3%"},
    {"title": "Unemployment Claims", "country": "USD", "date": "2026-10-15T08:30:00-04:00", "impact": "Medium", "forecast": "195K", "previous": "197K"},
    {"title": "GDP m/m", "country": "GBP", "date": "2026-10-13T02:00:00-04:00", "impact": "High", "forecast": "", "previous": ""},
    {"title": "Trade Balance", "country": "EUR", "date": "2026-10-13T05:00:00-04:00", "impact": "Medium", "forecast": "", "previous": ""},
    {"title": "Bank Holiday", "country": "USD", "date": "2026-10-12T07:00:00-04:00", "impact": "Holiday", "forecast": "", "previous": ""},
    {"title": "No date", "country": "USD", "impact": "High"},
    {"title": "Broken date", "country": "USD", "date": "yesterday", "impact": "High"},
]


def _yahoo(price: float, previous: float) -> bytes:
    return json.dumps({"chart": {"result": [{"meta": {"regularMarketPrice": price, "chartPreviousClose": previous}}]}}).encode()


def _rss(*items: str) -> bytes:
    return ("<?xml version='1.0' encoding='UTF-8'?><rss version='2.0'><channel><title>Feed</title>"
            + "".join(items) + "</channel></rss>").encode("utf-8")


def _item(title: str, link: str = "https://outlet.example/story", date: str = "Mon, 12 Oct 2026 19:40:46 +0000") -> str:
    return f"<item><title>{title}</title><link>{link}</link><pubDate>{date}</pubDate></item>"


def _web(pages: dict[str, Any]) -> Any:
    def get(url: str) -> bytes:
        for prefix, page in pages.items():
            if url.startswith(prefix):
                if isinstance(page, Exception):
                    raise page
                return page if isinstance(page, bytes) else json.dumps(page).encode("utf-8")
        raise urllib.error.URLError("no route")
    return get


# --- Each source -------------------------------------------------------------


def test_global_figures() -> None:
    assert fetch_global(_web({GLOBAL_URL: GLOBAL})) == {
        "market_cap_usd": 2.8e12, "volume_24h_usd": 7.7e10, "market_cap_change_24h_pct": -3.0,
        "btc_dominance_pct": 59.1, "eth_dominance_pct": 10.9}


def test_fear_and_greed_comes_with_its_history_oldest_first() -> None:
    assert fetch_fear_greed(_web({FEAR_GREED_URL: FEAR_GREED})) == {"value": 69, "label": "Greed", "history": [40, 61, 69]}
    unknown = {"data": [{"value": "50", "value_classification": "Something new"}]}
    assert fetch_fear_greed(_web({FEAR_GREED_URL: unknown}))["label"] == "Something new"


def test_macro_quotes_survive_one_symbol_failing() -> None:
    pages = {"https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB": _yahoo(102.6, 102.0),
             "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX": urllib.error.URLError("down")}
    quotes = fetch_macro(_web(pages))
    assert list(quotes) == ["DXY"] and quotes["DXY"]["price"] == 102.6
    assert quotes["DXY"]["change_pct"] == pytest.approx((102.6 / 102.0 - 1) * 100)
    with pytest.raises(urllib.error.URLError):
        fetch_macro(_web({}))


def test_the_calendar_keeps_what_moves_markets() -> None:
    events = fetch_calendar(_web({CALENDAR_URL: CALENDAR}))["events"]
    # High impact from any country and medium only from the US; in time order; what has no date, out.
    assert [(event["country"], event["title"], event["impact"]) for event in events] == [
        ("GBP", "GDP m/m", "High"), ("USD", "Core CPI m/m", "High"), ("USD", "Unemployment Claims", "Medium")]
    cpi = events[1]
    assert cpi["at"] == 1_791_981_000_000 and (cpi["forecast"], cpi["previous"]) == ("0.2%", "0.3%")


# --- Headlines ---------------------------------------------------------------


def test_headlines_are_plain_text_with_an_https_link() -> None:
    body = _rss(
        _item("Bitcoin <b>rises</b>   &amp; ETH   &lt;i&gt;falls&lt;/i&gt;"),
        _item("Unencrypted link", link="http://outlet.example/story"),
        _item("Trick link", link="javascript:alert(1)"),
        _item("", link="https://outlet.example/empty"),
        _item("No date", date=""),
        _item("x" * 500, link="https://outlet.example/long"),
    )
    items = parse_feed("Outlet", body)
    assert [item["title"] for item in items] == ["Bitcoin rises & ETH falls", "x" * 220]
    assert items[0] == {"title": "Bitcoin rises & ETH falls", "url": "https://outlet.example/story", "source": "Outlet",
                        "published_at": 1_791_834_046_000}


@pytest.mark.parametrize("body", [
    b"<?xml version='1.0'?><!DOCTYPE rss [<!ENTITY a 'aaaaaaaaaa'>]><rss><channel><item><title>&a;</title></item></channel></rss>",
    b"<?xml version='1.0'?><!doctype rss SYSTEM 'file:///etc/passwd'><rss/>",
    b"<rss><!ENTITY x SYSTEM 'http://example.com/x'></rss>",
])
def test_feeds_with_entity_declarations_are_not_parsed(body: bytes) -> None:
    with pytest.raises(ValueError, match="unsupported declarations"):
        parse_feed("Outlet", body)


def test_coins_are_found_by_ticker_or_by_name() -> None:
    coins = ["BTC", "ETH", "SOL", "S", "OP", "A", "1000PEPE", "NEAR", "FIL"]
    assert coins_in("The bitcoin correction holds Ethereum back", coins) == ["BTC", "ETH"]
    assert coins_in("SOL and FIL lead the gains", coins) == ["SOL", "FIL"]
    assert coins_in("Pepe soars after the announcement", coins) == ["1000PEPE"]
    # One- and two-letter tickers are not searched as words: they would match any text.
    assert coins_in("A long wait for a signal: OP is not S", coins) == []
    assert coins_in("Sonic launches its network and Optimism follows", coins) == ["S", "OP"]
    assert coins_in("Rooftop solar grows and near is not a ticker here", coins) == []
    assert coins_in("Nothing to see here", coins) == []


def test_news_come_newest_first_balanced_between_sources() -> None:
    old, new = "Mon, 12 Oct 2026 10:00:00 +0000", "Mon, 12 Oct 2026 20:00:00 +0000"
    prolific = _rss(*(_item(f"Story {i}", link=f"https://a.example/{i}", date=old) for i in range(30)))
    pages = {context.NEWS_FEEDS[0][1]: prolific,
             context.NEWS_FEEDS[1][1]: _rss(_item("The latest", link="https://b.example/1", date=new)),
             context.NEWS_FEEDS[2][1]: urllib.error.URLError("down"),
             context.NEWS_FEEDS[3][1]: b"this is not xml"}
    items = fetch_news(_web(pages))["items"]
    assert items[0]["title"] == "The latest" and len(items) == 1 + context.MAX_PER_SOURCE
    with pytest.raises(ValueError, match="no news feed"):
        fetch_news(_web({}))


def test_an_oversized_answer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None: ...

        def read(self, limit: int) -> bytes:
            return b"x" * limit

    seen: list[Any] = []
    monkeypatch.setattr(context.urllib.request, "urlopen", lambda request, timeout: seen.append(request) or Response())
    with pytest.raises(ValueError, match="too large"):
        context.http_get("https://outlet.example/feed", 5.0)
    assert seen[0].get_header("User-agent") == context.USER_AGENT


# --- All together ------------------------------------------------------------


def test_the_context_survives_sources_that_fail() -> None:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}")
    pages = {GLOBAL_URL: GLOBAL, FEAR_GREED_URL: {"data": "this is not a data list"},
             context.NEWS_FEEDS[0][1]: _rss(_item("Bitcoin hits new highs"), _item("Something else", link="https://x.example/2"))}
    coins = ["ETH"]
    try:
        market = MarketContext(lambda: coins, get=_web(pages))
        first = market.get()
        coins[:] = ["BTC", "ETH"]  # the basket changed: the headlines are tagged again without rereading them
        second = market.get()
    finally:
        logger.remove(sink)
    json.dumps(second, allow_nan=False)
    assert first["global"]["btc_dominance_pct"] == 59.1 and isinstance(first["global"]["at"], int)
    assert (first["fear_greed"], first["macro"], first["calendar"]) == (None, None, None)
    assert [item["coins"] for item in first["news"]["items"]] == [[], []]
    assert sorted(item["coins"] for item in second["news"]["items"]) == [[], ["BTC"]]
    # One warning per source that is down, and no exception toward the dashboard.
    assert sum(line.startswith("WARNING|") for line in lines) == 3


def test_a_failed_source_is_retried_soon_not_after_its_long_period() -> None:
    clock = [1_000.0]
    asked: list[str] = []
    answers: dict[str, Any] = {CALENDAR_URL: urllib.error.URLError("down")}

    def get(url: str) -> bytes:
        asked.append(url)
        return _web(answers)(url)

    market = MarketContext(lambda: [], get=get)
    for feed in market._feeds.values():
        feed._clock = lambda: clock[0]
    assert market.get()["calendar"] is None
    calendar_calls = lambda: asked.count(CALENDAR_URL)  # noqa: E731
    assert calendar_calls() == 1
    clock[0] += context.RETRY_SECONDS - 1
    market.get()
    assert calendar_calls() == 1
    answers[CALENDAR_URL] = CALENDAR
    clock[0] += 1
    assert len(market.get()["calendar"]["events"]) == 3 and calendar_calls() == 2
    clock[0] += context.RETRY_SECONDS  # it works now: it is not reread until its hour expires
    market.get()
    assert calendar_calls() == 2
