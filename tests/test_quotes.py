"""Tests for the current price of the open positions (``src/ui/quotes.py``)."""

from __future__ import annotations

import email.message
import http.client
import json
import threading
import time
import urllib.error
from typing import Any

import pytest
from loguru import logger

from src.ui import quotes
from src.ui.quotes import (
    PUBLIC_API,
    LiveBySymbol,
    LiveCandles,
    LiveFeed,
    LivePrices,
    LiveQuotes,
    Quote,
    fetch_book,
    fetch_candles,
    fetch_derivatives,
    fetch_funding,
    fetch_prices,
    fetch_quote,
    fetch_stats,
    funding_rates,
    market_stats,
)

DAY_MS = 86_400_000


class Feed:
    """A fake API: it counts what it is asked for and can stop answering."""

    def __init__(self) -> None:
        self.prices = {"AAAUSDT": 100.0, "BBBUSDT": 5.0}
        self.asked: list[str] = []
        self.down = False

    def __call__(self, symbol: str) -> Quote:
        self.asked.append(symbol)
        if self.down:
            raise TimeoutError("no answer")
        return Quote(self.prices[symbol], ((0, DAY_MS - 1, self.prices[symbol] * 0.9),))


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _live(feed: Feed, clock: Clock) -> LiveQuotes:
    return LiveQuotes("https://example.invalid", ttl_seconds=30.0, fetch=feed, clock=clock, wall_clock=clock)


def test_quotes_are_read_once_per_period() -> None:
    feed, clock = Feed(), Clock()
    live = _live(feed, clock)
    first, read_at = live.get(["AAAUSDT", "BBBUSDT"])
    assert {symbol: quote.price for symbol, quote in first.items()} == {"AAAUSDT": 100.0, "BBBUSDT": 5.0}
    assert read_at == 1_000_000 and feed.asked == ["AAAUSDT", "BBBUSDT"]

    feed.prices["AAAUSDT"] = 120.0
    clock.now += 29.0
    assert live.get(["AAAUSDT", "BBBUSDT"])[0]["AAAUSDT"].price == 100.0 and len(feed.asked) == 2
    clock.now += 1.0
    again, read_at = live.get(["AAAUSDT", "BBBUSDT"])
    assert again["AAAUSDT"].price == 120.0 and read_at == 1_030_000 and len(feed.asked) == 4


def test_no_positions_means_no_request() -> None:
    feed = Feed()
    assert _live(feed, Clock()).get([]) == ({}, None) and feed.asked == []


def test_a_new_symbol_is_read_without_waiting_for_the_period() -> None:
    feed, clock = Feed(), Clock()
    live = _live(feed, clock)
    live.get(["AAAUSDT"])
    clock.now += 5.0
    both, read_at = live.get(["AAAUSDT", "BBBUSDT"])
    assert sorted(both) == ["AAAUSDT", "BBBUSDT"] and read_at == 1_005_000


def test_a_failed_read_keeps_the_last_known_price_and_is_not_hammered() -> None:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}")
    feed, clock = Feed(), Clock()
    live = _live(feed, clock)
    try:
        live.get(["AAAUSDT", "BBBUSDT"])
        feed.down = True
        clock.now += 30.0
        known, read_at = live.get(["AAAUSDT", "BBBUSDT"])
        # What was known, with the time it was read; and after the first failure the second symbol is not asked for.
        assert known["AAAUSDT"].price == 100.0 and read_at == 1_000_000 and len(feed.asked) == 3
        clock.now += 10.0
        live.get(["AAAUSDT", "BBBUSDT"])
        assert len(feed.asked) == 3  # no retry within the period

        feed.down = False
        clock.now += 20.0
        assert live.get(["AAAUSDT", "BBBUSDT"])[1] == 1_060_000
    finally:
        logger.remove(sink)
    assert [line.split("|")[0] for line in lines] == ["WARNING", "INFO"]  # one message when it goes down and one when it comes back


def test_a_symbol_never_read_is_simply_missing() -> None:
    feed = Feed()
    feed.down = True
    assert _live(feed, Clock()).get(["AAAUSDT"]) == ({}, None)


@pytest.mark.parametrize("answer", ["[]", "{}", "not json", '[["x"]]'])
def test_an_unexpected_answer_counts_as_a_failed_read(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    monkeypatch.setattr(quotes.urllib.request, "urlopen", lambda url, timeout: _Response(answer))
    assert LiveQuotes("https://example.invalid").get(["AAAUSDT"]) == ({}, None)


class _Response:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None: ...

    def read(self) -> bytes:
        return self._body


def test_fetch_quote_reads_the_public_daily_candles(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[str, float]] = []
    rows = [[0, "1.0", "1.2", "0.9", "1.1", "5", DAY_MS - 1], [DAY_MS, "1.1", "1.5", "1.05", "1.4", "7", 2 * DAY_MS - 1]]

    def urlopen(url: str, timeout: float) -> Any:
        asked.append((url, timeout))
        return _Response(json.dumps(rows))

    monkeypatch.setattr(quotes.urllib.request, "urlopen", urlopen)
    quote = fetch_quote(PUBLIC_API["dry_run"], "1000PEPEUSDT", timeout=3.0)
    assert quote == Quote(1.4, ((0, DAY_MS - 1, 0.9), (DAY_MS, 2 * DAY_MS - 1, 1.05)))
    assert asked == [("https://fapi.binance.com/fapi/v1/klines?symbol=1000PEPEUSDT&interval=1d&limit=3", 3.0)]


def test_all_prices_come_in_one_request(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []
    rows = [{"symbol": "BTCUSDT", "price": "83723.10", "time": 1}, {"symbol": "1000PEPEUSDT", "price": "0.004215"}]

    def urlopen(url: str, timeout: float) -> Any:
        asked.append(url)
        return _Response(json.dumps(rows))

    monkeypatch.setattr(quotes.urllib.request, "urlopen", urlopen)
    assert fetch_prices(PUBLIC_API["testnet"], timeout=3.0) == {"BTCUSDT": 83_723.10, "1000PEPEUSDT": 0.004215}
    assert asked == ["https://testnet.binancefuture.com/fapi/v1/ticker/price"]


def test_prices_are_read_once_per_period_and_survive_a_failed_read() -> None:
    lines: list[str] = []
    sink = logger.add(lines.append, format="{level}|{message}")
    clock, answers = Clock(), [{"BTCUSDT": 100.0}, TimeoutError("no answer"), {"BTCUSDT": 120.0}]
    asked: list[float] = []

    def fetch() -> dict[str, float]:
        asked.append(clock.now)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    live = LivePrices("https://example.invalid", ttl_seconds=10.0, fetch=fetch, clock=clock, wall_clock=clock)
    try:
        assert live.get() == ({"BTCUSDT": 100.0}, 1_000_000)
        clock.now += 9.0
        assert live.get() == ({"BTCUSDT": 100.0}, 1_000_000) and len(asked) == 1
        clock.now += 1.0
        assert live.get() == ({"BTCUSDT": 100.0}, 1_000_000) and len(asked) == 2  # it failed: the last known
        clock.now += 5.0
        live.get()
        assert len(asked) == 2  # no retry within the period
        clock.now += 5.0
        assert live.get() == ({"BTCUSDT": 120.0}, 1_020_000)
    finally:
        logger.remove(sink)
    assert [line.split("|")[0] for line in lines] == ["WARNING", "INFO"]


def test_prices_never_read_are_simply_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quotes.urllib.request, "urlopen", lambda url, timeout: _Response("not json"))
    assert LivePrices("https://example.invalid").get() == ({}, None)


def test_market_figures_are_read_from_the_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {
        "/fapi/v1/ticker/24hr": [{"symbol": "BTCUSDT", "priceChangePercent": "-1.210", "quoteVolume": "4957000000.5"}],
        "/fapi/v1/premiumIndex": [{"symbol": "BTCUSDT", "lastFundingRate": "0.00010000"},
                                  {"symbol": "NORATEUSDT", "lastFundingRate": ""}],
        "/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=2": [
            [1_791_834_000_000, "1.0", "1.2", "0.9", "1.1", "50", 1_791_834_299_999, "55.5"],
            [1_791_834_300_000, "1.1", "1.3", "1.0", "1.25", "70", 1_791_834_599_999, "84.0"]],
    }
    monkeypatch.setattr(quotes.urllib.request, "urlopen",
                        lambda url, timeout: _Response(json.dumps(answers[url.removeprefix(PUBLIC_API["live"])])))
    assert fetch_stats(PUBLIC_API["live"], 4.0) == {"BTCUSDT": {"change_24h_pct": -1.21, "volume_24h_usd": 4_957_000_000.5}}
    assert fetch_funding(PUBLIC_API["live"], 4.0) == {"BTCUSDT": pytest.approx(0.01)}
    assert fetch_candles(PUBLIC_API["live"], "BTCUSDT", "5m", 2, 4.0) == {
        "time": [1_791_834_000, 1_791_834_300], "open": [1.0, 1.1], "high": [1.2, 1.3], "low": [0.9, 1.0],
        "close": [1.1, 1.25], "volume": [55.5, 84.0]}
    assert market_stats(PUBLIC_API["live"]).get()[0]["BTCUSDT"]["change_24h_pct"] == -1.21
    assert funding_rates(PUBLIC_API["live"]).get()[0] == {"BTCUSDT": pytest.approx(0.01)}


def test_candles_are_cached_per_symbol_and_interval_and_the_cache_is_bounded() -> None:
    clock, asked = Clock(), []

    def feed(symbol: str, interval: str) -> dict[str, list[float]]:
        asked.append((symbol, interval))
        return {"time": [len(asked)]}

    live = LiveCandles("https://example.invalid", ttl_seconds=10.0, max_entries=2, fetch=feed, clock=clock)
    assert live.get("BTCUSDT", "5m") == {"time": [1]} and live.get("BTCUSDT", "5m") == {"time": [1]}
    assert live.get("BTCUSDT", "1h") == {"time": [2]} and len(asked) == 2
    clock.now += 10.0
    assert live.get("BTCUSDT", "5m") == {"time": [3]}  # it expired: it is reread
    live.get("ETHUSDT", "5m")  # the third one evicts the oldest
    assert live.get("BTCUSDT", "1h") == {"time": [5]}


@pytest.mark.parametrize(("symbol", "interval"), [("BTCUSDT", "1s"), ("BTCUSDT", "1M"), ("BTCUSD", "5m"),
                                                  ("btcusdt", "5m"), ("BTC/USDT", "5m"), ("USDT", "5m"), ("", "")])
def test_only_usdt_contracts_and_known_intervals_are_asked(symbol: str, interval: str) -> None:
    asked: list[str] = []
    live = LiveCandles("https://example.invalid", fetch=lambda s, i: asked.append(s) or {})  # type: ignore[arg-type,return-value]
    with pytest.raises(ValueError, match="unsupported"):
        live.get(symbol, interval)
    assert asked == []


def test_a_failed_candle_read_is_remembered_briefly_and_then_retried() -> None:
    clock, answers = Clock(), [TimeoutError(), {"time": [1]}]
    asked: list[float] = []

    def feed(symbol: str, interval: str) -> dict[str, list[float]]:
        asked.append(clock.now)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    live = LiveCandles("https://example.invalid", failure_ttl_seconds=5.0, fetch=feed, clock=clock)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            live.get("BTCUSDT", "5m")
    assert len(asked) == 1  # a panel left open does not keep hitting an API that is failing
    clock.now += 5.0
    assert live.get("BTCUSDT", "5m") == {"time": [1]} and len(asked) == 2


def test_a_slow_contract_does_not_hold_up_the_others() -> None:
    release = threading.Event()

    def feed(symbol: str) -> dict[str, Any]:
        if symbol == "SLOWUSDT":
            release.wait(timeout=5)
        return {"symbol": symbol}

    live = LiveBySymbol(feed, ttl_seconds=60.0)
    slow = threading.Thread(target=live.get, args=("SLOWUSDT",))
    slow.start()
    try:
        started = time.monotonic()
        assert live.get("FASTUSDT") == {"symbol": "FASTUSDT"}
        assert time.monotonic() - started < 1.0
    finally:
        release.set()
        slow.join(timeout=5)


def _http_error(code: int, retry_after: str | None) -> urllib.error.HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://example.invalid", code, "limit", headers, None)


@pytest.mark.parametrize(("code", "retry_after", "paused_for"), [(429, "30", 30.0), (418, None, 60.0)])
def test_a_rate_limit_pauses_every_request_until_retry_after(
    monkeypatch: pytest.MonkeyPatch, code: int, retry_after: str | None, paused_for: float,
) -> None:
    clock, calls = Clock(), []
    monkeypatch.setattr(quotes, "RATE_LIMIT", quotes.RateLimitGate(clock=clock))

    def urlopen(url: str, timeout: float) -> Any:
        calls.append(url)
        if len(calls) == 1:
            raise _http_error(code, retry_after)
        return _Response("[]")

    monkeypatch.setattr(quotes.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        fetch_prices("https://example.invalid", 3.0)
    clock.now += paused_for - 1.0
    with pytest.raises(OSError, match="rate limit"):
        fetch_stats("https://example.invalid", 3.0)  # held back for every feed, not only the one that tripped
    assert len(calls) == 1
    clock.now += 1.0
    assert fetch_prices("https://example.invalid", 3.0) == {} and len(calls) == 2


def test_a_broken_http_answer_is_an_ordinary_failed_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(url: str, timeout: float) -> Any:
        raise http.client.IncompleteRead(b"[", 100)

    monkeypatch.setattr(quotes.urllib.request, "urlopen", urlopen)
    with pytest.raises(OSError, match="IncompleteRead"):
        fetch_prices("https://example.invalid", 3.0)
    assert LivePrices("https://example.invalid").get() == ({}, None)


def test_derivatives_and_book_are_read_from_the_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    base = PUBLIC_API["live"]
    answers = {
        "/futures/data/openInterestHist?symbol=NEARUSDT&period=1h&limit=25": [
            {"sumOpenInterestValue": "400.0", "CMCCirculatingSupply": "1000.0"},
            {"sumOpenInterestValue": "300.0", "CMCCirculatingSupply": "1300.0"}],
        "/futures/data/globalLongShortAccountRatio?symbol=NEARUSDT&period=1h&limit=1": [
            {"longAccount": "0.5866", "longShortRatio": "1.419", "shortAccount": "0.4134"}],
        "/futures/data/takerlongshortRatio?symbol=NEARUSDT&period=1h&limit=1": [{"buySellRatio": "1.2302"}],
        "/fapi/v1/depth?symbol=NEARUSDT&limit=20": {"bids": [["5.468", "2822"]], "asks": [["5.469", "2018"]]},
        "/fapi/v1/trades?symbol=NEARUSDT&limit=40": [
            {"time": 1_791_835_471_970, "price": "5.469", "qty": "8", "isBuyerMaker": False}],
    }
    monkeypatch.setattr(quotes.urllib.request, "urlopen",
                        lambda url, timeout: _Response(json.dumps(answers[url.removeprefix(base)])))
    assert fetch_derivatives(base, "NEARUSDT", 4.0) == {
        "open_interest_usd": 300.0, "open_interest_change_24h_pct": pytest.approx(-25.0),
        "open_interest_history": [400.0, 300.0], "long_accounts_pct": pytest.approx(58.66), "long_short_ratio": 1.419,
        "taker_buy_sell_ratio": 1.2302, "circulating_supply": 1300.0}
    assert fetch_book(base, "NEARUSDT", 3.0) == {
        "bids": [[5.468, 2822.0]], "asks": [[5.469, 2018.0]],
        "trades": [{"time": 1_791_835_471_970, "price": 5.469, "quantity": 8.0, "sell": False}]}


def test_per_contract_data_is_cached_bounded_and_only_for_usdt_contracts() -> None:
    clock, asked = Clock(), []

    def feed(symbol: str) -> dict[str, Any]:
        asked.append(symbol)
        return {"n": len(asked)}

    live = LiveBySymbol(feed, ttl_seconds=60.0, max_entries=2, clock=clock)
    assert live.get("BTCUSDT") == {"n": 1} == live.get("BTCUSDT")
    clock.now += 60.0
    assert live.get("BTCUSDT") == {"n": 2}
    live.get("ETHUSDT")
    live.get("SOLUSDT")  # the third one evicts the oldest
    assert live.get("BTCUSDT") == {"n": 5}
    for bad in ("btcusdt", "BTCUSD", "BTC/USDT", "USDT", ""):
        with pytest.raises(ValueError, match="unsupported"):
            live.get(bad)
    assert len(asked) == 5


def test_a_failed_read_can_be_retried_sooner_than_the_period() -> None:
    clock, answers = Clock(), [TimeoutError(), {"x": 1}, {"x": 2}]
    asked: list[float] = []

    def fetch() -> dict[str, Any]:
        asked.append(clock.now)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    feed = LiveFeed(fetch, what="something", ttl_seconds=3_600.0, retry_seconds=180.0, clock=clock, wall_clock=clock)
    assert feed.get() == ({}, None)
    clock.now += 179.0
    assert feed.get() == ({}, None) and len(asked) == 1
    clock.now += 1.0
    assert feed.get()[0] == {"x": 1} and len(asked) == 2
    clock.now += 3_599.0  # already read: it is valid for its whole period
    assert feed.get()[0] == {"x": 1} and len(asked) == 2
    clock.now += 1.0
    assert feed.get()[0] == {"x": 2}


def test_each_account_reads_its_own_public_api() -> None:
    assert PUBLIC_API["dry_run"] == PUBLIC_API["live"] == "https://fapi.binance.com"
    assert PUBLIC_API["testnet"] == "https://testnet.binancefuture.com"
    assert quotes.PUBLIC_STREAM["dry_run"] == quotes.PUBLIC_STREAM["live"] == "wss://fstream.binance.com"
    assert quotes.PUBLIC_STREAM["testnet"] == "wss://fstream.binancefuture.com"
