"""Research: the daily basket breakout on a point-in-time universe.

Simulates the daily cycle (``TrendBreakoutStrategy`` + ``BasketOrchestrator`` +
``PortfolioRiskManager``, the same objects ``DailyRunner`` uses) with a basket
picked each day from what was known that day: the ``basket_size`` coins with
the most USDT volume over the previous 30 days, **including the ones that
later disappeared from Binance**. This measures the rule without the
survivorship bias of a list picked today (docs/research.md, item 14).

Simulation assumptions:

* Spot. Buys and sells happen at the close of the daily candle on which the
  orchestrator decides; the fee is ``cost_per_side`` per side.
* The stop is checked against the day's low and fills at the lower of the stop
  and the open (plus ``stop_slippage`` if requested).
* The sizing balance is the realized one (cash plus the cost of the open
  positions), like the futures ``wallet_balance``.
* A position in a coin that leaves the basket is still managed until its
  exit. If the coin stops trading, it is sold at the last close.
* Each continuous run of candles of a symbol is a separate instrument: a gap
  (delisted and relisted with another asset behind it) cuts the series.
* Uninvested cash can earn ``cash_yield`` a year.

It is a study, not a live strategy: it does not talk to any broker.

    python -m src.strategies.research.basket_backtest download --out data/research/daily
    python -m src.strategies.research.basket_backtest run --data data/research/daily
    python -m src.strategies.research.basket_backtest run --data data/research/daily \\
        --risk-pct 2 --max-position 0.2 --max-heat 8 --cash-yield 0.04 --null 20
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import urllib.parse
import urllib.request
import warnings
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from loguru import logger

from src.console import use_utf8_output
from src.engine.orchestrator import CLOSE_LONG, OPEN_LONG, BasketOrchestrator
from src.risk.portfolio_risk import ActivePosition, PortfolioRiskManager
from src.strategies.trend_breakout import ChannelLevels, InstrumentSpec, TrendBreakoutStrategy

DAY_MS = 86_400_000
BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/spot/monthly/klines/"
ARCHIVE = "https://data.binance.vision/"
_MONTH = re.compile(r"-1d-(\d{4}-\d{2})\.zip$")

# Never part of the basket: leveraged tokens, fiat currencies, stablecoins and
# tokenized gold. Stablecoins missing from the list are dropped by the minimum
# volatility filter.
EXCLUDED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
EXCLUDED_SYMBOLS = frozenset({
    "EURUSDT", "GBPUSDT", "AUDUSDT", "BRLUSDT", "TRYUSDT", "PAXGUSDT", "WBTCUSDT", "USDCUSDT", "BUSDUSDT",
    "TUSDUSDT", "USDPUSDT", "PAXUSDT", "FDUSDUSDT", "DAIUSDT", "USDSUSDT", "USDSBUSDT", "AEURUSDT", "EURIUSDT",
    "XUSDUSDT", "USD1USDT", "BFUSDUSDT", "RLUSDUSDT", "USDEUSDT",
})
MIN_MEDIAN_QUOTE_VOLUME = 5_000_000.0
MIN_ANNUAL_VOLATILITY = 0.05
LIQUIDITY_DAYS = 30
PRESENCE_DAYS = 60
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_RETRY_SECONDS = 2.0


# --- Data ----------------------------------------------------------------


def _get(url: str, timeout: float) -> bytes:
    """One request to the public archive; among thousands some get cut, so it retries."""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except OSError:
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            time.sleep(DOWNLOAD_RETRY_SECONDS * (attempt + 1))
    raise AssertionError("unreachable")


def _s3_list(prefix: str, delimiter: str) -> list[str]:
    """Keys or prefixes of the public archive, page by page."""
    items: list[str] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "delimiter": delimiter}
        if token:
            params["continuation-token"] = token
        root = ElementTree.fromstring(_get(f"{BUCKET}?{urllib.parse.urlencode(params)}", timeout=60))
        token = None
        for child in root:
            name = child.tag.rsplit("}", 1)[-1]
            if name in ("CommonPrefixes", "Contents"):
                items += [sub.text for sub in child if sub.tag.rsplit("}", 1)[-1] in ("Prefix", "Key") and sub.text]
            elif name == "NextContinuationToken":
                token = (child.text or "").strip() or None
        if token is None:
            return items


def archived_symbols(quote: str = "USDT") -> list[str]:
    """Every spot pair that ever had an archive, including the ones no longer traded.

    ``exchangeInfo`` only knows the present; this listing is the only public
    source that includes the dead ones.
    """
    symbols = [p[len(PREFIX):].strip("/") for p in _s3_list(PREFIX, "/")]
    return sorted(s for s in symbols if s.endswith(quote) and "/" not in s)


def download(out_dir: Path, *, quote: str = "USDT", workers: int = 16) -> int:
    """Download the monthly daily-candle files of each pair to ``out_dir/<SYMBOL>/``.

    Files already on disk are not requested again. Returns how many were downloaded.
    """
    symbols = archived_symbols(quote)
    logger.info("{} pairs against {} in the archive, delisted included", len(symbols), quote)

    def months(symbol: str) -> list[str]:
        return [key for key in _s3_list(f"{PREFIX}{symbol}/1d/", "") if _MONTH.search(key)]

    def fetch(key: str) -> int:
        symbol = key.split("/")[4]
        target = out_dir / symbol / key.rsplit("/", 1)[-1]
        if target.exists() and target.stat().st_size > 0:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        # Some pairs have non-ASCII names: the path is encoded before requesting it.
        target.write_bytes(_get(ARCHIVE + urllib.parse.quote(key), timeout=120))
        return 1

    with ThreadPoolExecutor(workers) as pool:
        keys = [key for found in pool.map(months, symbols) for key in found]
        return sum(pool.map(fetch, keys))


def read_daily(files: Iterable[Path]) -> pd.DataFrame:
    """Daily candles from data.binance.vision files (``.zip`` or ``.csv``), by UTC day.

    Accepts the microsecond timestamps the archive uses since 2025.
    """
    parts = []
    for path in sorted(files):
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                raw = archive.read(archive.namelist()[0])
        else:
            raw = path.read_bytes()
        parts.append(pd.read_csv(io.BytesIO(raw), header=None, usecols=range(8)))
    if not parts:
        return pd.DataFrame(columns=["day", "open", "high", "low", "close", "quote_volume"])
    frame = pd.concat(parts)
    frame = frame[pd.to_numeric(frame[0], errors="coerce").notna()].astype(float)
    open_time = frame[0].to_numpy()
    open_time = np.where(open_time > 1e14, open_time // 1000, open_time)
    out = pd.DataFrame({
        "day": (open_time // DAY_MS).astype(np.int64), "open": frame[1].to_numpy(), "high": frame[2].to_numpy(),
        "low": frame[3].to_numpy(), "close": frame[4].to_numpy(), "quote_volume": frame[7].to_numpy(),
    })
    return out.drop_duplicates("day").sort_values("day").reset_index(drop=True)


def split_segments(symbol: str, candles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each continuous run of days is an instrument: ``SYM``, ``SYM#1``, ``SYM#2``..."""
    if candles.empty:
        return {}
    cut = np.concatenate([[0], np.cumsum(np.diff(candles["day"].to_numpy()) != 1)])
    return {
        symbol if k == 0 else f"{symbol}#{k}": candles[cut == k].reset_index(drop=True)
        for k in np.unique(cut)
    }


def tradable(symbol: str) -> bool:
    return symbol.endswith("USDT") and symbol not in EXCLUDED_SYMBOLS and not symbol.endswith(EXCLUDED_SUFFIXES)


def load_universe(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Every run of every pair in ``data_dir`` that can enter the basket."""
    series: dict[str, pd.DataFrame] = {}
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir() and tradable(p.name)):
        files = [*folder.glob("*.zip"), *folder.glob("*.csv")]
        series.update(split_segments(folder.name, read_daily(files)))
    if not series:
        raise ValueError(f"no usable daily candles in {data_dir}")
    return series


# --- Universe and simulation --------------------------------------------


class Universe:
    """Close and volume panels by day, and each day's basket without looking ahead."""

    def __init__(self, series: dict[str, pd.DataFrame], *, basket_size: int = 12, window: int = 500,
                 fixed: Iterable[str] | None = None) -> None:
        """
        Args:
            series: runs by name, with ``day``, ``open``, ``high``, ``low``,
                ``close`` and ``quote_volume``.
            basket_size: coins in each day's basket.
            window: continuous candles a symbol needs to decide; the bot
                asks for 500.
            fixed: if given, the basket is always this list (not picked by
                volume), like the one in ``config/settings.py``.
        """
        self.series = series
        self.names = sorted(series)
        self.basket_size, self.window = basket_size, window
        self.fixed = None if fixed is None else sorted(n for n in self.names if n.split("#")[0] in set(fixed))
        self.day0 = int(min(int(df["day"].iloc[0]) for df in series.values()))
        self.days = int(max(int(df["day"].iloc[-1]) for df in series.values())) - self.day0 + 1
        self.first = {n: int(df["day"].iloc[0]) - self.day0 for n, df in series.items()}
        self.close = np.full((self.days, len(self.names)), np.nan)
        self.volume = np.zeros((self.days, len(self.names)))
        for j, name in enumerate(self.names):
            rows = series[name]["day"].to_numpy() - self.day0
            self.close[rows, j] = series[name]["close"].to_numpy()
            self.volume[rows, j] = series[name]["quote_volume"].to_numpy()
        self._baskets: dict[int, list[str]] = {}

    def index(self, day: str) -> int:
        return int(np.datetime64(day, "D").astype(np.int64)) - self.day0

    def basket(self, t: int) -> list[str]:
        """The coins with the highest median volume over the 30 days before ``t``; nothing from ``t`` on."""
        if self.fixed is not None:
            return self.fixed
        if t not in self._baskets:
            if t < PRESENCE_DAYS:
                self._baskets[t] = []
                return []
            closes = self.close[t - PRESENCE_DAYS:t]
            present = np.isfinite(closes).all(axis=0)
            median = np.median(self.volume[t - LIQUIDITY_DAYS:t], axis=0)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # columns without data: ``present`` leaves them out
                vol = np.nanstd(np.diff(np.log(closes), axis=0), axis=0) * np.sqrt(365.0)
            ok = np.where(present & (median >= MIN_MEDIAN_QUOTE_VOLUME) & (vol >= MIN_ANNUAL_VOLATILITY))[0]
            ranked = sorted(((median[j], self.names[j]) for j in ok), reverse=True)
            self._baskets[t] = [name for _, name in ranked[:self.basket_size]]
        return self._baskets[t]

    def row(self, name: str, t: int) -> int | None:
        r = t - self.first[name]
        frame = self.series[name]
        return r if 0 <= r < len(frame) and int(frame["day"].iat[r]) == self.day0 + t else None

    def candles(self, name: str, t: int) -> pd.DataFrame | None:
        """The ``window`` candles the bot would see on day ``t``, or ``None``."""
        r = self.row(name, t)
        if r is None or r < self.window - 1:
            return None
        return self.series[name].iloc[r - self.window + 1:r + 1].rename(columns={"day": "close_time"})


@dataclass
class Result:
    # Per day: equity at the close (positions marked to market) and invested fraction.
    curve: pd.DataFrame
    trades: list[dict[str, Any]] = field(default_factory=list)

    def equity(self, cash_yield: float = 0.0) -> pd.Series:
        """Equity with uninvested cash earning ``cash_yield`` a year."""
        daily = (1.0 + cash_yield) ** (1.0 / 365.25) - 1.0
        returns = self.curve["equity"].pct_change().fillna(0.0).to_numpy(copy=True)
        returns[1:] += daily * (1.0 - self.curve["exposure"].to_numpy()[:-1])
        return pd.Series(self.curve["equity"].iat[0] * np.cumprod(1.0 + returns), index=self.curve.index)


def simulate(universe: Universe, strategy: TrendBreakoutStrategy, risk: PortfolioRiskManager, *,
             start: str = "2019-01-01", end: str | None = None, capital: float = 10_000.0,
             cost_per_side: float = 0.003, stop_slippage: float = 0.0) -> Result:
    """Walk the days from ``start`` to ``end`` with the bot's orchestrator."""
    cash = capital
    positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    rows: list[tuple[int, float, float]] = []
    last = universe.days if end is None else min(universe.days, universe.index(end) + 1)

    def sell(name: str, price: float, reason: str, t: int) -> None:
        nonlocal cash
        position = positions.pop(name)
        proceeds = position["units"] * price * (1.0 - cost_per_side)
        cash += proceeds
        trades.append({"symbol": name, "opened": position["opened"], "closed": universe.day0 + t,
                       "entry": position["entry"], "exit": price, "pnl": proceeds - position["cost"], "reason": reason})

    for t in range(max(universe.index(start), 0), last):
        for name in list(positions):
            position, r = positions[name], universe.row(name, t)
            if r is None:
                sell(name, position["last"], "DELIST", t)
                continue
            bar = universe.series[name].iloc[r]
            position["last"] = float(bar["close"])
            if bar["low"] <= position["stop"]:
                sell(name, min(position["stop"] * (1.0 - stop_slippage), float(bar["open"])), "STOP", t)

        market = {}
        for name in set(universe.basket(t)) | set(positions):
            candles = universe.candles(name, t)
            if candles is not None:
                market[name] = candles
        wallet = cash + sum(p["cost"] for p in positions.values())
        active = [ActivePosition(name, 1, p["units"] * p["last"], p["units"] * max(p["entry"] - p["stop"], 0.0), "crypto")
                  for name, p in positions.items()]
        specs = {name: InstrumentSpec(name, "crypto", step_size=1e-8) for name in market}
        intents = BasketOrchestrator(risk, strategy, specs).evaluate_daily_cycle(market, active, wallet) if market else []
        for intent in intents:
            if intent.action == CLOSE_LONG and intent.symbol in positions:
                sell(intent.symbol, float(market[intent.symbol]["close"].iat[-1]), "CHANNEL", t)
        for intent in intents:
            if intent.action == OPEN_LONG:
                price = float(market[intent.symbol]["close"].iat[-1])
                cost = intent.volume * price * (1.0 + cost_per_side)
                cash -= cost
                positions[intent.symbol] = {"units": intent.volume, "entry": price, "stop": float(intent.stop_loss),
                                            "cost": cost, "last": price, "opened": universe.day0 + t}
        value = sum(p["units"] * p["last"] for p in positions.values())
        rows.append((universe.day0 + t, cash + value, value / (cash + value)))

    curve = pd.DataFrame(rows, columns=["day", "equity", "exposure"])
    curve.index = pd.to_datetime(curve["day"], unit="D")
    return Result(curve, trades)


class RandomEntry(TrendBreakoutStrategy):
    """Random entries at the breakout frequency; exits, stop and size come from the strategy."""

    def __init__(self, probability: float, seed: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._probability = probability
        self._rng = np.random.default_rng(seed)

    def levels(self, candles: pd.DataFrame) -> ChannelLevels:
        levels = super().levels(candles)
        enters = self._rng.random() < self._probability
        entry = levels.close - (self._rng.random() + 0.01) * levels.atr if enters else levels.close + levels.atr
        return ChannelLevels(levels.close, entry, levels.exit_channel, levels.atr)


def breakout_rate(universe: Universe, strategy: TrendBreakoutStrategy, start: str, every: int = 7) -> float:
    """Fraction of basket coin-days with a breakout, sampled every ``every`` days."""
    hits = total = 0
    for t in range(universe.index(start), universe.days, every):
        for name in universe.basket(t):
            candles = universe.candles(name, t)
            if candles is not None:
                levels = TrendBreakoutStrategy.levels(strategy, candles)
                hits += levels.close > levels.entry_channel
                total += 1
    return hits / total if total else 0.0


# --- Metrics and benchmarks ---------------------------------------------


def summary(equity: pd.Series) -> dict[str, Any]:
    years = max((equity.index[-1] - equity.index[0]).days, 1) / 365.25
    growth = float(equity.iat[-1] / equity.iat[0])
    yearly = equity.resample("YE").last()
    previous = pd.concat([equity.iloc[:1], yearly.iloc[:-1]])
    returns = equity.pct_change().dropna()
    return {
        "final": float(equity.iat[-1]), "cagr": growth ** (1.0 / years) - 1.0,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(365.0)) if returns.std() > 0 else 0.0,
        "years": {int(d.year): float(v) for d, v in zip(yearly.index, yearly.to_numpy() / previous.to_numpy() - 1.0)},
    }


def passive(prices: pd.Series, weight: float, *, cost_per_side: float = 0.003, band: float = 0.2,
            cash_yield: float = 0.0, capital: float = 10_000.0) -> pd.Series:
    """Always hold ``weight`` of equity in the asset, rebalancing when it drifts by a relative ``band``."""
    daily = (1.0 + cash_yield) ** (1.0 / 365.25) - 1.0
    equity, w = capital, weight
    out = [equity]
    values = prices.to_numpy()
    for i in range(1, len(values)):
        if weight > 0.0 and abs(w - weight) > band * weight:
            equity -= abs(w - weight) * equity * cost_per_side
            w = weight
        move = values[i] / values[i - 1] - 1.0
        gain = w * move + (1.0 - w) * daily
        equity *= 1.0 + gain
        w = w * (1.0 + move) / (1.0 + gain)
        out.append(equity)
    return pd.Series(out, index=prices.index)


# --- Command line ---------------------------------------------------------


def _describe(title: str, stats: dict[str, Any]) -> str:
    years = " ".join(f"{y}: {v:+.0%}" for y, v in stats["years"].items())
    return (f"{title}: {stats['final']:,.0f} | {stats['cagr']:.1%} a year | max drawdown {stats['max_drawdown']:.1%} | "
            f"Sharpe {stats['sharpe']:.2f}\n    {years}")


def run(args: argparse.Namespace, progress: Callable[[str], None] = print) -> dict[str, Any]:
    series = load_universe(Path(args.data))
    universe = Universe(series, basket_size=args.basket, window=args.window,
                        fixed=args.fixed.split(",") if args.fixed else None)
    params = {"risk_pct": args.risk_pct, "max_leverage": args.max_position}
    risk = {"max_portfolio_heat_pct": args.max_heat}
    result = simulate(universe, TrendBreakoutStrategy(**params), PortfolioRiskManager(**risk),
                      start=args.start, end=args.end, cost_per_side=args.cost)
    equity = result.equity(args.cash_yield)
    stats = summary(equity)
    exposure = float(result.curve["exposure"].mean())
    progress(_describe("Basket", stats))
    progress(f"    {len(result.trades)} trades | mean exposure {exposure:.1%}")
    out: dict[str, Any] = {"basket": stats, "exposure": exposure, "trades": len(result.trades)}
    if "BTCUSDT" in series:
        btc = series["BTCUSDT"]
        prices = btc.set_index(pd.to_datetime(btc["day"], unit="D"))["close"].reindex(equity.index).ffill()
        out["passive_same_exposure"] = summary(passive(prices, exposure, cost_per_side=args.cost,
                                                       cash_yield=args.cash_yield))
        out["hold"] = summary(passive(prices, 1.0, cost_per_side=0.0))
        progress(_describe(f"Passive BTC at the same exposure ({exposure:.0%})", out["passive_same_exposure"]))
        progress(_describe("Buy and hold BTC", out["hold"]))
    if args.null:
        probability = breakout_rate(universe, TrendBreakoutStrategy(**params), args.start)
        real = float(result.curve["equity"].iat[-1])
        finals = []
        for seed in range(args.null):
            random = simulate(universe, RandomEntry(probability, seed, **params), PortfolioRiskManager(**risk),
                              start=args.start, end=args.end, cost_per_side=args.cost)
            finals.append(float(random.curve["equity"].iat[-1]))
        beaten = sum(real > f for f in finals)
        progress(f"Random entries: the rule beats {beaten} of {args.null} (random median {np.median(finals):,.0f}, "
                 f"no cash yield on either side)")
        out["null"] = {"beaten": beaten, "runs": args.null, "finals": finals}
    return out


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description="Daily breakout on a point-in-time basket")
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="download the daily candles of every pair, delisted included")
    get.add_argument("--out", default="data/research/daily")
    sim = commands.add_parser("run", help="simulate the daily breakout")
    sim.add_argument("--data", default="data/research/daily")
    sim.add_argument("--start", default="2019-01-01")
    sim.add_argument("--end", default=None)
    sim.add_argument("--basket", type=int, default=12, help="coins in each day's basket")
    sim.add_argument("--fixed", default=None, help="fixed comma-separated basket, instead of picking it by volume")
    sim.add_argument("--window", type=int, default=500, help="continuous candles required to decide")
    sim.add_argument("--cost", type=float, default=0.003, help="cost per side")
    sim.add_argument("--risk-pct", type=float, default=1.0)
    sim.add_argument("--max-position", type=float, default=0.1, help="max notional per coin, as a multiple of the balance")
    sim.add_argument("--max-heat", type=float, default=4.0, help="max open risk of the portfolio, in %%")
    sim.add_argument("--cash-yield", type=float, default=0.0, help="annual yield of uninvested cash")
    sim.add_argument("--null", type=int, default=0, help="simulations with random entries")
    args = parser.parse_args(argv)
    # Each candidate the portfolio rejects is a WARNING from the risk manager: over
    # thousands of simulated days it would only bury the summary.
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    if args.command == "download":
        print(f"{download(Path(args.out))} new files in {args.out}")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
