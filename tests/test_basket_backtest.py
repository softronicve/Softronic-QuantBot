"""Tests for the point-in-time basket study: data, picking the basket without looking ahead, and accounting."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.risk.portfolio_risk import PortfolioRiskManager
from src.strategies.research import basket_backtest as study
from src.strategies.research.basket_backtest import RandomEntry, Result, Universe, simulate
from src.strategies.trend_breakout import TrendBreakoutStrategy

DAY0 = int(np.datetime64("2020-01-01", "D").astype(np.int64))


def _series(closes: list[float] | np.ndarray, *, first_day: int = DAY0, volume: float | np.ndarray = 1e8,
            spread: float = 0.01) -> pd.DataFrame:
    """Hand-written daily candles: opens at the previous close and moves a ``spread`` around it."""
    close = np.asarray(closes, dtype=np.float64)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "day": np.arange(close.size, dtype=np.int64) + first_day, "open": open_,
        "high": np.maximum(open_, close) * (1 + spread), "low": np.minimum(open_, close) * (1 - spread),
        "close": close, "quote_volume": np.broadcast_to(volume, close.shape).astype(np.float64),
    })


def _date(day: int) -> str:
    return str(np.datetime64(day, "D"))


def _trend(flat: int = 300, rise: int = 40, fall: int = 25) -> np.ndarray:
    """Flat, rises 2 % a day (breaks the channel), then falls 3 % a day (loses it)."""
    up = 100.0 * 1.02 ** np.arange(1, rise + 1)
    down = up[-1] * 0.97 ** np.arange(1, fall + 1)
    return np.concatenate([np.full(flat, 100.0), up, down])


def _run(series: dict[str, pd.DataFrame], start: int, **kwargs) -> Result:
    universe = Universe(series, window=150, fixed=list(series))
    return simulate(universe, TrendBreakoutStrategy(), PortfolioRiskManager(), start=_date(start), **kwargs)


# --- Data ------------------------------------------------------------------


def test_read_daily_accepts_the_archive_formats(tmp_path: Path) -> None:
    ms = DAY0 * study.DAY_MS
    row = "{t},10,11,9,10.5,1,{c},1000,5,0,0,0\n"
    (tmp_path / "a.csv").write_text("open_time,open,high,low,close,volume,close_time,quote_volume,n,tb,tq,i\n"
                                    + row.format(t=ms, c=ms + study.DAY_MS - 1))
    with zipfile.ZipFile(tmp_path / "b.zip", "w") as archive:  # since 2025 timestamps are in microseconds
        archive.writestr("b.csv", row.format(t=(ms + study.DAY_MS) * 1000, c=(ms + 2 * study.DAY_MS) * 1000 - 1))
    candles = study.read_daily([tmp_path / "b.zip", tmp_path / "a.csv"])
    assert candles["day"].tolist() == [DAY0, DAY0 + 1]
    assert candles["quote_volume"].tolist() == [1000.0, 1000.0] and candles["close"].iat[0] == 10.5


def test_download_survives_a_dropped_connection_and_names_outside_ascii(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = {"BTCUSDT": "data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip",
            "牛来USDT": "data/spot/monthly/klines/牛来USDT/1d/牛来USDT-1d-2025-11.zip"}
    asked: list[str] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None: ...

        def read(self) -> bytes:
            return b"zip"

    def urlopen(url: str, timeout: float) -> Response:
        url.encode("ascii")  # what urllib requires: an unencoded path breaks here
        asked.append(url)
        if len(asked) == 1:
            raise ConnectionResetError(104, "Connection reset by peer")
        return Response()

    monkeypatch.setattr(study, "archived_symbols", lambda quote="USDT": list(keys))
    monkeypatch.setattr(study, "_s3_list", lambda prefix, delimiter: [keys[prefix.split("/")[4]]])
    monkeypatch.setattr(study.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(study, "DOWNLOAD_RETRY_SECONDS", 0.0)

    assert study.download(tmp_path, workers=1) == 2
    assert (tmp_path / "牛来USDT" / "牛来USDT-1d-2025-11.zip").read_bytes() == b"zip"
    assert len(asked) == 3 and study.download(tmp_path, workers=1) == 0  # what is already downloaded is not requested again


def test_a_gap_splits_a_symbol_into_separate_instruments() -> None:
    candles = pd.concat([_series([1.0] * 5), _series([50.0] * 3, first_day=DAY0 + 20)], ignore_index=True)
    parts = study.split_segments("LUNAUSDT", candles)
    assert list(parts) == ["LUNAUSDT", "LUNAUSDT#1"]
    assert len(parts["LUNAUSDT"]) == 5 and parts["LUNAUSDT#1"]["close"].iat[0] == 50.0


def test_leveraged_tokens_currencies_and_stablecoins_never_enter() -> None:
    assert study.tradable("BTCUSDT") and study.tradable("LUNAUSDT")
    for symbol in ("BTCUPUSDT", "ETHDOWNUSDT", "EURUSDT", "USDCUSDT", "PAXGUSDT", "WBTCUSDT", "BTCBUSD"):
        assert not study.tradable(symbol)


# --- Each day's basket -------------------------------------------------------


def _wiggle(n: int) -> np.ndarray:
    return 100.0 * np.cumprod(np.where(np.arange(n) % 2 == 0, 1.03, 1 / 1.03))


def test_the_basket_of_a_day_only_looks_at_earlier_days() -> None:
    surge = np.where(np.arange(200) < 100, 1e6, 1e9)  # C only has volume from day 100 on
    universe = Universe({"AUSDT": _series(_wiggle(200), volume=1e8), "BUSDT": _series(_wiggle(200), volume=5e7),
                         "CUSDT": _series(_wiggle(200), volume=surge)}, basket_size=2)
    assert universe.basket(100) == ["AUSDT", "BUSDT"]
    assert universe.basket(130) == ["CUSDT", "AUSDT"]


def test_illiquid_and_pegged_coins_stay_out() -> None:
    universe = Universe({"AUSDT": _series(_wiggle(120)), "PEGUSDT": _series(np.full(120, 1.0), volume=1e10),
                         "THINUSDT": _series(_wiggle(120), volume=1e5)})
    assert universe.basket(100) == ["AUSDT"]
    assert universe.basket(30) == []  # no basket without 60 days of history


# --- Simulation ----------------------------------------------------------------


def test_a_breakout_is_bought_at_the_close_and_sold_when_the_channel_breaks() -> None:
    closes = _trend()
    result = _run({"AAAUSDT": _series(closes)}, DAY0 + 200, cost_per_side=0.0)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade["opened"] == DAY0 + 300 and trade["entry"] == pytest.approx(closes[300])
    assert trade["reason"] == "CHANNEL" and trade["pnl"] > 0
    entry_day = pd.Timestamp(np.datetime64(DAY0 + 300, "D"))
    assert result.curve.loc[entry_day, "exposure"] == pytest.approx(0.1, rel=0.05)  # 10 % cap per coin


def test_a_coin_that_stops_trading_is_sold_at_its_last_close() -> None:
    closes = _trend(rise=30, fall=0)
    series = {"AAAUSDT": _series(closes), "BBBUSDT": _series(np.full(closes.size + 30, 100.0))}
    result = _run(series, DAY0 + 200, cost_per_side=0.0)
    assert [t["reason"] for t in result.trades] == ["DELIST"]
    assert result.trades[0]["exit"] == pytest.approx(closes[-1])


def test_a_stop_that_gaps_is_filled_at_the_open() -> None:
    closes = _trend(rise=10, fall=0)
    candles = _series(np.concatenate([closes, [50.0, 50.0]]))
    candles.loc[closes.size, ["open", "high", "low"]] = [50.0, 51.0, 49.0]
    result = _run({"AAAUSDT": candles}, DAY0 + 200, cost_per_side=0.0)
    assert result.trades[0]["reason"] == "STOP" and result.trades[0]["exit"] == 50.0


def test_without_trend_the_simulation_does_not_make_money() -> None:
    """Null test of the simulator: with driftless prices it cannot make more than chance explains."""
    finals = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        series = {}
        for k in range(4):
            steps = rng.normal(-0.5 * 0.8**2 / 8760, 0.8 / np.sqrt(8760), size=(700, 24))
            path = 100.0 * np.exp(np.cumsum(steps.ravel())).reshape(700, 24)
            open_ = np.concatenate([[100.0], path[:-1, -1]])
            series[f"S{k}USDT"] = pd.DataFrame({
                "day": np.arange(700) + DAY0, "open": open_, "high": np.maximum(path.max(axis=1), open_),
                "low": np.minimum(path.min(axis=1), open_), "close": path[:, -1], "quote_volume": 1e8})
        finals.append(_run(series, DAY0 + 160).curve["equity"].iat[-1])
    assert np.mean(finals) < 10_000.0 + 2 * np.std(finals, ddof=1) / np.sqrt(len(finals))


def test_random_entries_follow_the_requested_frequency() -> None:
    series = {"AAAUSDT": _series(_wiggle(400))}
    universe = Universe(series, window=150, fixed=list(series))
    never = simulate(universe, RandomEntry(0.0, seed=1), PortfolioRiskManager(), start=_date(DAY0 + 200))
    always = simulate(universe, RandomEntry(1.0, seed=1), PortfolioRiskManager(), start=_date(DAY0 + 200))
    assert never.trades == [] and never.curve["exposure"].max() == 0.0
    assert always.curve["exposure"].max() > 0.0


def test_idle_cash_can_earn_interest() -> None:
    days = pd.to_datetime(np.arange(366) + DAY0, unit="D")
    idle = Result(pd.DataFrame({"equity": 10_000.0, "exposure": 0.0}, index=days))
    assert idle.equity(0.04).iat[-1] == pytest.approx(10_400.0, rel=1e-3)
    assert idle.equity().iat[-1] == 10_000.0


def test_full_passive_weight_is_buy_and_hold() -> None:
    prices = pd.Series([100.0, 120.0, 90.0, 150.0], index=pd.to_datetime(np.arange(4) + DAY0, unit="D"))
    assert study.passive(prices, 1.0, cost_per_side=0.0).iat[-1] == pytest.approx(15_000.0)
    assert study.summary(study.passive(prices, 1.0, cost_per_side=0.0))["max_drawdown"] == pytest.approx(-0.25)


def test_the_command_line_runs_from_a_folder_of_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for symbol, closes in (("BTCUSDT", _trend()), ("ETHUSDT", _wiggle(365))):
        folder = tmp_path / symbol
        folder.mkdir()
        candles = _series(closes)
        ms = candles["day"] * study.DAY_MS
        pd.DataFrame({0: ms, 1: candles["open"], 2: candles["high"], 3: candles["low"], 4: candles["close"],
                      5: 1.0, 6: ms + study.DAY_MS - 1, 7: candles["quote_volume"]}).to_csv(
            folder / f"{symbol}-1d-2020.csv", header=False, index=False)
    code = study.main(["run", "--data", str(tmp_path), "--window", "150", "--start", _date(DAY0 + 200),
                       "--fixed", "BTCUSDT,ETHUSDT", "--null", "2"])
    out = capsys.readouterr().out
    assert code == 0 and "Basket:" in out and "Buy and hold BTC" in out and "Random entries" in out
