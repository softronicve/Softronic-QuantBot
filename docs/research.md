# Research log

Everything we measured on historical data before deciding what the bot trades:
what was tried, on which data, what came out and what we did with it.

It exists for two reasons. One is not to test again what was already thrown
away. The other is to know how many things were looked at before believing any
single result: with hundreds of comparisons on the same data, some will look
good by chance.

Last updated: 12 October 2026.

## How much was looked at

| Block | Comparisons |
|---|---:|
| ML filter (direction and meta-labeling) | ~30 |
| Entry/exit channel lengths and direction | 29 |
| Heikin-Ashi filter | 12 |
| Shorts | 4 |
| Twelve coins, spot and futures, hourly | 40 |
| Re-run of an outside audit | 4 |
| RSI mean reversion ("strategy B") | 13 |
| Daily regime filter | 36 |
| SuperSmoother turns | 36 |
| Time-series momentum and volatility squeeze | 420 |
| Exit rules compared | 15 |
| Breakout on daily candles | 4 |
| Portfolio limits and candidate order | 11 |
| Point-in-time basket, costs and stop slippage | 7 |
| Four risk and cash variants | 4 |
| Neighbouring parameters and three add-ons | 30 |
| Sterling Canyon, a third-party gold strategy | 4 |
| **Total** | **~695** |

A comparison is one concrete rule on one symbol or basket whose result could
have been picked had it come out well. The count is a floor: it leaves out
breakdowns by period and by coin inside a basket.

With about 695 comparisons, roughly 35 are expected to show p < 0.05 by chance
alone. The Bonferroni threshold that corrects for all of them is p < 0.00007.
**Nothing in this log reaches it.** That correction is harsh, because the tests
share data and are not independent, but it makes the point: no p-value here is
proof on its own.

No stretch of history is left unseen. Every period below was used to decide
something, so the only clean confirmation left is forward.

## What the bot ended up trading, and why

The first version traded hourly candles on single symbols, with a low-pass
filter on price and an ML filter on top. None of that survived measurement:

- **ML filter.** XGBoost for direction on BTC (AUC 0.489 walk-forward) and
  meta-labeling on BTC and gold (AUC around 0.5 with wide intervals). No
  predictive power. Removed.
- **Hourly breakout on crypto.** Across twelve coins the profit factor was 1.02
  on spot and 1.09 on futures; it did not beat holding the same average
  exposure. Shorts lost or broke even in everything measured.
- **Heikin-Ashi filter, daily regime filter (200-day EMA), SuperSmoother
  turns.** None beat vetoing at random with the same frequency.
- **RSI mean reversion with a 1.1 % take-profit.** Profit factor between 0.69
  and 0.81 over twelve variants.
- **Time-series momentum and volatility squeeze.** 420 contrasts over fifteen
  symbols, hourly and daily, with and without shorts. Nothing in favour passed
  Bonferroni; the squeeze gave nothing at all. This is where the lead came
  from: on a basket of ten coins the plain breakout on *daily* candles earned
  +0.82 R per trade.

What remains is the plainest rule of the lot, on daily candles and on a basket:

- buy when the daily close is above the high of the previous 20 candles;
- sell when it closes below the low of the previous 10;
- initial stop 2 × ATR(100) below the entry; long only.

## The daily breakout, measured

**Twelve coins, formal backtest, 200 random-entry runs per coin** (spot, 0.10 %
per side):

| Variant | Basket | Passive, same exposure | Random (median) | p-value | Trades |
|---|---|---|---|---|---:|
| Breakout with the old low-pass conditions | +32.3 % | +16.2 % | +17.4 % | 0.0075 | 323 |
| Breakout alone | +41.1 % | +21.4 % | +23.0 % | 0.002 | 419 |

Breakout alone: +1.02 R per trade, 11 of 12 coins positive, 9 above their own
passive benchmark. Taken one by one only ADA goes under p = 0.05.

Caveats that come with it: it beats the passive benchmark in 2020, 2021, 2023
and 2024, ties in 2019 and 2022 and trails in 2025 and 2026. The five best
trades out of 419 make half the profit. And those twelve coins are today's
large caps: the ones that died are not in the sample.

**One shared account and portfolio limits** (7.8 years, twelve coins):

| Max open risk | Account | Max drawdown | Sharpe |
|---|---|---|---|
| 3 % | +399 % | 19.6 % | 1.16 |
| 4 % | +552 % | 24.7 % | 1.19 |
| 6 % | +767 % | 30.1 % | 1.17 |

We chose 4 % after seeing this table, which makes it a data-driven choice, not
a prior one. Ranking candidates by breakout strength gave +552 %; a random
order gave a median of +589 % over eight runs. The ranking is kept because it
is fixed and reproducible, not because it earns more.

**Without survivorship bias.** A second study rebuilt the test with the bot's
own strategy, orchestrator and risk manager, on a basket chosen each day from
what was known that day: the twelve coins with the highest median volume over
the previous 30 days, *including the ones later delisted* (the full Binance
public archive). Spot, January 2019 to September 2026, 10,000 USDT:

| Basket (0.1 % per side) | Final | Max drawdown | Average exposure |
|---|---:|---:|---:|
| Today's twelve | 66,550 | −23 % | 20 % |
| Twelve most traded at each date | 79,038 | −19 % | 20 % |
| 49 most traded at each date | 46,572 | −48 % | 23 % |
| BTC at the same exposure | 23,716 | — | 20 % |

So the earlier result does not come from picking survivors. Widening the basket
to 49 makes it worse: more weak breakouts get in and more stops are hit.

With 0.3 % per side the point-in-time twelve end at 73,603 (29.4 % a year,
−20 % drawdown, Sharpe 1.22); with 0.5 %, at 68,535. The rule beat 20 of 20
random-entry runs. One caveat on that null test: the random runs make about
426 trades against 221 and carry more exposure (25 % against 20 %), so it is
not a like-for-like comparison, although the gap is large (median 27,206
against 73,603).

`python -m src.strategies.research.basket_backtest` reproduces this block.

**Risk variants.** Four variants were written down and frozen before measuring.
Doubling the risk (2 % per position, 20 % per coin, 8 % open risk) with idle
cash earning 4 % ends at 250,236 (51.5 % a year) with a −32 % drawdown, against
−17 % for the default settings with the same cash yield. Same Sharpe (1.35) in
both: it is not a better edge, it is more risk. The cash yield only exists on
spot; on futures the balance earns nothing and funding is paid. Fourteen of
eighteen neighbouring parameter sets still beat a fixed 40 % BTC allocation,
but the doubled-risk variant sits on a peak, its neighbours earn 40 to 45 % a
year. A Turtle-style pyramid and a BTC 200-day filter did not improve it.

The defaults in this repository stay at 1 % / 10 % / 4 %.

## A third-party strategy we checked

Sterling Canyon is a gold strategy published by a commercial library of
machine-generated strategies. Its page describes it fully enough to rebuild:
XAUUSD hourly, long and short, stop orders at the 58-candle high or low with a
40-period ADX filter, stop at 6.6 % of price, target at 4.4 %, exit after 11
candles. The page does not give the ADX threshold, so we measured four variants
(no filter, ADX above 15, 20 and 25) on four years of hourly gold, against
1,000 random-entry runs each.

| Filter | Trades (per year) | Win rate | Profit factor | Total | Random: median / p95 | p |
|---|---:|---:|---:|---:|---:|---:|
| None | 848 (212) | 51.4 % | 1.24 | +52.4 % | −10.6 % / +26.3 % | 0.005 |
| ADX > 15 | 474 (119) | 52.5 % | 1.28 | +36.6 % | −6.1 % / +23.2 % | 0.008 |
| ADX > 20 | 256 (64) | 51.6 % | 1.19 | +13.7 % | −3.1 % / +17.7 % | 0.098 |
| ADX > 25 | 133 (33) | 54.9 % | 1.48 | +18.8 % | −2.6 % / +12.8 % | 0.015 |

What we take from it:

- The breakout with a time exit does beat random entries in this sample, as a
  plain 20/10 breakout on gold already did in our own earlier tests.
- Nearly all of it comes from 2025 and 2026, with gold up 150 % over the
  sample. Through 2024 the unfiltered variant made +3.9 % and could not be
  told apart from random (p = 0.20).
- The stop and the target do nothing. Of 848 trades, 751 left on time, 92 at
  the weekend, 4 at the target and 1 at the stop. The loss of a trade is not
  bounded by its stop.
- The ADX filter does not order the results (20 is worse than both 15 and 25),
  so there is no threshold one could defend.

Our sample is shorter than theirs and sits inside it, so this is not an
independent confirmation of their backtest. It is not part of the bot.

## Choices made while looking at the data

Each of these spends some of the credibility of the history. Whatever follows
from them can only be confirmed forward.

- Trading daily candles: we got there after hourly did not work.
- The basket of twelve coins: they are today's survivors.
- The 4 % open-risk ceiling: chosen with the results table in view.
- The doubled-risk variant: the highest-return one of four, although the
  selection rule was written before measuring.

## Forward test

| Period | What ran |
|---|---|
| 8–11 October 2026 | Daily 20/10 breakout on the twelve-coin basket, DRY_RUN on Binance USDT-M Futures. Four cycles, no breakout, no trade. |
| 12–25 October 2026 | Same rule on 49 contracts (the twelve plus 37 more), as a stress test of data handling and risk limits. Six trades, all in the 37 added coins and all closed at their stop (−1 R): 9,407.19 USDT out of 10,000. None of the twelve broke out in that time. |
| From 26 October 2026 | Back to the twelve measured coins, same rule and same paper account (9,407.19 USDT, no open positions). |

The 49-contract basket had no historical measurement behind it, and the study
above suggests it should do worse than the twelve. We ran it for two weeks
because it exercises more of the code. Its six losing trades are far too few
to judge those coins; the move back to the twelve rests on the study above, not
on them. What the account does from 26 October is what counts for the
twelve-coin result.

A few weeks of dry run show that the bot works: cycles on time, orders, stops,
accounting. They do not show that the strategy makes money. In the history the
twelve-coin basket makes about 28 trades a year.

**The rule for going live with real money has not been set.** Until it is
written down here, with a date, no paper-trading result meets it.

## Limits of this log

- The numbers for the hourly work and the first daily backtest are the ones
  reported at the time; the scripts and candles behind them are not in this
  repository. The point-in-time study is fully reproducible from here.
- All spot figures use spot fees and no funding. The bot runs on futures,
  where the fee is lower and funding is paid or received.
- The comparison count for the ML filter is approximate.
