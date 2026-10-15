"""Basket orchestrator: one cycle per closed candle, from signals to proposed orders.

It runs the strategy over every symbol and allocates the portfolio risk in
three steps: first the exits, which free notional and risk; then the entry
candidates, ranked by breakout strength; and last the allocation one by one,
asking ``PortfolioRiskManager`` with the portfolio as it would stand after
each approval.

It only proposes: it returns order intents and does not talk to any broker. A
symbol with unusable data is skipped without stopping the rest of the cycle,
and an open position that cannot be evaluated is kept as it is.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from src.risk.portfolio_risk import APPROVED, ActivePosition, PortfolioRiskManager
from src.risk.turtle_engine import RiskError
from src.strategies.trend_breakout import (
    Action,
    Decision,
    InstrumentSpec,
    StrategyError,
    TrendBreakoutStrategy,
)

OPEN_LONG = "OPEN_LONG"
CLOSE_LONG = "CLOSE_LONG"
HOLD = "HOLD"
MARKET = "MARKET"

# The orchestrator's own reasons; the rest come from the strategy or the portfolio.
NO_DATA = "NO_DATA"
NO_INSTRUMENT = "NO_INSTRUMENT"
INVALID_DATA = "INVALID_DATA"
NOT_MANAGED = "NOT_MANAGED"
LONG = 1


def entry_priority(breakout_strength: float, atr_pct: float, symbol: str) -> tuple[float, float, str]:
    """Order in which the cycle allocates risk among the day's breakouts.

    The strongest first (how many ATR it closed above the high it breaks); at
    equal strength, the one with the larger ATR relative to price; and if those
    match too, by symbol. It does not depend on the basket or the dict order.
    """
    return (-breakout_strength, -atr_pct, symbol)


@dataclass(frozen=True, slots=True)
class CandidateSignal:
    """What the strategy says about a symbol with no position, before the portfolio check."""

    symbol: str
    signal: Action
    close_price: float
    # (close - entry channel high) / ATR; 0.0 if it is not an entry.
    breakout_strength: float
    # The next five are 0.0 if it is not an entry.
    nominal_value: float
    risk_usd: float
    lot_size: float
    stop_loss_price: float
    reason: str = ""
    asset_class: str = ""
    # ATR / close; 0.0 if it is not an entry. Breaks the tie between two breakouts of equal strength.
    atr_pct: float = 0.0


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A proposed order, or the record that nothing is done on that symbol."""

    symbol: str
    # "OPEN_LONG", "CLOSE_LONG" or "HOLD".
    action: str
    # Lots to buy. On a close it is 0.0: the whole position is always closed.
    volume: float
    order_type: str
    stop_loss: float | None
    reason: str


class BasketOrchestrator:
    """Turns the strategy signals over the whole basket into orders the portfolio accepts."""

    def __init__(
        self,
        risk_manager: PortfolioRiskManager,
        strategy: TrendBreakoutStrategy,
        instruments: Mapping[str, InstrumentSpec],
    ) -> None:
        """
        Args:
            risk_manager: decides whether the portfolio accepts each entry. It is
                the only one asked; the one inside the strategy plays no part.
            strategy: the entry and exit rule.
            instruments: trading rules of each symbol in the basket. A symbol
                without them cannot be sized and is skipped.
        """
        if not isinstance(risk_manager, PortfolioRiskManager):
            raise StrategyError(f"risk_manager must be a PortfolioRiskManager, got {type(risk_manager).__name__}")
        if not isinstance(strategy, TrendBreakoutStrategy):
            raise StrategyError(f"strategy must be a TrendBreakoutStrategy, got {type(strategy).__name__}")
        if not isinstance(instruments, Mapping):
            raise StrategyError(f"instruments must be a mapping keyed by symbol, got {type(instruments).__name__}")
        for symbol, spec in instruments.items():
            if not isinstance(spec, InstrumentSpec) or spec.symbol != symbol:
                raise StrategyError(f"instruments[{symbol!r}] is not the InstrumentSpec of that symbol: {spec!r}")
        self._risk_manager = risk_manager
        self._strategy = strategy
        self._instruments = dict(instruments)

    def evaluate_daily_cycle(
        self,
        market_data: dict[str, pd.DataFrame],
        active_positions: list[ActivePosition],
        account_balance: float,
    ) -> list[OrderIntent]:
        """Evaluates the basket on the last closed candle of each symbol.

        Args:
            market_data: closed candles by symbol, from oldest to newest. The
                last one of each is the one evaluated.
            active_positions: every open position in the account.
            account_balance: account balance.

        Returns:
            One intent per symbol: first the closes, then the approved buys in
            the order the risk was allocated, and last what stays as it is,
            with its reason. Whoever executes only has to act on
            ``CLOSE_LONG`` and ``OPEN_LONG``, and in that order.

        Raises:
            StrategyError: if ``market_data`` is not a mapping keyed by symbol.
            RiskError: if the balance or the positions are not valid.
        """
        if not isinstance(market_data, Mapping):
            raise StrategyError(f"market_data must be a mapping keyed by symbol, got {type(market_data).__name__}")
        if isinstance(account_balance, bool) or not _is_number(account_balance) or account_balance <= 0.0:
            raise RiskError(f"account_balance must be a positive number, got {account_balance!r}")
        if isinstance(active_positions, (str, bytes)) or not isinstance(active_positions, Sequence):
            raise RiskError(
                f"active_positions must be a list of ActivePosition, got {type(active_positions).__name__}"
            )
        for index, position in enumerate(active_positions):
            if not isinstance(position, ActivePosition):
                raise RiskError(f"active_positions[{index}] is not an ActivePosition: {position!r}")

        # Step A: exits go first, because they free notional and risk for the entries.
        closes: list[OrderIntent] = []
        holds: list[OrderIntent] = []
        projected: list[ActivePosition] = []
        verdicts: dict[str, tuple[bool, str]] = {}
        for position in active_positions:
            if position.symbol not in verdicts:
                verdicts[position.symbol] = self._exit_verdict(position, market_data, account_balance)
                close, reason = verdicts[position.symbol]
                action = CLOSE_LONG if close else HOLD
                (closes if close else holds).append(OrderIntent(position.symbol, action, 0.0, MARKET, None, reason))
            if not verdicts[position.symbol][0]:
                projected.append(position)

        # Step B: candidates among the symbols with no position, strongest breakout first.
        held = {position.symbol for position in active_positions}
        candidates = self.scan_candidates(market_data, held, account_balance)
        ranked = [candidate for candidate in candidates if candidate.signal is Action.ENTER_LONG]
        if len(ranked) > 1:
            # The strategy's buy lines come out in basket order; the allocation follows this one.
            logger.info("Breakouts in the order risk is allocated: {}", ", ".join(
                f"{candidate.symbol} {candidate.breakout_strength:+.2f} ATR" for candidate in ranked))

        # Step C: risk is allocated one by one; each approval counts for the next.
        opens: list[OrderIntent] = []
        rejected: Counter[str] = Counter()
        entries = 0
        for candidate in candidates:
            if candidate.signal is not Action.ENTER_LONG:
                holds.append(OrderIntent(candidate.symbol, HOLD, 0.0, MARKET, None, candidate.reason))
                continue
            entries += 1
            approved, verdict = self._risk_manager.can_open_position(
                account_balance, projected, candidate.nominal_value, candidate.risk_usd,
                candidate.symbol, LONG, candidate.asset_class,
            )
            if approved:
                opens.append(OrderIntent(candidate.symbol, OPEN_LONG, candidate.lot_size, MARKET,
                                         candidate.stop_loss_price, candidate.reason))
                projected.append(ActivePosition(candidate.symbol, LONG, candidate.nominal_value,
                                                candidate.risk_usd, candidate.asset_class))
            else:
                rejected[verdict] += 1
                logger.info(
                    "{}: candidate dropped ({}), breakout strength {:.2f} ATR",
                    candidate.symbol, verdict, candidate.breakout_strength,
                )
                holds.append(OrderIntent(candidate.symbol, HOLD, 0.0, MARKET, None, verdict))

        skipped = sum(intent.reason in (NO_DATA, NO_INSTRUMENT) or intent.reason.startswith(INVALID_DATA)
                      for intent in holds)
        logger.info(
            "Basket cycle: {} closes | {} entry candidates, {} approved{} | "
            "{} positions after the cycle | {} symbols skipped for their data",
            len(closes), entries, len(opens),
            "".join(f", {count} for {reason}" for reason, count in sorted(rejected.items())),
            len(projected), skipped,
        )
        return [*closes, *opens, *holds]

    def scan_candidates(
        self,
        market_data: Mapping[str, pd.DataFrame],
        held_symbols: set[str] | frozenset[str],
        account_balance: float,
    ) -> list[CandidateSignal]:
        """Signal of each symbol with no position, entries first and from strongest to weakest.

        The strength of a breakout is how many ATR the price closed above the
        high it breaks. At equal strength the one with the larger ATR relative
        to price goes first, since it broke out by a larger percentage; and if
        those match too, it is sorted by symbol, so the result depends on
        neither the dict order nor the basket order.

        It is a fixed, reproducible rule, not a measured edge: in the history
        of the crypto basket on daily candles, allocating risk in this order
        did no better than doing it at random.
        """
        candidates = [
            self._candidate(symbol, market_data[symbol], account_balance)
            for symbol in market_data
            if symbol not in held_symbols
        ]
        return sorted(
            candidates,
            key=lambda c: (c.signal is not Action.ENTER_LONG, *entry_priority(c.breakout_strength, c.atr_pct, c.symbol)),
        )

    def _exit_verdict(
        self, position: ActivePosition, market_data: Mapping[str, pd.DataFrame], account_balance: float
    ) -> tuple[bool, str]:
        """Whether the position must be closed, and the reason for closing or keeping it."""
        if position.direction != LONG:
            logger.warning("{}: short position, this strategy does not manage it; kept", position.symbol)
            return False, NOT_MANAGED
        decision = self._evaluate(position.symbol, market_data.get(position.symbol), account_balance, [position])
        if isinstance(decision, str):
            logger.warning("{}: open position not evaluated; kept", position.symbol)
            return False, decision
        return decision.action is Action.EXIT_LONG, decision.reason

    def _candidate(self, symbol: str, candles: Any, account_balance: float) -> CandidateSignal:
        decision = self._evaluate(symbol, candles, account_balance, [])
        if isinstance(decision, str):
            return CandidateSignal(symbol, Action.HOLD, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, reason=decision)
        if decision.action is not Action.ENTER_LONG or decision.stop_price is None:
            return CandidateSignal(symbol, decision.action, decision.close, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   reason=decision.reason, asset_class=decision.asset_class)
        return CandidateSignal(
            symbol=symbol,
            signal=decision.action,
            close_price=decision.close,
            breakout_strength=(decision.close - decision.entry_channel) / decision.atr,
            nominal_value=decision.nominal_value,
            risk_usd=decision.risk_amount,
            lot_size=decision.volume,
            stop_loss_price=decision.stop_price,
            reason=decision.reason,
            asset_class=decision.asset_class,
            atr_pct=decision.atr / decision.close,
        )

    def _evaluate(
        self, symbol: str, candles: Any, account_balance: float, own_position: list[ActivePosition]
    ) -> Decision | str:
        """The strategy decision or, if the symbol cannot be evaluated, the reason."""
        instrument = self._instruments.get(symbol)
        if instrument is None:
            logger.warning("{}: no trading rules, skipped", symbol)
            return NO_INSTRUMENT
        if not isinstance(candles, pd.DataFrame) or candles.empty:
            logger.warning("{}: no candles, skipped", symbol)
            return NO_DATA
        try:
            # The portfolio is asked afterwards, with every candidate already ranked.
            return self._strategy.evaluate(
                candles, instrument, account_balance=account_balance, open_positions=own_position,
                check_portfolio=False,
            )
        except (StrategyError, RiskError) as error:
            logger.warning("{}: unusable data, skipped: {}", symbol, error)
            return f"{INVALID_DATA}: {error}"
        except Exception as error:  # noqa: BLE001 - one symbol must not bring down the cycle of the whole basket
            logger.error("{}: unexpected failure while evaluating, skipped: {}: {}", symbol, type(error).__name__, error)
            return f"{INVALID_DATA}: {type(error).__name__}: {error}"


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(value)
    except TypeError:
        return False
