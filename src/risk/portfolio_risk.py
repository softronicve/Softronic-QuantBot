"""Portfolio risk limits: total notional, open risk and correlated positions.

``ATRSizer`` caps each position on its own; this module decides whether the
portfolio can take one more. There are three checks, in this order: total
notional value against the balance, the sum of what would be lost if every
stop were hit, and the number of positions making the same bet.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.risk.turtle_engine import RiskError

APPROVED = "APPROVED"
MAX_LEVERAGE_EXCEEDED = "MAX_LEVERAGE_EXCEEDED"
MAX_HEAT_EXCEEDED = "MAX_HEAT_EXCEEDED"
CORRELATED_EXPOSURE_BLOCKED = "CORRELATED_EXPOSURE_BLOCKED"

ASSET_CLASSES = ("forex", "crypto", "commodity")
# Relative slack when comparing with a limit: reaching it exactly is not exceeding
# it, even if the floating point sum goes over by a billionth.
TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class ActivePosition:
    """An open position, as the portfolio check sees it."""

    symbol: str
    # +1 buy, -1 short sell.
    direction: int
    # Size x price x units per lot, in the account currency. Always positive.
    nominal_value: float
    # What is lost if the stop is hit. It is 0 if the stop is already at the entry or beyond.
    risk_usd: float
    asset_class: str

    def __post_init__(self) -> None:
        _require_symbol("symbol", self.symbol)
        _require_direction("direction", self.direction)
        _require_asset_class("asset_class", self.asset_class)
        _require_positive("nominal_value", self.nominal_value)
        _require_non_negative("risk_usd", self.risk_usd)


class PortfolioRiskManager:
    """Decides whether the portfolio can take a new position.

    Two positions count as correlated when they bet the same way on the same
    currency: buying EURUSD and buying GBPUSD are two dollar sales, just like
    buying EURUSD and selling USDJPY. Buying EURUSD and buying USDJPY, on the
    other hand, offset each other and are not blocked. The rule looks at the
    currencies of each pair, not just at whether both orders are buys.
    """

    def __init__(
        self,
        max_portfolio_leverage: float = 1.0,
        max_portfolio_heat_pct: float = 4.0,
        max_correlated_positions: int = 1,
        correlated_classes: Collection[str] = ("forex",),
    ) -> None:
        """
        Args:
            max_portfolio_leverage: maximum notional value of all positions
                together, in times the balance. ``1.0`` is ten positions of 10 %:
                the whole balance allocated, with no debt.
            max_portfolio_heat_pct: maximum percentage of the balance that would
                be lost if every stop were hit at once. On daily candles this is
                the limit that binds most often: a 10 % position risks between
                0.5 % and 1 %, so 4 % fits four to seven of them.
            max_correlated_positions: positions that can share the same bet
                before the next one is blocked.
            correlated_classes: asset classes where that limit applies. By
                default only ``forex``: a crypto basket can have several coins
                open at once, limited only by total notional and total risk.
                In ``forex`` the currencies of each pair are compared; in the
                other classes, every position of the class in the same
                direction counts as the same bet.
        """
        _require_positive("max_portfolio_leverage", max_portfolio_leverage)
        _require_positive("max_portfolio_heat_pct", max_portfolio_heat_pct)
        if max_portfolio_heat_pct > 100.0:
            raise RiskError(f"max_portfolio_heat_pct cannot exceed 100, got {max_portfolio_heat_pct}")
        if isinstance(max_correlated_positions, bool) or not isinstance(max_correlated_positions, int):
            raise RiskError(f"max_correlated_positions must be an integer, got {max_correlated_positions!r}")
        if max_correlated_positions < 1:
            raise RiskError(f"max_correlated_positions must be >= 1, got {max_correlated_positions}")
        if isinstance(correlated_classes, str) or not isinstance(correlated_classes, Collection):
            raise RiskError(f"correlated_classes must be a collection of classes, got {correlated_classes!r}")
        for asset_class in correlated_classes:
            _require_asset_class("correlated_classes", asset_class)

        self._max_leverage = float(max_portfolio_leverage)
        self._max_heat_pct = float(max_portfolio_heat_pct)
        self._max_correlated = max_correlated_positions
        self._correlated_classes = frozenset(correlated_classes)

    @property
    def max_portfolio_leverage(self) -> float:
        return self._max_leverage

    @property
    def max_portfolio_heat_pct(self) -> float:
        return self._max_heat_pct

    def can_open_position(
        self,
        account_balance: float,
        current_positions: list[ActivePosition],
        new_nominal: float,
        new_risk_usd: float,
        new_symbol: str,
        new_direction: int,
        new_asset_class: str,
    ) -> tuple[bool, str]:
        """Says whether the position can be opened and, if not, which limit stops it.

        Returns:
            ``(True, "APPROVED")`` or ``(False, reason)``, where the reason is the
            first limit that is not met: ``MAX_LEVERAGE_EXCEEDED``,
            ``MAX_HEAT_EXCEEDED`` or ``CORRELATED_EXPOSURE_BLOCKED``.

        Raises:
            RiskError: if the balance, the amounts or the positions are not
                valid. Corrupt data never turns into an approval.
        """
        _require_positive("account_balance", account_balance)
        if isinstance(current_positions, (str, bytes)) or not isinstance(current_positions, Sequence):
            raise RiskError(
                f"current_positions must be a list of ActivePosition, got {type(current_positions).__name__}"
            )
        for index, position in enumerate(current_positions):
            if not isinstance(position, ActivePosition):
                raise RiskError(f"current_positions[{index}] is not an ActivePosition: {position!r}")
        # It goes through the same checks as an open position, except that a
        # new order with no declared risk cannot be evaluated.
        _require_positive("new_risk_usd", new_risk_usd)
        candidate = _candidate(new_symbol, new_direction, new_nominal, new_risk_usd, new_asset_class)

        nominal = sum(position.nominal_value for position in current_positions) + candidate.nominal_value
        leverage = nominal / account_balance
        if leverage > self._max_leverage * (1.0 + TOLERANCE):
            logger.warning(
                "Entry on {} rejected ({}): portfolio notional would reach {:.2f} "
                "({:.3f}x the balance of {:.2f}) and the maximum is {:g}x",
                candidate.symbol, MAX_LEVERAGE_EXCEEDED, nominal, leverage, account_balance, self._max_leverage,
            )
            return False, MAX_LEVERAGE_EXCEEDED

        heat = sum(position.risk_usd for position in current_positions) + candidate.risk_usd
        heat_pct = heat / account_balance * 100.0
        if heat_pct > self._max_heat_pct * (1.0 + TOLERANCE):
            logger.warning(
                "Entry on {} rejected ({}): open risk would reach {:.2f} "
                "({:.3f} % of the balance of {:.2f}) and the maximum is {:g} %",
                candidate.symbol, MAX_HEAT_EXCEEDED, heat, heat_pct, account_balance, self._max_heat_pct,
            )
            return False, MAX_HEAT_EXCEEDED

        if candidate.asset_class in self._correlated_classes:
            bets = _bets(candidate)
            shared = [
                position.symbol
                for position in current_positions
                if position.asset_class in self._correlated_classes and bets & _bets(position)
            ]
            if len(shared) >= self._max_correlated:
                logger.warning(
                    "Entry on {} rejected ({}): {} already bet{} the same way "
                    "and the maximum of correlated positions is {}",
                    candidate.symbol, CORRELATED_EXPOSURE_BLOCKED, ", ".join(shared),
                    "" if len(shared) > 1 else "s", self._max_correlated,
                )
                return False, CORRELATED_EXPOSURE_BLOCKED

        logger.info(
            "Entry on {} approved: portfolio notional {:.2f} ({:.3f}x of {:g}x) | "
            "open risk {:.2f} ({:.3f} % of {:g} %) | {} prior positions",
            candidate.symbol, nominal, leverage, self._max_leverage, heat, heat_pct, self._max_heat_pct,
            len(current_positions),
        )
        return True, APPROVED


def _candidate(symbol: Any, direction: Any, nominal: Any, risk: Any, asset_class: Any) -> ActivePosition:
    """The position to be opened, with the names of its arguments in the errors."""
    _require_symbol("new_symbol", symbol)
    _require_direction("new_direction", direction)
    _require_asset_class("new_asset_class", asset_class)
    _require_positive("new_nominal", nominal)
    return ActivePosition(symbol, direction, nominal, risk, asset_class)


def _bets(position: ActivePosition) -> frozenset[tuple[str, int]]:
    """What a position bets on: each currency of a pair with its sign, or its class and direction."""
    if position.asset_class == "forex":
        pair = _currencies(position.symbol)
        if pair is not None:
            base, quote = pair
            return frozenset({(base, position.direction), (quote, -position.direction)})
    return frozenset({(f"class:{position.asset_class}", position.direction)})


def _currencies(symbol: str) -> tuple[str, str] | None:
    """Currencies of a pair (``EURUSD``, ``eurusd.m``, ``GBPUSDm``…); ``None`` if it does not look like one."""
    letters = symbol.upper()[:6]
    if len(letters) == 6 and letters.isalpha():
        return letters[:3], letters[3:]
    return None


def _is_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(value)
    except TypeError:
        return False


def _require_positive(name: str, value: Any) -> None:
    if not _is_number(value) or value <= 0.0:
        raise RiskError(f"{name} must be a positive number, got {value!r}")


def _require_non_negative(name: str, value: Any) -> None:
    if not _is_number(value) or value < 0.0:
        raise RiskError(f"{name} must be a non-negative number, got {value!r}")


def _require_symbol(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RiskError(f"{name} must be a non-empty string, got {value!r}")


def _require_direction(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, -1):
        raise RiskError(f"{name} must be +1 (buy) or -1 (short), got {value!r}")


def _require_asset_class(name: str, value: Any) -> None:
    if not isinstance(value, str) or value not in ASSET_CLASSES:
        raise RiskError(f"{name} must be one of {ASSET_CLASSES}, got {value!r}")
