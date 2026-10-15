"""Position size from a fixed risk, capped by a maximum leverage.

The loss at the stop is a percentage of the balance, and the notional value of
the position never exceeds ``max_leverage`` times that balance. The smaller of
the two sizes wins, so the cap can only reduce the risk, never increase it.
The result also honors the broker's step and its minimum and maximum volumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from loguru import logger

from src.risk.turtle_engine import RiskError

# Ceiling for ``max_leverage``: a higher value is almost always a configuration
# mistake (30 for 3.0), and here it is rejected instead of traded.
MAX_LEVERAGE = 3.0

LIMITED_BY_RISK = "risk"
LIMITED_BY_LEVERAGE = "leverage"
LIMITED_BY_MAX_VOLUME = "the broker's maximum volume"


@dataclass(frozen=True, slots=True)
class SizedPosition:
    """A sized position, with its value and its risk in the account currency."""

    # Lots to trade; 0.0 if the size does not reach the broker's step or minimum.
    lots: float
    nominal_value: float
    # What is lost if the stop is hit.
    risk_amount: float
    # Distance from the entry to the stop, in price.
    stop_distance: float
    # Which limit set the size: one of the ``LIMITED_BY_*``.
    limited_by: str


class ATRSizer:
    """Sizes an order so that hitting the stop costs at most ``risk_pct`` of the balance."""

    def calculate_position_size(
        self,
        *,
        account_balance: float,
        risk_pct: float = 1.0,
        atr_val: float,
        multiplier: float = 2.0,
        contract_size: float,
        step_size: float,
        asset_price: float,
        max_leverage: float = 0.1,
        min_volume: float = 0.0,
        max_volume: float | None = None,
        tick_value: float | None = None,
        tick_size: float | None = None,
    ) -> float:
        """Lots or contracts to trade; see :meth:`size_position` for the calculation and arguments."""
        return self.size_position(
            account_balance=account_balance, risk_pct=risk_pct, atr_val=atr_val, multiplier=multiplier,
            contract_size=contract_size, step_size=step_size, asset_price=asset_price,
            max_leverage=max_leverage, min_volume=min_volume, max_volume=max_volume,
            tick_value=tick_value, tick_size=tick_size,
        ).lots

    def size_position(
        self,
        *,
        account_balance: float,
        risk_pct: float = 1.0,
        atr_val: float,
        multiplier: float = 2.0,
        contract_size: float,
        step_size: float,
        asset_price: float,
        max_leverage: float = 0.1,
        min_volume: float = 0.0,
        max_volume: float | None = None,
        tick_value: float | None = None,
        tick_size: float | None = None,
    ) -> SizedPosition:
        """Position to trade, with its lots rounded down to the broker's step.

        ::

            risk           = account_balance * risk_pct / 100
            stop_distance  = atr_val * multiplier
            rate           = tick_value / (tick_size * contract_size)     # 1 if not given
            risk_lots      = risk / (stop_distance * contract_size * rate)
            cap_lots       = account_balance * max_leverage / (asset_price * contract_size * rate)
            lots           = min(risk_lots, cap_lots, max_volume)

        ``rate`` is what one unit of the asset's quote currency is worth in the
        account currency. Without ``tick_value`` it is taken as 1: the asset
        is quoted in the account currency (XAUUSD, EURUSD or BTCUSDT with a
        balance in dollars). In a pair such as USDJPY or EURGBP the broker
        gives that rate through the tick value, and with it both the loss at
        the stop and the notional end up in the account currency.

        Args:
            account_balance: total account balance, in its currency.
            risk_pct: percentage of the balance lost if the stop is hit.
            atr_val: current ATR on the traded timeframe, in price.
            multiplier: ATRs of distance between the entry and the stop.
            contract_size: units of the asset per lot (100 000 in forex,
                100 in XAUUSD, 1 in crypto futures).
            step_size: broker's minimum lot step (0.01 in MT5, 0.001 in Binance).
            asset_price: current price of the asset, to value the position.
            max_leverage: maximum notional value of the position, in times the
                balance. Default ``0.1``: each position takes at most 10 % of
                the balance and ten fit at once without leveraging the account.
                ``1.0`` is the whole balance in one position (the maximum in
                Spot); on margin it allows up to ``MAX_LEVERAGE``.
            min_volume: minimum volume the broker accepts (``volume_min`` in
                MT5, ``minQty`` in Binance). A smaller size is not traded.
            max_volume: maximum volume per order (``volume_max``); ``None`` if
                there is no limit. If it is not a multiple of the step it is
                lowered to the previous one.
            tick_value: value of one tick in the account currency, for one lot
                (``trade_tick_value_loss`` in MT5). It moves with the exchange
                rate: read it when sizing, do not store it.
            tick_size: price move of one tick (``trade_tick_size``).
                Always goes together with ``tick_value``.

        Returns:
            The lots, as an exact multiple of ``step_size``, with their notional
            value and their loss at the stop. Lots are ``0.0`` if the size is
            not enough for a single step or falls below ``min_volume``.

        Raises:
            RiskError: (a ``ValueError``) if any argument is not a finite,
                positive number (``min_volume`` allows 0), if ``risk_pct``
                exceeds 100, if ``max_leverage`` exceeds ``MAX_LEVERAGE``, if
                ``max_volume`` is less than ``min_volume`` or if only one of
                ``tick_value`` and ``tick_size`` is given.
        """
        positive: dict[str, Any] = {
            "account_balance": account_balance,
            "risk_pct": risk_pct,
            "atr_val": atr_val,
            "multiplier": multiplier,
            "contract_size": contract_size,
            "step_size": step_size,
            "asset_price": asset_price,
            "max_leverage": max_leverage,
        }
        if max_volume is not None:
            positive["max_volume"] = max_volume
        if (tick_value is None) != (tick_size is None):
            raise RiskError(
                "tick_value and tick_size go together: "
                f"got tick_value={tick_value!r} and tick_size={tick_size!r}"
            )
        if tick_value is not None:
            positive.update(tick_value=tick_value, tick_size=tick_size)
        for name, value in positive.items():
            _require_number(name, value)
            if value <= 0.0:
                raise RiskError(f"{name} must be a positive number, got {value}")
        _require_number("min_volume", min_volume)
        if min_volume < 0.0:
            raise RiskError(f"min_volume cannot be negative, got {min_volume}")
        if risk_pct > 100.0:
            raise RiskError(f"risk_pct cannot exceed 100, got {risk_pct}")
        if max_leverage > MAX_LEVERAGE:
            raise RiskError(f"max_leverage cannot exceed {MAX_LEVERAGE:g}, got {max_leverage}")
        if max_volume is not None and max_volume < min_volume:
            raise RiskError(
                f"max_volume ({max_volume}) cannot be less than min_volume ({min_volume})"
            )

        # In Decimal: with floats, 0.07 lots can come out as 0.0699… and round down to 0.06.
        balance, contract = _exact(account_balance), _exact(contract_size)
        rate = Decimal(1)
        if tick_value is not None and tick_size is not None:
            rate = _exact(tick_value) / (_exact(tick_size) * contract)
        risk = balance * _exact(risk_pct) / Decimal(100)
        stop_distance = _exact(atr_val) * _exact(multiplier)
        # What one lot loses at the stop and what it is worth, in the account currency.
        loss_per_lot = stop_distance * contract * rate
        value_per_lot = _exact(asset_price) * contract * rate
        risk_lots = risk / loss_per_lot
        cap_lots = balance * _exact(max_leverage) / value_per_lot
        raw_lots, limit = (cap_lots, LIMITED_BY_LEVERAGE) if cap_lots < risk_lots else (risk_lots, LIMITED_BY_RISK)
        if max_volume is not None and raw_lots > _exact(max_volume):
            raw_lots, limit = _exact(max_volume), LIMITED_BY_MAX_VOLUME
        step = _exact(step_size)
        lots = (raw_lots / step).to_integral_value(rounding=ROUND_DOWN) * step
        nothing = SizedPosition(0.0, 0.0, 0.0, float(stop_distance), limit)

        if lots < step:
            if limit == LIMITED_BY_LEVERAGE:
                logger.warning(
                    "Not enough capital for the maximum leverage: {:g}x of {:.2f} USD "
                    "at price {:g} gives {:.6g} lots and the minimum step is {:g}. No position opened",
                    max_leverage, account_balance, asset_price, float(raw_lots), step_size,
                )
            elif limit == LIMITED_BY_RISK:
                logger.warning(
                    "Not enough capital for the configured risk: risking {:.2f} USD "
                    "({:g} % of {:.2f}) with the stop at {:g} gives {:.6g} lots and the minimum step is {:g}. "
                    "No position opened",
                    float(risk), risk_pct, account_balance, float(stop_distance), float(raw_lots),
                    step_size,
                )
            else:
                logger.warning(
                    "The broker's maximum volume ({:g}) is below the lot step ({:g}). "
                    "No position opened",
                    max_volume, step_size,
                )
            return nothing
        if lots < _exact(min_volume):
            logger.warning(
                "Size below the broker's minimum volume: {:g} lots (limited by {}) "
                "and the minimum is {:g}. No position opened",
                float(lots), limit, min_volume,
            )
            return nothing

        logger.debug(
            "Position size: {:g} lots, limited by {} | by risk {:.6g} "
            "({:.2f} USD, {:g} % of {:.2f}, stop at {:g}) | by leverage {:.6g} ({:g}x) | "
            "notional {:.2f} USD ({:.2f}x) | loss at the stop {:.2f} USD",
            float(lots), limit, float(risk_lots), float(risk), risk_pct, account_balance,
            float(stop_distance), float(cap_lots), max_leverage, float(lots * value_per_lot),
            float(lots * value_per_lot / balance), float(lots * loss_per_lot),
        )
        return SizedPosition(
            lots=float(lots), nominal_value=float(lots * value_per_lot),
            risk_amount=float(lots * loss_per_lot), stop_distance=float(stop_distance), limited_by=limit,
        )


def _require_number(name: str, value: Any) -> None:
    """A finite real number; neither ``bool`` nor text, even if they convert."""
    try:
        number = not isinstance(value, bool) and math.isfinite(value)
    except TypeError:
        number = False
    if not number:
        # nan and inf also end up here: the message names them as they are.
        if isinstance(value, float):
            raise RiskError(f"{name} must be a positive number, got {value}")
        raise RiskError(f"{name} must be a number, got {value!r}")


def _exact(value: float) -> Decimal:
    return Decimal(str(value))
