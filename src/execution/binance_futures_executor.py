"""Execution of the basket's orders on Binance USDT-M Futures.

Turns each ``OrderIntent`` from the orchestrator into exchange orders and tells
what is really in the account: balance, positions and the stop that protects
each one. Three modes:

* ``DRY_RUN``: real prices and rules, paper account. No order is sent; the
  exact JSON that would have been sent is logged.
* ``TESTNET``: real orders against the futures testnet. Its prices and its
  history are not those of the real market: it is for testing the mechanics,
  not the strategy.
* ``LIVE``: real money. Only with ``live_trading_acknowledged=True``; without
  that confirmation it falls back to ``DRY_RUN``.

Design guarantees:

* Buys only: a position opens with ``BUY`` and closes with ``SELL`` and
  ``reduceOnly``, which can never leave a short position.
* Every buy has its stop on the exchange. If the stop cannot be placed, the
  position just opened is closed.
* An order whose outcome is unknown (network drop) is not resent blindly: the
  exchange is first asked whether it arrived.
* Endpoint whitelist, all of them USDT-M futures.

Since December 2025 Binance does not accept ``STOP_MARKET`` on ``/fapi/v1/order``:
conditional orders go to ``/fapi/v1/algoOrder`` with ``triggerPrice``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from loguru import logger
from requests.exceptions import RequestException

from src.engine.orchestrator import CLOSE_LONG, OPEN_LONG, OrderIntent
from src.atomic_file import write_atomically
from src.execution.errors import BrokerError, BrokerUnavailableError, OrderRejectedError
from src.risk.portfolio_risk import ActivePosition
from src.strategies.trend_breakout import InstrumentSpec

DRY_RUN = "DRY_RUN"
TESTNET = "TESTNET"
LIVE = "LIVE"
MODES = (DRY_RUN, TESTNET, LIVE)

QUOTE_ASSET = "USDT"
ASSET_CLASS = "crypto"
STOP_MARKET = "STOP_MARKET"
# Conditional orders that protect a long, whoever placed them: a stop set by hand counts as one.
PROTECTIVE_ORDER_TYPES = frozenset({STOP_MARKET, "STOP", "TRAILING_STOP_MARKET"})
DAILY = "1d"

# Outcome of ``execute_order_intent``.
FILLED = "FILLED"
SIMULATED = "SIMULATED"
REJECTED = "REJECTED"
SKIPPED = "SKIPPED"
NO_POSITION = "NO_POSITION"
STOP_FAILED_CLOSED = "STOP_FAILED_CLOSED"
UNPROTECTED = "UNPROTECTED"
# Only part of a close was filled: the rest stays open, still under its stop.
PARTIAL_CLOSE = "PARTIAL_CLOSE"
# A position found without a stop: one was placed (STOP_RESTORED) or, failing that, it was closed.
STOP_RESTORED = "STOP_RESTORED"
CLOSED_UNPROTECTED = "CLOSED_UNPROTECTED"
# ``action`` of those results: not an intent of the orchestrator, a repair done by the executor.
PROTECT = "PROTECT"
DONE = frozenset({FILLED, SIMULATED})

# Reasons an order is not sent.
BELOW_STEP = "BELOW_STEP"
BELOW_MIN_QTY = "BELOW_MIN_QTY"
BELOW_MIN_NOTIONAL = "BELOW_MIN_NOTIONAL"
INVALID_STOP = "INVALID_STOP"
POSITION_EXISTS = "POSITION_EXISTS"
NO_RULES = "NO_RULES"

# Whitelist: the only things this module may ask Binance for.
FUTURES_ENDPOINTS = frozenset(
    {
        "futures_time",
        "futures_exchange_info",
        "futures_klines",
        "futures_mark_price",
        "futures_funding_rate",
        "futures_account",
        "futures_position_information",
        "futures_get_position_mode",
        "futures_change_leverage",
        "futures_create_order",
        "futures_get_order",
        "futures_create_algo_order",
        "futures_get_open_algo_orders",
        "futures_cancel_algo_order",
    }
)
# Binance codes that do not say whether the request was executed: internal
# error, disconnection and "no response from the server; order status unknown".
UNCERTAIN_CODES = frozenset({-1000, -1001, -1007})
# A retry fixes these: too many requests and clock out of sync.
RETRYABLE_CODES = UNCERTAIN_CODES | {-1003, -1008, -1021}
ORDER_NOT_FOUND = -2013
MAX_CLIENT_ID = 36
FILL_POLLS = 5


class FuturesApiError(BrokerError):
    """Binance answered with an error. ``code`` is the one it gave."""

    def __init__(self, message: str, code: int, uncertain: bool) -> None:
        super().__init__(message)
        self.code = code
        # It is not known whether the request was executed.
        self.uncertain = uncertain


class FuturesNetworkError(BrokerError):
    """There was no answer: the request may or may not have been executed."""

    uncertain = True


@dataclass(frozen=True, slots=True)
class MarketRules:
    """Trading rules of a perpetual contract, as ``exchangeInfo`` gives them."""

    symbol: str
    # Quantity step and minimum (LOT_SIZE and MARKET_LOT_SIZE filters).
    step_size: Decimal
    min_qty: Decimal
    # Maximum per market order; None if the exchange sets no limit.
    max_qty: Decimal | None
    # Price step (PRICE_FILTER filter).
    tick_size: Decimal
    # Minimum value of an order, in USDT (MIN_NOTIONAL filter).
    min_notional: Decimal

    def round_quantity(self, quantity: float | Decimal) -> Decimal:
        """The quantity, rounded down to a multiple of the step: it never buys too much."""
        return _floor(quantity, self.step_size)

    def round_price(self, price: float | Decimal) -> Decimal:
        """The price, rounded down to a multiple of the tick."""
        return _floor(price, self.tick_size)

    def to_spec(self) -> InstrumentSpec:
        """What the strategy needs from these rules for sizing."""
        return InstrumentSpec(
            symbol=self.symbol, asset_class=ASSET_CLASS, step_size=float(self.step_size),
            min_volume=float(self.min_qty), max_volume=None if self.max_qty is None else float(self.max_qty),
        )


@dataclass(frozen=True, slots=True)
class FuturesAccount:
    """The money in the futures account, in USDT."""

    # Realized balance: what was deposited plus what was won or lost in closed
    # trades. It is the base the size of the positions was measured with.
    wallet_balance: float
    # Free collateral to open new positions.
    available_balance: float
    # PnL of the open positions at mark price.
    unrealized_pnl: float


@dataclass(frozen=True, slots=True)
class FuturesPosition:
    """An open position on the exchange (or in the paper account)."""

    symbol: str
    # Contracts; negative if the position is short.
    quantity: float
    entry_price: float
    mark_price: float
    # Stop that protects it on the exchange; None if it has none.
    stop_price: float | None


class PaperFuturesAccount:
    """Paper futures account for ``DRY_RUN``, saved in a JSON file.

    It keeps the balance, the positions with their stop and the closed trades,
    so the basket's paper trading survives a restart. Buys and closes are
    booked at the mark price of the moment, with a fee. Between one cycle and
    the next, ``settle`` applies what would have happened on the exchange: the
    stop, if any daily candle touched it, and the funding of every eight hours.

    Known limit: the stop is checked against the low of the daily candles,
    including the one of the day of the buy. If the cycle ran hours late, that
    low may have happened before the buy and the account will still count it
    as a stop.
    """

    def __init__(self, path: Path, capital: float, fee_rate: float = 0.0005) -> None:
        if not capital > 0.0:
            raise ValueError(f"the paper capital must be positive, got {capital}")
        if not 0.0 <= fee_rate < 0.1:
            raise ValueError(f"fee_rate must be between 0 and 0.1, got {fee_rate}")
        self._path = path
        self._fee_rate = fee_rate
        self._state: dict[str, Any] = {"balance": float(capital), "positions": {}, "trades": []}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                self._state = {
                    "balance": float(loaded["balance"]),
                    "positions": dict(loaded["positions"]),
                    "trades": list(loaded["trades"]),
                }
            except (ValueError, KeyError, TypeError) as error:
                # An unreadable file is not replaced by a new account: the
                # paper positions would be lost without anyone noticing.
                raise BrokerUnavailableError(f"the paper account {path} cannot be read: {error}") from error

    @property
    def balance(self) -> float:
        return float(self._state["balance"])

    @property
    def positions(self) -> dict[str, dict[str, Any]]:
        return {symbol: dict(position) for symbol, position in self._state["positions"].items()}

    @property
    def trades(self) -> list[dict[str, Any]]:
        return [dict(trade) for trade in self._state["trades"]]

    def open(self, symbol: str, quantity: float, price: float, stop_price: float, when_ms: int) -> None:
        if symbol in self._state["positions"]:
            raise OrderRejectedError(f"the paper account already has a position in {symbol}")
        fee = quantity * price * self._fee_rate
        self._state["balance"] -= fee
        self._state["positions"][symbol] = {
            "quantity": quantity, "entry_price": price, "stop_price": stop_price, "opened_at": when_ms,
            # How far the stop and the funding have already been checked.
            "checked_until": when_ms, "funded_until": when_ms, "fees": fee, "funding": 0.0,
        }
        self._save()

    def close(self, symbol: str, price: float, reason: str, when_ms: int) -> dict[str, Any] | None:
        position = self._state["positions"].pop(symbol, None)
        if position is None:
            return None
        quantity = position["quantity"]
        fee = quantity * price * self._fee_rate
        gross = quantity * (price - position["entry_price"])
        self._state["balance"] += gross - fee
        trade = {
            "symbol": symbol, "quantity": quantity, "entry_price": position["entry_price"], "exit_price": price,
            "stop_price": position["stop_price"], "opened_at": position["opened_at"], "closed_at": when_ms,
            "reason": reason, "fees": position["fees"] + fee, "funding": position["funding"],
            "pnl": gross - position["fees"] - fee - position["funding"],
        }
        self._state["trades"].append(trade)
        self._save()
        return trade

    def settle(
        self,
        market_data: Mapping[str, pd.DataFrame],
        funding_rates: Callable[[str, int, int], list[dict[str, Any]]] | None,
        now_ms: int,
    ) -> list[dict[str, Any]]:
        """Applies stops and funding since the last check; returns the positions that were stopped out.

        Args:
            market_data: closed daily candles per symbol, with ``open_time``,
                ``close_time``, ``open`` and ``low``.
            funding_rates: ``(symbol, from_ms, to_ms)`` -> funding payments
                of that span, with ``fundingTime``, ``fundingRate`` and
                ``markPrice``. If it fails, the funding stays pending for
                the next check.
            now_ms: current instant.
        """
        stopped: list[dict[str, Any]] = []
        for symbol in list(self._state["positions"]):
            position = self._state["positions"][symbol]
            candles = market_data.get(symbol)
            hit: tuple[float, int] | None = None
            if candles is not None and len(candles):
                pending = candles[candles["close_time"] > position["checked_until"]]
                for row in pending.itertuples(index=False):
                    if row.low <= position["stop_price"]:
                        # A candle after the buy that already opens below is filled at its open.
                        gapped = row.open_time > position["opened_at"]
                        price = min(position["stop_price"], row.open) if gapped else position["stop_price"]
                        hit = (float(price), max(int(row.open_time), int(position["opened_at"])))
                        break
                if len(pending):
                    position["checked_until"] = int(pending["close_time"].iloc[-1])
            until = hit[1] if hit else now_ms
            if funding_rates is not None and until > position["funded_until"]:
                try:
                    payments = funding_rates(symbol, int(position["funded_until"]) + 1, until)
                except BrokerError as error:
                    logger.warning("{}: paper funding pending, could not be queried: {}", symbol, error)
                else:
                    charged = sum(
                        position["quantity"] * float(payment["markPrice"] or position["entry_price"])
                        * float(payment["fundingRate"])
                        for payment in payments
                    )
                    position["funding"] += charged
                    position["funded_until"] = until
                    self._state["balance"] -= charged
            if hit:
                trade = self.close(symbol, hit[0], "STOP", hit[1])
                logger.warning(
                    "DRY_RUN | {}: paper stop at {:g}, PnL {:+.2f} USDT", symbol, hit[0], trade["pnl"],
                )
                stopped.append(trade)
        self._save()
        return stopped

    def _save(self) -> None:
        write_atomically(self._path, json.dumps(self._state, indent=2))


class BinanceFuturesExecutor:
    """Account, rules and orders of Binance USDT-M Futures for the basket's daily cycle."""

    def __init__(
        self,
        *,
        mode: str = DRY_RUN,
        live_trading_acknowledged: bool = False,
        api_key: str | None = None,
        api_secret: str | None = None,
        paper: PaperFuturesAccount | None = None,
        leverage: int = 1,
        stop_working_type: str = "CONTRACT_PRICE",
        retries: int = 3,
        retry_wait_seconds: float = 2.0,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """
        Args:
            mode: ``DRY_RUN``, ``TESTNET`` or ``LIVE``.
            live_trading_acknowledged: explicit confirmation that trading with
                real money is wanted. Without it, ``LIVE`` stays in ``DRY_RUN``.
            api_key, api_secret: futures credentials; required outside
                ``DRY_RUN``.
            paper: paper account; required in ``DRY_RUN``.
            leverage: leverage set on the exchange before opening in a
                symbol. With 1 there is no debt.
            stop_working_type: price that triggers the stop, ``CONTRACT_PRICE``
                (the one in the candles) or ``MARK_PRICE``.
            retries: retries of a request after a transient failure.
            retry_wait_seconds: wait before the first retry; it doubles on
                each one.
            client: ``python-binance`` client already built (for tests).
            sleep, clock: for tests.
        """
        if mode not in MODES:
            raise BrokerUnavailableError(f"unknown execution mode: {mode!r}; must be one of {MODES}")
        # Only a literal True counts: a "no" or a 1 that arrived by mistake is not a confirmation.
        if mode == LIVE and live_trading_acknowledged is not True:
            logger.warning(
                "LIVE mode requested without live_trading_acknowledged: no real money is traded, staying in DRY_RUN"
            )
            mode = DRY_RUN
        if isinstance(leverage, bool) or not isinstance(leverage, int) or leverage < 1:
            raise BrokerUnavailableError(f"leverage must be an integer >= 1, got {leverage!r}")
        if mode == DRY_RUN and paper is None:
            raise BrokerUnavailableError("DRY_RUN mode needs a paper account (paper)")
        self._mode = mode
        self._paper = paper if mode == DRY_RUN else None
        self._leverage = leverage
        self._stop_working_type = stop_working_type
        self._retries = max(0, retries)
        self._retry_wait = retry_wait_seconds
        self._sleep = sleep
        self._clock = clock
        self._rules: dict[str, MarketRules] = {}
        self._leveraged: set[str] = set()
        # Symbols whose last exchangeInfo said they can no longer be traded (e.g. SETTLING before a delisting).
        self._retired: set[str] = set()
        self._account_checked = False
        if client is not None:
            self._client = client
        elif mode == DRY_RUN:
            # Public data only: in this mode no credential is sent.
            self._client = self._connect(Client)
        else:
            if not (api_key and api_secret):
                raise BrokerUnavailableError(
                    f"{mode} mode requires BOT_FUTURES__API_KEY and BOT_FUTURES__API_SECRET"
                )
            self._client = self._connect(lambda: Client(api_key, api_secret, testnet=mode == TESTNET))
        logger.info("Binance Futures ready in mode {}", self.describe_mode())

    # --- Mode -------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_simulated(self) -> bool:
        return self._mode == DRY_RUN

    def describe_mode(self) -> str:
        return {
            DRY_RUN: "DRY_RUN (paper account, no order is sent)",
            TESTNET: "TESTNET (real orders, play money)",
            LIVE: "LIVE (real money)",
        }[self._mode]

    # --- API access -------------------------------------------------------

    @staticmethod
    def _connect(build: Callable[[], Any]) -> Any:
        try:
            return build()
        except BinanceAPIException as error:
            raise BrokerError(f"Binance refused the connection: {error.message}") from error
        except (BinanceRequestException, RequestException) as error:
            raise BrokerError(f"could not reach Binance: {error}") from error

    def _once(self, endpoint: str, **parameters: Any) -> Any:
        """A single request, with its failures translated."""
        if endpoint not in FUTURES_ENDPOINTS:
            raise RuntimeError(f"endpoint not allowed (USDT-M futures only): {endpoint}")
        try:
            return getattr(self._client, endpoint)(**parameters)
        except BinanceAPIException as error:
            status = getattr(error, "status_code", 0) or 0
            raise FuturesApiError(
                f"Binance rejected {endpoint}: [{error.code}] {error.message}", int(error.code),
                uncertain=status >= 500 or error.code in UNCERTAIN_CODES,
            ) from error
        except (BinanceRequestException, RequestException) as error:
            raise FuturesNetworkError(f"could not reach Binance ({endpoint}): {error}") from error

    def _read(self, endpoint: str, **parameters: Any) -> Any:
        """A query, which can be repeated safely: it is retried on transient failures."""
        wait = self._retry_wait
        for attempt in range(self._retries + 1):
            try:
                return self._once(endpoint, **parameters)
            except BrokerError as error:
                transient = isinstance(error, FuturesNetworkError) or error.uncertain or error.code in RETRYABLE_CODES
                if not transient or attempt == self._retries:
                    raise
                logger.warning("{} (attempt {} of {}); retrying in {:g} s", error, attempt + 1,
                               self._retries + 1, wait)
                self._sleep(wait)
                wait *= 2.0
        raise AssertionError("unreachable")

    # --- Market data ------------------------------------------------------

    def server_time_ms(self) -> int:
        return int(self._read("futures_time")["serverTime"])

    def fetch_daily_klines(self, symbol: str, limit: int) -> list[list[Any]]:
        """The last ``limit`` daily candles of the contract, the last one maybe still forming."""
        return self._read("futures_klines", symbol=symbol, interval=DAILY, limit=limit)

    def mark_price(self, symbol: str) -> float:
        return float(self._read("futures_mark_price", symbol=symbol)["markPrice"])

    def funding_rates(self, symbol: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        """Funding payments of the contract between two instants, both included."""
        return self._read("futures_funding_rate", symbol=symbol, startTime=start_ms, endTime=end_ms, limit=1000)

    def load_market_rules(self, symbols: Iterable[str], refresh: bool = False) -> dict[str, MarketRules]:
        """Trading rules of each symbol; they are downloaded once and kept.

        A symbol that does not exist, is not trading or is not a USDT perpetual
        is skipped with a warning: without rules nothing can be sized or sent.

        Args:
            refresh: download them again even if known. The daily cycle asks for
                this: the process runs for months, and Binance changes steps and
                minimums, and moves contracts to ``SETTLING`` before delisting them.
        """
        wanted = list(symbols)
        if refresh or any(symbol not in self._rules for symbol in wanted):
            listed = {entry["symbol"]: entry for entry in self._read("futures_exchange_info")["symbols"]}
            for symbol in wanted:
                if symbol in self._rules and not refresh:
                    continue
                entry = listed.get(symbol)
                problem = _untradable(entry)
                if problem:
                    self._retired.add(symbol)
                    logger.warning("{}: no trading rules, {}", symbol, problem)
                    continue
                try:
                    self._rules[symbol] = _parse_rules(entry)
                except (KeyError, ValueError, ArithmeticError) as error:
                    self._retired.add(symbol)
                    logger.warning("{}: unreadable trading rules ({}); skipped", symbol, error)
                    continue
                self._retired.discard(symbol)
        # The last known rules of a retired symbol are kept (see ``known_rules``) but no longer offered.
        return {symbol: self._rules[symbol] for symbol in wanted
                if symbol in self._rules and symbol not in self._retired}

    def known_rules(self, symbol: str) -> MarketRules | None:
        """The last rules seen for ``symbol``, even if it can no longer be traded.

        A held position in a contract on its way to delisting still needs them:
        its exit has to keep being checked, and its stop restored if missing.
        """
        return self._rules.get(symbol)

    # --- Reconciliation ---------------------------------------------------

    def settle_paper(self, market_data: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
        """In ``DRY_RUN``, applies the pending stops and funding to the paper account."""
        if self._paper is None:
            return []
        return self._paper.settle(market_data, self.funding_rates, self._now_ms())

    def get_account(self) -> FuturesAccount:
        if self._paper is not None:
            return self._paper_account(self._paper_positions())
        self._check_account_mode()
        account = self._read("futures_account")
        for asset in account.get("assets", []):
            if asset.get("asset") == QUOTE_ASSET:
                return FuturesAccount(
                    float(asset["walletBalance"]), float(asset["availableBalance"]), float(asset["unrealizedProfit"]),
                )
        raise BrokerError(f"the futures account does not report its {QUOTE_ASSET} balance")

    def get_account_balance(self) -> float:
        """Free collateral in USDT, available as margin for new positions."""
        return self.get_account().available_balance

    def get_account_and_positions(self) -> tuple[FuturesAccount, list[FuturesPosition]]:
        """The account and its positions at once, to show them together.

        In the paper account they come from a single price read, so the
        unrealized PnL of the account is exactly the sum of that of its
        positions. On the exchange they are two queries in a row.
        """
        if self._paper is not None:
            positions = self._paper_positions()
            return self._paper_account(positions), positions
        return self.get_account(), self.get_position_details()

    def held_symbols(self) -> list[str]:
        """Symbols with a live position, without prices or stops: tells which candles to load besides the basket's."""
        if self._paper is not None:
            return sorted(self._paper.positions)
        rows = self._read("futures_position_information")
        return sorted(row["symbol"] for row in rows if float(row["positionAmt"]) != 0.0)

    def get_position_details(self) -> list[FuturesPosition]:
        """Live positions, with the stop they have placed on the exchange."""
        if self._paper is not None:
            return self._paper_positions()
        self._check_account_mode()
        rows = [row for row in self._read("futures_position_information") if float(row["positionAmt"]) != 0.0]
        if not rows:
            return []
        stops = self._open_stops()
        positions = []
        for row in rows:
            quantity = float(row["positionAmt"])
            triggers = [stop["trigger"] for stop in stops.get(row["symbol"], []) if stop["side"] == "SELL"]
            positions.append(FuturesPosition(
                symbol=row["symbol"], quantity=quantity, entry_price=float(row["entryPrice"]),
                mark_price=float(row["markPrice"]),
                # With several stops the highest one rules: it is the first to trigger.
                stop_price=max(triggers) if triggers and quantity > 0.0 else None,
            ))
        return positions

    def get_open_positions(
        self, fallback_stop_distances: Mapping[str, float] | None = None
    ) -> list[ActivePosition]:
        """Live positions on the exchange, as the portfolio control needs them.

        The notional is contracts times mark price. The risk is what would be
        lost from the entry if the stop placed on the exchange triggered (0 if
        the stop is already at the entry or above), the same as in the
        simulations the portfolio limits were set with.

        Args:
            fallback_stop_distances: protective distance per symbol (2 ATR)
                for a position that has no stop on the exchange. Without a stop
                and without that distance the whole notional counts as risk.
        """
        fallback = fallback_stop_distances or {}
        active = []
        for position in self.get_position_details():
            contracts = abs(position.quantity)
            nominal = contracts * position.mark_price
            if position.quantity > 0.0 and position.stop_price is not None:
                risk = contracts * max(position.entry_price - position.stop_price, 0.0)
            elif position.symbol in fallback:
                risk = contracts * float(fallback[position.symbol])
                logger.warning(
                    "{}: position without a stop on the exchange; risk estimated from the protective distance ({:g})",
                    position.symbol, fallback[position.symbol],
                )
            else:
                risk = nominal
                logger.warning(
                    "{}: position without a stop on the exchange and without a protective distance; "
                    "its whole notional counts as risk ({:.2f})", position.symbol, nominal,
                )
            active.append(ActivePosition(
                position.symbol, 1 if position.quantity > 0.0 else -1, nominal, risk, ASSET_CLASS,
            ))
        return active

    def protect_open_positions(self, stop_distances: Mapping[str, float], cycle_id: str) -> list[dict[str, Any]]:
        """Puts a stop back on every long position that has none on the exchange.

        A crash between a fill and its stop order, or a stop cancelled by hand,
        would otherwise leave the position bare for good: later cycles see the
        symbol as held and never open it again. The stop goes at entry minus the
        protective distance (2 ATR), as for an entry. If the price is already
        below that level, or the exchange refuses the stop, the position is
        closed, as a fresh entry whose stop fails is. Without a known distance
        nothing is sent and the position is reported as unprotected.

        Returns:
            One result per position that needed attention (``action`` is
            ``PROTECT``). Always empty in ``DRY_RUN``, where every simulated
            position carries its stop.
        """
        if self._paper is not None:
            return []
        self._check_account_mode()
        rows = [row for row in self._read("futures_position_information") if float(row["positionAmt"]) > 0.0]
        if not rows:
            return []
        stops = self._open_stops(types=PROTECTIVE_ORDER_TYPES)
        results = []
        for row in rows:
            symbol = row["symbol"]
            if any(stop["side"] == "SELL" for stop in stops.get(symbol, [])):
                continue
            try:
                results.append(self._restore_stop(symbol, float(row["positionAmt"]), float(row["entryPrice"]),
                                                  stop_distances.get(symbol), cycle_id))
            except BrokerError as error:
                # One position the exchange cannot answer about must not hide what was done for the others.
                logger.critical("{}: POSITION WITHOUT STOP; it could not be fixed ({})", symbol, error)
                results.append({"symbol": symbol, "action": PROTECT, "status": UNPROTECTED, "mode": self._mode,
                                "payloads": [], "quantity": float(row["positionAmt"]), "reason": str(error)})
        return results

    def _restore_stop(
        self, symbol: str, quantity: float, entry: float, distance: float | None, cycle: str,
    ) -> dict[str, Any]:
        def result(status: str, **fields: Any) -> dict[str, Any]:
            return {"symbol": symbol, "action": PROTECT, "status": status, "mode": self._mode, "payloads": [],
                    "quantity": quantity, **fields}

        logger.error("{}: {:g} units without a stop on the exchange", symbol, quantity)
        rules = self.known_rules(symbol)
        if rules is None or distance is None or not distance > 0.0:
            logger.critical("{}: POSITION WITHOUT STOP and no data to work one out; check it by hand", symbol)
            return result(UNPROTECTED, reason="no rules or no candles to work out the stop")
        stop = rules.round_price(entry - distance)
        price = Decimal(str(self.mark_price(symbol)))
        payloads: list[dict[str, Any]] = []
        if 0 < stop < price:
            protect = self._stop_payload(symbol, rules.round_quantity(quantity), stop, cycle, kind="r")
            payloads.append(protect)
            try:
                order = self._submit_stop(protect)
            except Exception as error:  # noqa: BLE001 - any failure leads to the close below
                cause = f"the stop could not be placed: {error}"
            else:
                logger.warning("{}: stop repuesto en {}", symbol, protect["triggerPrice"])
                return result(STOP_RESTORED, stop_price=float(stop), stop_order_id=order.get("algoId"),
                              payloads=payloads)
        elif stop <= 0:
            cause = f"the protective distance ({distance:g}) is larger than the entry price"
        else:
            cause = f"the price {price:f} is already below the stop {stop:f}"
        logger.error("{}: {}; the position is closed", symbol, cause)
        close = self._close_payload(symbol, quantity, _client_id(cycle, symbol, "u"))
        payloads.append(close)
        try:
            filled, average = self._submit_market(close, lambda: quantity - max(self._position_amount(symbol), 0.0))
        except BrokerError as error:
            logger.critical("{}: POSITION WITHOUT STOP. It could not be closed either ({}); check it by hand", symbol, error)
            return result(UNPROTECTED, reason=cause, payloads=payloads)
        return result(CLOSED_UNPROTECTED, reason=cause, quantity=filled, price=average, payloads=payloads)

    # --- Orders -----------------------------------------------------------

    def execute_order_intent(self, intent: OrderIntent, cycle_id: str | None = None) -> dict[str, Any]:
        """Executes an intent from the orchestrator and reports what happened.

        Args:
            intent: the proposed order. A ``HOLD`` does nothing.
            cycle_id: identifies the cycle (the date). With it, the order
                identifiers are always the same for that cycle and symbol,
                which lets the exchange be asked about a doubtful order.

        Returns:
            A dictionary with ``status`` (``FILLED``, ``SIMULATED``,
            ``REJECTED``, ``NO_POSITION``, ``SKIPPED``, ``STOP_FAILED_CLOSED`` or
            ``UNPROTECTED``), the reason if it was not executed, the quantities
            and prices already rounded and the bodies sent (``payloads``).

        Raises:
            BrokerError: if the exchange fails and the order could not be completed.
        """
        if not isinstance(intent, OrderIntent):
            raise OrderRejectedError(f"expected an OrderIntent, got {type(intent).__name__}")
        cycle = cycle_id or time.strftime("%Y%m%d%H%M%S", time.gmtime(self._clock()))
        if intent.action == CLOSE_LONG:
            return self._close_long(intent, cycle)
        if intent.action == OPEN_LONG:
            return self._open_long(intent, cycle)
        return self._result(intent, SKIPPED, reason=intent.reason)

    def _open_long(self, intent: OrderIntent, cycle: str) -> dict[str, Any]:
        symbol = intent.symbol
        rules = self.load_market_rules([symbol]).get(symbol)
        if rules is None:
            return self._refuse(intent, NO_RULES, "there are no trading rules for the symbol")
        quantity = rules.round_quantity(intent.volume)
        if rules.max_qty is not None and quantity > rules.max_qty:
            quantity = rules.round_quantity(rules.max_qty)
        if quantity <= 0:
            return self._refuse(intent, BELOW_STEP, f"{intent.volume:g} is below the step {rules.step_size}")
        if quantity < rules.min_qty:
            return self._refuse(intent, BELOW_MIN_QTY, f"{quantity} is less than the minimum {rules.min_qty}")
        price = Decimal(str(self.mark_price(symbol)))
        notional = quantity * price
        if notional < rules.min_notional:
            return self._refuse(
                intent, BELOW_MIN_NOTIONAL,
                f"{quantity} at {price:f} is worth {notional:.2f} USDT and the minimum per order is {rules.min_notional:f}",
            )
        if intent.stop_loss is None or not intent.stop_loss > 0.0:
            return self._refuse(intent, INVALID_STOP, "the entry has no positive stop")
        stop = rules.round_price(intent.stop_loss)
        if stop <= 0 or stop >= price:
            return self._refuse(intent, INVALID_STOP, f"the stop {stop:f} is not below the price {price:f}")
        if self._has_position(symbol):
            return self._refuse(intent, POSITION_EXISTS, "there is already an open position in the symbol")

        buy = {
            "symbol": symbol, "side": "BUY", "type": "MARKET", "quantity": _text(quantity),
            "newOrderRespType": "RESULT", "newClientOrderId": _client_id(cycle, symbol, "o"),
        }
        if self._paper is not None:
            protect = self._stop_payload(symbol, quantity, stop, cycle)
            self._log_dry("POST /fapi/v1/order", buy)
            self._log_dry("POST /fapi/v1/algoOrder", protect)
            self._paper.open(symbol, float(quantity), float(price), float(stop), self._now_ms())
            return self._result(intent, SIMULATED, quantity=float(quantity), price=float(price),
                                stop_price=float(stop), payloads=[buy, protect])

        self._prepare_symbol(symbol)
        filled, average = self._submit_market(buy, lambda: max(self._position_amount(symbol), 0.0))
        if filled <= 0.0:
            raise BrokerError(f"{symbol}: the buy was not filled")
        logger.info("{}: bought {:g} units at {:g}", symbol, filled, average)
        protect = self._stop_payload(symbol, rules.round_quantity(filled), stop, cycle)
        try:
            stop_order = self._submit_stop(protect)
        except Exception as error:  # noqa: BLE001 - whatever stops the stop, the position must not stay bare
            return self._unprotected(intent, symbol, filled, average, cycle, [buy, protect], error)
        logger.info("{}: stop placed at {} (order {})", symbol, protect["triggerPrice"], stop_order.get("algoId"))
        return self._result(intent, FILLED, quantity=filled, price=average, stop_price=float(stop),
                            stop_order_id=stop_order.get("algoId"), payloads=[buy, protect])

    def _unprotected(
        self, intent: OrderIntent, symbol: str, filled: float, average: float, cycle: str,
        payloads: list[dict[str, Any]], cause: Exception,
    ) -> dict[str, Any]:
        """The stop could not be placed: the buy is undone rather than left unprotected."""
        logger.error("{}: could not place the stop ({}); closing the position just opened", symbol, cause)
        undo = self._close_payload(symbol, filled, _client_id(cycle, symbol, "x"))
        try:
            self._submit_market(undo, lambda: filled - max(self._position_amount(symbol), 0.0))
        except BrokerError as error:
            logger.critical(
                "{}: POSITION WITHOUT STOP. Neither the stop nor the emergency close went through ({}). "
                "There are {:g} units open without protection: check it by hand", symbol, error, filled,
            )
            return self._result(intent, UNPROTECTED, reason=str(cause), quantity=filled, price=average,
                                payloads=[*payloads, undo])
        return self._result(intent, STOP_FAILED_CLOSED, reason=str(cause), quantity=filled, price=average,
                            payloads=[*payloads, undo])

    def _close_long(self, intent: OrderIntent, cycle: str) -> dict[str, Any]:
        symbol = intent.symbol
        if self._paper is not None:
            position = self._paper.positions.get(symbol)
            if position is None:
                return self._result(intent, NO_POSITION, reason="there is no position to close")
            price = self.mark_price(symbol)
            sell = self._close_payload(symbol, position["quantity"], _client_id(cycle, symbol, "c"))
            self._log_dry("POST /fapi/v1/order", sell)
            self._log_dry("DELETE /fapi/v1/algoOrder", {"symbol": symbol, "type": STOP_MARKET})
            trade = self._paper.close(symbol, price, intent.reason or "CLOSE_LONG", self._now_ms())
            return self._result(intent, SIMULATED, quantity=position["quantity"], price=price,
                                pnl=trade["pnl"], payloads=[sell])

        amount = self._position_amount(symbol)
        if amount <= 0.0:
            # Without a position (the stop may have triggered) nothing is sold, but no stops are left hanging.
            cancelled = self._cancel_stops(symbol)
            return self._result(intent, NO_POSITION, reason="there is no long position to close",
                                cancelled_stops=cancelled)
        sell = self._close_payload(symbol, amount, _client_id(cycle, symbol, "c"))
        filled, average = self._submit_market(sell, lambda: amount - max(self._position_amount(symbol), 0.0))
        logger.info("{}: closed {:g} units at {:g}", symbol, filled, average)
        left = max(self._position_amount(symbol), 0.0)
        if left > 0.0:
            # The stop is reduceOnly: left in place it covers exactly what is still open.
            logger.warning("{}: the close filled {:g} of {:g}; {:g} stay open with their stop", symbol, filled,
                           amount, left)
            return self._result(intent, PARTIAL_CLOSE, reason=f"{left:g} left unsold", quantity=filled,
                                price=average, payloads=[sell])
        # After selling, never before: if the sell fails, the position keeps its stop.
        cancelled = self._cancel_stops(symbol)
        return self._result(intent, FILLED, quantity=filled, price=average, cancelled_stops=cancelled,
                            payloads=[sell])

    def _close_payload(self, symbol: str, quantity: float, client_id: str) -> dict[str, Any]:
        # No padding zeros: "12264.0" on a contract with step 1 exceeds its precision and Binance rejects it.
        return {
            "symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": _text(Decimal(str(quantity)).normalize()),
            "reduceOnly": "true", "newOrderRespType": "RESULT", "newClientOrderId": client_id,
        }

    def _stop_payload(
        self, symbol: str, quantity: Decimal, stop: Decimal, cycle: str, kind: str = "s",
    ) -> dict[str, Any]:
        return {
            "algoType": "CONDITIONAL", "symbol": symbol, "side": "SELL", "type": STOP_MARKET,
            "quantity": _text(quantity), "triggerPrice": _text(stop), "reduceOnly": "true",
            "workingType": self._stop_working_type, "clientAlgoId": _client_id(cycle, symbol, kind),
        }

    def _submit_market(self, payload: dict[str, Any], already_filled: Callable[[], float]) -> tuple[float, float]:
        """Sends a market order and returns what was filled and its average price.

        If no answer arrives, the order is looked up by its identifier and, if
        the exchange does not know it, the position is checked (``already_filled``)
        before resending: that way a network drop does not end in a double buy.
        """
        symbol, client_id = payload["symbol"], payload["newClientOrderId"]
        wait = self._retry_wait
        for attempt in range(self._retries + 1):
            try:
                order = self._once("futures_create_order", **payload)
            except BrokerError as error:
                if _not_executed_and_retryable(error) and attempt < self._retries:
                    logger.warning("{}: order {} not accepted for now ({}); sending it again in {:g} s", symbol, client_id,
                                   error, wait)
                    self._sleep(wait)
                    wait *= 2.0
                    continue
                if not getattr(error, "uncertain", False):
                    raise OrderRejectedError(str(error)) from error
                logger.warning("{}: order {} got no answer ({}); checking whether it arrived", symbol, client_id, error)
                self._sleep(wait)
                wait *= 2.0
                order = self._find_order(symbol, client_id)
                if order is None:
                    done = already_filled()
                    if done > 0.0:
                        logger.warning("{}: order {} did get filled ({:g} units)", symbol, client_id, done)
                        # For a sell with no answer there is no price left to query.
                        return done, self._entry_price(symbol) if payload["side"] == "BUY" else 0.0
                    if attempt == self._retries:
                        raise BrokerError(f"{symbol}: order {client_id} did not arrive in {attempt + 1} attempts") from error
                    continue
            return self._await_fill(symbol, client_id, order)
        raise AssertionError("unreachable")

    def _await_fill(self, symbol: str, client_id: str, order: dict[str, Any]) -> tuple[float, float]:
        for _ in range(FILL_POLLS):
            if order.get("status") not in ("NEW", "PARTIALLY_FILLED"):
                break
            self._sleep(self._retry_wait)
            order = self._read("futures_get_order", symbol=symbol, origClientOrderId=client_id)
        filled = float(order.get("executedQty") or 0.0)
        if order.get("status") != FILLED:
            logger.warning("{}: order {} ended as {} with {:g} units filled", symbol, client_id,
                           order.get("status"), filled)
        return filled, float(order.get("avgPrice") or 0.0)

    def _find_order(self, symbol: str, client_id: str) -> dict[str, Any] | None:
        try:
            return self._read("futures_get_order", symbol=symbol, origClientOrderId=client_id)
        except FuturesApiError as error:
            if error.code == ORDER_NOT_FOUND:
                return None
            raise

    def _submit_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        symbol, client_id = payload["symbol"], payload["clientAlgoId"]
        wait = self._retry_wait
        for attempt in range(self._retries + 1):
            try:
                return self._once("futures_create_algo_order", **payload)
            except BrokerError as error:
                if _not_executed_and_retryable(error) and attempt < self._retries:
                    logger.warning("{}: stop {} not accepted for now ({}); sending it again in {:g} s", symbol, client_id,
                                   error, wait)
                    self._sleep(wait)
                    wait *= 2.0
                    continue
                if not getattr(error, "uncertain", False):
                    raise
                logger.warning("{}: stop {} got no answer ({}); checking whether it arrived", symbol, client_id, error)
                self._sleep(wait)
                wait *= 2.0
                placed = [stop for stop in self._open_stops(symbol).get(symbol, []) if stop["client_id"] == client_id]
                if placed:
                    return {"algoId": placed[0]["id"], "clientAlgoId": client_id}
                if attempt == self._retries:
                    raise
        raise AssertionError("unreachable")

    def _open_stops(
        self, symbol: str | None = None, types: frozenset[str] = frozenset({STOP_MARKET}),
    ) -> dict[str, list[dict[str, Any]]]:
        """Live conditional stops, per symbol."""
        parameters = {"symbol": symbol} if symbol else {}
        stops: dict[str, list[dict[str, Any]]] = {}
        for order in self._read("futures_get_open_algo_orders", **parameters):
            if (order.get("orderType") or order.get("type")) not in types:
                continue
            stops.setdefault(order["symbol"], []).append({
                "id": order.get("algoId"), "client_id": order.get("clientAlgoId"), "side": order.get("side"),
                "trigger": float(order.get("triggerPrice") or 0.0),
            })
        return stops

    def _cancel_stops(self, symbol: str) -> int:
        """Cancels the sell stops of the symbol. A failure here does not undo the close: it warns."""
        cancelled = 0
        try:
            for stop in self._open_stops(symbol).get(symbol, []):
                if stop["side"] != "SELL":
                    continue
                self._read("futures_cancel_algo_order", symbol=symbol, algoId=stop["id"])
                cancelled += 1
        except BrokerError as error:
            logger.warning("{}: some stop was left uncancelled ({}); it is reduceOnly and cannot open a short",
                           symbol, error)
        return cancelled

    def _has_position(self, symbol: str) -> bool:
        if self._paper is not None:
            return symbol in self._paper.positions
        return self._position_amount(symbol) != 0.0

    def _position_amount(self, symbol: str) -> float:
        rows = self._read("futures_position_information", symbol=symbol)
        return sum(float(row["positionAmt"]) for row in rows if row["symbol"] == symbol)

    def _entry_price(self, symbol: str) -> float:
        rows = self._read("futures_position_information", symbol=symbol)
        return next((float(row["entryPrice"]) for row in rows if row["symbol"] == symbol), 0.0)

    def _check_account_mode(self) -> None:
        """The account must be in one-way mode: in hedge mode ``reduceOnly`` does not exist."""
        if self._account_checked:
            return
        if self._read("futures_get_position_mode").get("dualSidePosition"):
            raise BrokerUnavailableError(
                "the futures account is in hedge mode; this module only trades in one-way "
                "mode. Change it on Binance before starting"
            )
        self._account_checked = True

    def _prepare_symbol(self, symbol: str) -> None:
        self._check_account_mode()
        if symbol in self._leveraged:
            return
        try:
            self._read("futures_change_leverage", symbol=symbol, leverage=self._leverage)
        except FuturesApiError as error:
            raise OrderRejectedError(f"{symbol}: could not set the leverage to {self._leverage}x: {error}") from error
        self._leveraged.add(symbol)

    # --- Helpers ----------------------------------------------------------

    def _paper_account(self, positions: list[FuturesPosition]) -> FuturesAccount:
        assert self._paper is not None
        unrealized = sum(p.quantity * (p.mark_price - p.entry_price) for p in positions)
        margin = sum(p.quantity * p.mark_price for p in positions) / self._leverage
        balance = self._paper.balance
        return FuturesAccount(balance, balance + unrealized - margin, unrealized)

    def _paper_positions(self) -> list[FuturesPosition]:
        assert self._paper is not None
        return [
            FuturesPosition(symbol, position["quantity"], position["entry_price"], self.mark_price(symbol),
                            position["stop_price"])
            for symbol, position in self._paper.positions.items()
        ]

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _log_dry(self, request: str, payload: dict[str, Any]) -> None:
        logger.info("DRY_RUN | {} | {}", request, json.dumps(payload, ensure_ascii=False))

    def _refuse(self, intent: OrderIntent, reason: str, detail: str) -> dict[str, Any]:
        logger.warning("{}: order not sent ({}): {}", intent.symbol, reason, detail)
        return self._result(intent, REJECTED, reason=reason, detail=detail)

    def _result(self, intent: OrderIntent, status: str, **fields: Any) -> dict[str, Any]:
        return {"symbol": intent.symbol, "action": intent.action, "status": status, "mode": self._mode,
                "payloads": [], **fields}


def _not_executed_and_retryable(error: BrokerError) -> bool:
    """Binance refused the request without executing it (clock skew, rate limit, overload): resending is safe."""
    return isinstance(error, FuturesApiError) and not error.uncertain and error.code in RETRYABLE_CODES


def _floor(value: float | Decimal, step: Decimal) -> Decimal:
    """``value`` rounded down to a multiple of ``step``, in exact decimal."""
    exact = value if isinstance(value, Decimal) else Decimal(str(value))
    return (exact / step).to_integral_value(rounding=ROUND_DOWN) * step


def _text(value: Decimal) -> str:
    """A decimal as the API expects it: no scientific notation."""
    return format(value, "f")


def _client_id(cycle: str, symbol: str, kind: str) -> str:
    """Our own order identifier: the same for the same cycle, symbol and kind of order."""
    return f"tb-{cycle.replace('-', '')}-{symbol}-{kind}"[-MAX_CLIENT_ID:]


def _untradable(entry: dict[str, Any] | None) -> str | None:
    if entry is None:
        return "the exchange does not list it"
    if entry.get("status") != "TRADING":
        return f"its status is {entry.get('status')}"
    if entry.get("contractType") != "PERPETUAL":
        return f"it is not a perpetual contract ({entry.get('contractType')})"
    if entry.get("quoteAsset") != QUOTE_ASSET or entry.get("marginAsset") != QUOTE_ASSET:
        return f"it is not quoted and margined in {QUOTE_ASSET}"
    return None


def _parse_rules(entry: dict[str, Any]) -> MarketRules:
    filters = {f["filterType"]: f for f in entry["filters"]}
    lot, price = filters["LOT_SIZE"], filters["PRICE_FILTER"]
    # Orders go at market: if their own filter is stricter, it rules.
    market = filters.get("MARKET_LOT_SIZE", lot)
    notional = filters.get("MIN_NOTIONAL", {})
    # No padding zeros ("0.10" -> 0.1): that way the rounded amounts have none either.
    step = max(Decimal(lot["stepSize"]), Decimal(market["stepSize"])).normalize()
    tick = Decimal(price["tickSize"]).normalize()
    if step <= 0 or tick <= 0:
        raise ValueError(f"quantity step {step} or price step {tick} not positive")
    maxima = [Decimal(f["maxQty"]) for f in (lot, market) if Decimal(f.get("maxQty", "0")) > 0]
    return MarketRules(
        symbol=entry["symbol"], step_size=step,
        min_qty=max(Decimal(lot["minQty"]), Decimal(market["minQty"])),
        max_qty=min(maxima) if maxima else None, tick_size=tick,
        min_notional=Decimal(str(notional.get("notional", notional.get("minNotional", "0")))),
    )
