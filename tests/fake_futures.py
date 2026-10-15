"""Double of ``binance.client.Client`` limited to the USDT-M futures endpoints the project uses.

It keeps a fake account (balance, positions, orders and stops) and records
every call. It defines no other method: if the code tried to use an endpoint
outside the list, the test would fail with ``AttributeError``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from binance.exceptions import BinanceAPIException
from requests.exceptions import ConnectionError as RequestsConnectionError

DAY_MS = 86_400_000
# 2026-10-09 00:01:30 UTC: the instant at which that day's cycle would be due.
NOW_MS = 1_791_504_000_000 + 90_000


def api_error(code: int, message: str = "error", status: int = 400) -> BinanceAPIException:
    return BinanceAPIException(None, status, json.dumps({"code": code, "msg": message}))


def network_error() -> RequestsConnectionError:
    return RequestsConnectionError("connection dropped")


def symbol_info(
    symbol: str, *, step: str = "0.001", min_qty: str = "0.001", tick: str = "0.10", min_notional: str = "5",
    market_max: str = "120", status: str = "TRADING", contract: str = "PERPETUAL", quote: str = "USDT",
) -> dict[str, Any]:
    return {
        "symbol": symbol, "status": status, "contractType": contract, "quoteAsset": quote, "marginAsset": quote,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": tick, "minPrice": "0.01", "maxPrice": "1000000"},
            {"filterType": "LOT_SIZE", "stepSize": step, "minQty": min_qty, "maxQty": "10000000"},
            {"filterType": "MARKET_LOT_SIZE", "stepSize": step, "minQty": min_qty, "maxQty": market_max},
            {"filterType": "MIN_NOTIONAL", "notional": min_notional},
        ],
    }


def daily_klines(last: float, n: int = 150, now_ms: int = NOW_MS, forming: bool = True) -> list[list[Any]]:
    """Daily candles as Binance gives them: a range between 98 and 102 and a last close at ``last``.

    The last closed candle is the one of the day before ``now_ms``. With ``forming``
    the current day's candle is appended after it, which the loader must drop.
    """
    close = np.array([*(100.0 + 2.0 * np.sin(np.arange(n - 1) * 0.7)), last])
    open_ = np.concatenate([[close[0]], close[:-1]])
    first = (now_ms // DAY_MS - n) * DAY_MS
    rows = [
        [first + i * DAY_MS, f"{open_[i]:.4f}", f"{max(open_[i], close[i]) + 1.0:.4f}",
         f"{min(open_[i], close[i]) - 1.0:.4f}", f"{close[i]:.4f}", "1000.0", first + (i + 1) * DAY_MS - 1,
         "0", 0, "0", "0", "0"]
        for i in range(n)
    ]
    if forming:
        today = first + n * DAY_MS
        rows.append([today, f"{last:.4f}", f"{last + 0.5:.4f}", f"{last - 0.5:.4f}", f"{last:.4f}", "10.0",
                     today + DAY_MS - 1, "0", 0, "0", "0", "0"])
    return rows


class FakeFuturesClient:
    """Fake futures account, with programmable failures per endpoint."""

    def __init__(self, symbols: dict[str, dict[str, Any]] | None = None, wallet: float = 10_000.0) -> None:
        self.now_ms = NOW_MS
        self.symbols = symbols if symbols is not None else {"BTCUSDT": symbol_info("BTCUSDT")}
        self.marks: dict[str, float] = {symbol: 100.0 for symbol in self.symbols}
        self.klines: dict[str, list[list[Any]]] = {}
        self.funding: dict[str, list[dict[str, Any]]] = {}
        self.wallet = wallet
        self.dual_side = False
        self.leverage: dict[str, int] = {}
        # symbol -> [quantity, entry price]
        self.positions: dict[str, list[float]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.algo_orders: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # Pending failures per endpoint, in order. ``fail`` fails without executing;
        # ``fail_after`` executes and then fails (the answer is lost).
        self.fail: dict[str, list[Exception]] = {}
        self.fail_after: dict[str, list[Exception]] = {}
        self._ids = iter(range(1, 1_000_000))

    # --- Test helpers ------------------------------------------------------

    def sent(self, endpoint: str) -> list[dict[str, Any]]:
        return [parameters for name, parameters in self.calls if name == endpoint]

    def _enter(self, endpoint: str, parameters: dict[str, Any]) -> None:
        self.calls.append((endpoint, dict(parameters)))
        if self.fail.get(endpoint):
            raise self.fail[endpoint].pop(0)

    def _leave(self, endpoint: str) -> None:
        if self.fail_after.get(endpoint):
            raise self.fail_after[endpoint].pop(0)

    # --- Market ------------------------------------------------------------

    def futures_time(self) -> dict[str, Any]:
        self._enter("futures_time", {})
        return {"serverTime": self.now_ms}

    def futures_exchange_info(self) -> dict[str, Any]:
        self._enter("futures_exchange_info", {})
        return {"symbols": list(self.symbols.values())}

    def futures_klines(self, **parameters: Any) -> list[list[Any]]:
        self._enter("futures_klines", parameters)
        return self.klines[parameters["symbol"]][-parameters["limit"] :]

    def futures_mark_price(self, **parameters: Any) -> dict[str, Any]:
        self._enter("futures_mark_price", parameters)
        return {"symbol": parameters["symbol"], "markPrice": f"{self.marks[parameters['symbol']]:.8f}"}

    def futures_funding_rate(self, **parameters: Any) -> list[dict[str, Any]]:
        self._enter("futures_funding_rate", parameters)
        return [
            payment for payment in self.funding.get(parameters["symbol"], [])
            if parameters["startTime"] <= payment["fundingTime"] <= parameters["endTime"]
        ]

    # --- Account -----------------------------------------------------------

    def futures_account(self, **parameters: Any) -> dict[str, Any]:
        self._enter("futures_account", parameters)
        unrealized = sum(q * (self.marks[s] - entry) for s, (q, entry) in self.positions.items())
        margin = sum(abs(q) * self.marks[s] for s, (q, _) in self.positions.items())
        return {"assets": [
            {"asset": "BNB", "walletBalance": "0", "availableBalance": "0", "unrealizedProfit": "0"},
            {"asset": "USDT", "walletBalance": f"{self.wallet:.8f}", "unrealizedProfit": f"{unrealized:.8f}",
             "availableBalance": f"{self.wallet + unrealized - margin:.8f}"},
        ]}

    def futures_position_information(self, **parameters: Any) -> list[dict[str, Any]]:
        self._enter("futures_position_information", parameters)
        return [
            {"symbol": symbol, "positionSide": "BOTH", "positionAmt": f"{quantity:.8f}",
             "entryPrice": f"{entry:.8f}", "markPrice": f"{self.marks[symbol]:.8f}"}
            for symbol, (quantity, entry) in self.positions.items()
            if parameters.get("symbol") in (None, symbol)
        ]

    def futures_get_position_mode(self, **parameters: Any) -> dict[str, Any]:
        self._enter("futures_get_position_mode", parameters)
        return {"dualSidePosition": self.dual_side}

    def futures_change_leverage(self, **parameters: Any) -> dict[str, Any]:
        self._enter("futures_change_leverage", parameters)
        self.leverage[parameters["symbol"]] = parameters["leverage"]
        return {"symbol": parameters["symbol"], "leverage": parameters["leverage"]}

    # --- Orders ------------------------------------------------------------

    def futures_create_order(self, **parameters: Any) -> dict[str, Any]:
        self._enter("futures_create_order", parameters)
        symbol, quantity, price = parameters["symbol"], float(parameters["quantity"]), self.marks[parameters["symbol"]]
        held, entry = self.positions.get(symbol, [0.0, 0.0])
        if parameters["side"] == "BUY":
            self.positions[symbol] = [held + quantity, price]
        else:
            if parameters.get("reduceOnly") != "true":
                raise AssertionError("a sell without reduceOnly could open a short")
            quantity = min(quantity, max(held, 0.0))
            if quantity <= 0.0:
                raise api_error(-2022, "ReduceOnly Order is rejected.")
            self.wallet += quantity * (price - entry)
            if held - quantity > 1e-12:
                self.positions[symbol] = [held - quantity, entry]
            else:
                del self.positions[symbol]
        order = {
            "orderId": next(self._ids), "clientOrderId": parameters["newClientOrderId"], "symbol": symbol,
            "status": "FILLED", "executedQty": f"{quantity:.8f}", "avgPrice": f"{price:.8f}",
            "side": parameters["side"], "type": "MARKET",
        }
        self.orders[parameters["newClientOrderId"]] = order
        self._leave("futures_create_order")
        return dict(order)

    def futures_get_order(self, **parameters: Any) -> dict[str, Any]:
        self._enter("futures_get_order", parameters)
        order = self.orders.get(parameters["origClientOrderId"])
        if order is None:
            raise api_error(-2013, "Order does not exist.")
        return dict(order)

    def futures_create_algo_order(self, **parameters: Any) -> dict[str, Any]:
        self._enter("futures_create_algo_order", parameters)
        order = {
            "algoId": next(self._ids), "clientAlgoId": parameters["clientAlgoId"], "algoType": "CONDITIONAL",
            "orderType": parameters["type"], "symbol": parameters["symbol"], "side": parameters["side"],
            "quantity": parameters["quantity"], "triggerPrice": parameters["triggerPrice"], "algoStatus": "NEW",
            "reduceOnly": parameters.get("reduceOnly") == "true", "workingType": parameters.get("workingType"),
        }
        self.algo_orders.append(order)
        self._leave("futures_create_algo_order")
        return dict(order)

    def futures_get_open_algo_orders(self, **parameters: Any) -> list[dict[str, Any]]:
        self._enter("futures_get_open_algo_orders", parameters)
        return [dict(order) for order in self.algo_orders if parameters.get("symbol") in (None, order["symbol"])]

    def futures_cancel_algo_order(self, **parameters: Any) -> dict[str, Any]:
        self._enter("futures_cancel_algo_order", parameters)
        self.algo_orders = [order for order in self.algo_orders if order["algoId"] != parameters["algoId"]]
        return {"algoId": parameters["algoId"], "code": "200", "msg": "success"}
