"""The quant floor: the panel's figures regrouped as five desks, each with a status light.

Nothing here is new data. It reads the last report and what the panel already
computes live (positions at the current price, the radar) and arranges them by
job: data intake, the breakout signal, risk, execution and stops, and the
nightly audit. Each desk gets a light:

* ``ok`` (green): its part of the cycle is fine.
* ``watch`` (amber): something deserves a look.
* ``alert`` (red): a position without its stop, or a cycle that failed.
* ``idle`` (blue): nothing to do, or nothing yet to show.
"""

from __future__ import annotations

from typing import Any

from src.engine.daily_reporter import stop_check_text

OK, WATCH, ALERT, IDLE = "ok", "watch", "alert", "idle"
TOP_BREAKOUTS = 5
# Share of the open-risk limit from which the risk desk asks for a look.
HEAT_WATCH_SHARE = 0.75
# A tail correlation this high means the basket falls as one on bad days.
TAIL_CORRELATION_WATCH = 0.7
# A position closer than this (in %) to the level that would sell it.
EXIT_WATCH_PCT = 3.0
SETTLEMENT_SHOWN = 6


def quant_floor(
    report: dict[str, Any] | None, positions: list[dict[str, Any]], radar: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """The five desks, ready for JSON. Without a report every desk is idle."""
    if not report:
        return {"cycle_date": None, "desks": {name: {"light": IDLE} for name in
                                              ("data", "signal", "risk", "execution", "audit")}}
    rows = radar.get("rows") or {}
    return {
        "cycle_date": report["cycle_date"],
        "mode": report["mode"],
        "desks": {
            "data": _data_desk(report, history),
            "signal": _signal_desk(report, rows),
            "risk": _risk_desk(report),
            "execution": _execution_desk(report, positions, rows),
            "audit": _audit_desk(report),
        },
    }


def _data_desk(report: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    cycle = next((item for item in history if item.get("cycle_date") == report["cycle_date"]), {})
    measured = [row for row in report["basket"] if "close" in row]
    anomalies = sorted((item for item in report["observations"]["volatility"] if item["anomaly"]),
                       key=lambda item: -item["range_in_atr"])
    status = report["status"]
    light = (ALERT if status not in ("COMPLETED", "COMPLETED_WITH_ERRORS") else
             WATCH if status != "COMPLETED" or len(measured) < len(report["basket"]) or anomalies else OK)
    return {
        "light": light, "status": status, "completed_at": cycle.get("completed_at"),
        "late_minutes": report["late_minutes"], "symbols_with_data": len(measured),
        "symbols": len(report["basket"]),
        "missing": [row["symbol"] for row in report["basket"] if "close" not in row],
        "anomalies": [{key: item[key] for key in ("symbol", "range_in_atr", "return_pct")} for item in anomalies],
    }


def _signal_desk(report: dict[str, Any], rows: dict[str, Any]) -> dict[str, Any]:
    regime = report["observations"].get("regime") or {}
    candidates = sorted(
        (row for row in report["basket"] if row.get("breakout_strength") is not None and row["state"] != "IN_POSITION"),
        key=lambda row: -row["breakout_strength"],
    )[:TOP_BREAKOUTS]
    top = []
    for row in candidates:
        live = rows.get(row["symbol"]) or {}
        price = live.get("price")
        target = live.get("entry_channel") or row["high_entry"]
        top.append({
            "symbol": row["symbol"], "strength": row["breakout_strength"], "close": row["close"],
            "high_entry": row["high_entry"], "progress_pct": row["close"] / row["high_entry"] * 100.0,
            "price": price, "live_progress_pct": price / target * 100.0 if price and target else None,
        })
    light = OK if regime.get("breaking_out") else WATCH if regime.get("pre_alert") else IDLE
    return {"light": light, "regime": regime, "benchmarks": report["observations"].get("benchmarks"), "top": top,
            "turbulence": report["observations"].get("turbulence"), "edge_ratio": report["observations"].get("edge_ratio")}


def _risk_desk(report: dict[str, Any]) -> dict[str, Any]:
    account, correlation = report["account"], report["observations"]["correlation"]
    check = report.get("stop_check")
    unprotected = bool(check and (check.get("unprotected") or check.get("uncovered") or check.get("error")))
    heat_share = account["heat_pct"] / account["heat_limit_pct"] if account["heat_limit_pct"] else 0.0
    tail = correlation.get("tail_mean")
    light = (ALERT if unprotected else
             WATCH if heat_share >= HEAT_WATCH_SHARE or (tail is not None and tail >= TAIL_CORRELATION_WATCH) else OK)
    return {
        "light": light,
        "heat_pct": account["heat_pct"], "heat_limit_pct": account["heat_limit_pct"],
        "leverage": account["leverage"], "leverage_limit": account["leverage_limit"],
        "available_balance": account["available_balance"], "wallet_balance": account["wallet_balance"],
        "equity": account["equity"],
        "correlation_mean": correlation.get("basket_mean"), "correlation_days": correlation.get("days"),
        "tail_mean": tail, "tail_days": correlation.get("tail_days"),
        "tail_down_share_pct": correlation.get("tail_down_share_pct"),
        "stops": stop_check_text(check), "stops_ok": check is not None and not unprotected,
        "stops_checked": check is not None,
    }


def _execution_desk(report: dict[str, Any], positions: list[dict[str, Any]], rows: dict[str, Any]) -> dict[str, Any]:
    reported = {position["symbol"]: position for position in report["positions"]}
    basket = {row["symbol"]: row for row in report["basket"]}
    desk_positions = []
    for live in positions:
        symbol, before = live["symbol"], reported.get(live["symbol"], {})
        stop, channel, price = live.get("stop_price"), live.get("exit_channel"), live.get("price")
        levels = [level for level in (stop, channel) if level is not None]
        nearest = max(levels) if levels else None
        risk = before.get("risk_usd")
        desk_positions.append({
            "symbol": symbol, "quantity": live["quantity"], "entry_price": live["entry_price"], "price": price,
            "stop_price": stop, "exit_channel": channel,
            "high_entry": (rows.get(symbol) or {}).get("entry_channel") or (basket.get(symbol) or {}).get("high_entry"),
            "pnl_usd": live.get("pnl_usd"), "r_multiple": live["pnl_usd"] / risk if risk and live.get("pnl_usd") is not None else None,
            "nearest_exit": nearest, "nearest_exit_is_stop": nearest is not None and nearest == stop,
            "exit_distance_pct": (nearest / price - 1.0) * 100.0 if nearest and price else None,
            "status": live.get("status"),
        })
    settled = report.get("closed_trades") or []
    protected = [trade["protected_usd"] for trade in settled if trade.get("protected_usd") is not None]
    close_call = any(position["status"] != "HOLD" or (position["exit_distance_pct"] is not None
                                                      and position["exit_distance_pct"] > -EXIT_WATCH_PCT)
                     for position in desk_positions)
    light = IDLE if not desk_positions else WATCH if close_call else OK
    return {"light": light, "positions": desk_positions, "settlement": settled[:SETTLEMENT_SHOWN],
            "protected_total_usd": sum(protected) if protected else None}


def _audit_desk(report: dict[str, Any]) -> dict[str, Any]:
    audit = report.get("llm_audit")
    light = WATCH if report["status"] != "COMPLETED" else OK if audit else IDLE
    return {
        "light": light, "cycle_date": report["cycle_date"],
        "prompts": [{"title": prompt["title"], "prompt": prompt["prompt"]} for prompt in report["llm_prompts"]],
        "audit": audit,
    }
