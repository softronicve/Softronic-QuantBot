"""Memory of the daily cycle: which days already ran and which orders went out on each.

It keeps a restart from repeating the buys of a day already traded. A cycle is
on record as started before its first order and each order is recorded as soon
as it is sent, so after a crash halfway through it is known how far it got.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.atomic_file import write_atomically

STARTED = "STARTED"
COMPLETED = "COMPLETED"


class StateError(RuntimeError):
    """The state file cannot be read or what is asked to be saved is not valid."""


class DailyStateStore:
    """State of the daily cycle in a JSON file, written atomically."""

    def __init__(self, path: Path, keep_cycles: int = 400) -> None:
        """
        Args:
            path: state file; it is created when the first cycle is recorded.
            keep_cycles: cycles that are kept; the oldest are dropped.

        Raises:
            StateError: if the file exists and cannot be read. It is not
                treated as empty: that would allow repeating a day already traded.
        """
        self._path = path
        self._keep = max(1, keep_cycles)
        self._cycles: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                cycles = json.loads(path.read_text(encoding="utf-8"))["cycles"]
                if not isinstance(cycles, dict):
                    raise TypeError("'cycles' is not a dictionary")
            except (ValueError, KeyError, TypeError) as error:
                raise StateError(f"the daily cycle state {path} cannot be read: {error}") from error
            self._cycles = cycles

    def is_cycle_already_executed(self, cycle_date: str) -> bool:
        """``True`` if the cycle of that date was completed."""
        cycle = self._cycles.get(_valid(cycle_date))
        return cycle is not None and cycle.get("status") == COMPLETED

    def get_cycle(self, cycle_date: str) -> dict[str, Any] | None:
        cycle = self._cycles.get(_valid(cycle_date))
        return json.loads(json.dumps(cycle)) if cycle is not None else None

    def recent_cycles(self, count: int = 30) -> list[dict[str, Any]]:
        """The last cycles recorded, from newest to oldest, each with its ``cycle_date``."""
        dates = sorted(self._cycles, reverse=True)[: max(0, count)]
        return [{"cycle_date": day, **json.loads(json.dumps(self._cycles[day]))} for day in dates]

    def mark_cycle_started(self, cycle_date: str) -> None:
        """Record that the cycle has started. If it was already on record, nothing changes."""
        if _valid(cycle_date) not in self._cycles:
            self._cycles[cycle_date] = {"status": STARTED, "started_at": _now(), "dispatched": {}}
            self._save()

    def record_dispatch(self, cycle_date: str, symbol: str, action: str, result: dict[str, Any]) -> None:
        """Record the result of an order of the cycle, right after sending it."""
        self.mark_cycle_started(cycle_date)
        self._cycles[cycle_date]["dispatched"][f"{symbol}:{action}"] = _plain(result)
        self._save()

    def dispatched(self, cycle_date: str) -> dict[str, dict[str, Any]]:
        """Orders already recorded in the cycle, by ``"SYMBOL:ACTION"``."""
        cycle = self.get_cycle(cycle_date)
        return cycle["dispatched"] if cycle else {}

    def mark_cycle_incomplete(self, cycle_date: str, entry_block: str | None) -> None:
        """Notes that the cycle ran but left something pending, and how its entries were settled.

        ``entry_block`` is why that attempt took no entries (e.g. a failed
        close), or ``None`` if it decided them. Either way a later attempt of the
        same day takes no new ones: entries are decided once a day.
        """
        self.mark_cycle_started(cycle_date)
        self._cycles[cycle_date]["incomplete"] = {"entry_block": entry_block}
        self._save()

    def mark_cycle_completed(self, cycle_date: str, summary: dict[str, Any]) -> None:
        """Mark the cycle as executed: from here on it is not repeated."""
        self.mark_cycle_started(cycle_date)
        self._cycles[cycle_date].update(status=COMPLETED, completed_at=_now(), summary=_plain(summary))
        self._save()

    def _save(self) -> None:
        for old in sorted(self._cycles)[: -self._keep]:
            del self._cycles[old]
        write_atomically(self._path, json.dumps({"version": 1, "cycles": self._cycles}, indent=2))


def _valid(cycle_date: str) -> str:
    try:
        if not isinstance(cycle_date, str) or date.fromisoformat(cycle_date).isoformat() != cycle_date:
            raise ValueError
    except ValueError:
        raise StateError(f"the cycle date must have the form YYYY-MM-DD, got {cycle_date!r}") from None
    return cycle_date


def _plain(value: dict[str, Any]) -> dict[str, Any]:
    """A copy that can be saved as JSON; whatever cannot is saved as text."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError) as error:
        raise StateError(f"the result cannot be saved: {error}") from error


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
