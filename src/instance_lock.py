"""Instance lock: keeps two processes from using the same state at once.

Two bots writing the same position and the same ledger would overwrite each
other without raising any error. Each bot holds a lock on a file in its state
folder while it lives; the operating system releases it by itself if the process
dies, so there is never an "orphan" lock left to delete by hand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO


class StateInUseError(RuntimeError):
    """Another process is already working with that same state."""


def _try_lock(handle: IO[str]) -> bool:
    try:
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415 (only exists on Windows)

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl  # noqa: PLC0415 (does not exist on Windows)

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


class InstanceLock:
    """Exclusive, non-blocking lock on a file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: IO[str] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> bool:
        """Take the lock; returns False if another process holds it."""
        if self._handle is not None:
            return True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+", encoding="utf-8")
        if not _try_lock(handle):
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is not None:
            self._handle.close()  # closing the file releases the lock
            self._handle = None

    @staticmethod
    def is_held(path: Path) -> bool:
        """True if another process holds the lock on ``path`` right now."""
        if not path.exists():
            return False
        probe = InstanceLock(path)
        if probe.acquire():
            probe.release()
            return False
        return True
