"""Atomic, durable writes for the files the bot cannot afford to lose half of."""

from __future__ import annotations

import os
from pathlib import Path


def write_atomically(path: Path, text: str, temporary: Path | None = None) -> None:
    """Replaces ``path`` with ``text``: a reader sees the old file or the new one, never half of it.

    The new content reaches the disk before the rename (``fsync`` of the file,
    then of its folder), so a host crash or a power cut cannot leave an empty or
    truncated state file behind.

    Args:
        temporary: where the content is written first; ``path`` with a ``.tmp``
            suffix by default. It must be on the same file system as ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary if temporary is not None else path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_folder(path.parent)


def _sync_folder(folder: Path) -> None:
    """Makes the rename itself durable. Windows cannot open a folder: there the rename is all there is."""
    try:
        descriptor = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
