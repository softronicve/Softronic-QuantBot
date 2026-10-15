"""Console setup shared by the programs of the project."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Leave a single destination for the log: the console or, if asked, a file.

    The file rotates at 5 MB and the five previous ones are kept, so that a
    bot running for weeks with nobody watching does not fill the disk.
    """
    use_utf8_output()
    logger.remove()
    if settings.log_file is not None:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            settings.log_file, level=settings.log_level, serialize=settings.log_json,
            rotation="5 MB", retention=5, encoding="utf-8",
        )
    else:
        logger.add(sys.stderr, level=settings.log_level, serialize=settings.log_json)


def use_utf8_output() -> None:
    """Make the output accept accents and symbols on any console.

    On Windows, when the output is redirected to a file, Python uses the system
    code page and a character outside it would stop the program.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
