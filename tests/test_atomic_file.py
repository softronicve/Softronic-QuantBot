"""Tests of the atomic, durable file write (``src/atomic_file.py``)."""

from __future__ import annotations

from pathlib import Path

from src.atomic_file import write_atomically


def test_the_file_is_replaced_whole_and_no_temporary_is_left(tmp_path: Path) -> None:
    path = tmp_path / "state" / "runtime_state.json"
    write_atomically(path, "first")
    write_atomically(path, "second")
    assert path.read_text(encoding="utf-8") == "second"
    assert sorted(item.name for item in path.parent.iterdir()) == ["runtime_state.json"]


def test_the_temporary_name_can_be_chosen(tmp_path: Path) -> None:
    report = tmp_path / "daily_audit_2026-10-15.md"
    sibling = tmp_path / "daily_audit_2026-10-15.json"
    sibling.write_text("{}", encoding="utf-8")
    write_atomically(report, "# report", report.with_name(report.name + ".tmp"))
    assert report.read_text(encoding="utf-8") == "# report" and sibling.read_text(encoding="utf-8") == "{}"
