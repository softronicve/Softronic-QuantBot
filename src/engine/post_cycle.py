"""Optional work after a daily cycle: a healthcheck ping and an LLM review of the report.

Both are off unless configured, run after the cycle is recorded, its report
written and its alert sent, and never raise: a monitoring service or a model
being down must not touch the trading cycle.

* Healthcheck ping (``BOT_HEALTHCHECK_PING_URL``): a GET when a cycle finishes
  as ``COMPLETED``. Pointed at a dead-man's-switch service (healthchecks.io,
  Uptime Kuma, ...), a missing ping is what raises the alarm.
* LLM auditor (``BOT_AUDIT_LLM__ENABLED=true``): answers the three questions of
  section 5 of the report and appends them as section 6. It sends the report's
  figures to Anthropic and costs tokens on every cycle.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from loguru import logger

from src.atomic_file import write_atomically

PING_TIMEOUT_SECONDS = 10.0
AUDIT_TIMEOUT_SECONDS = 300.0
# The default fallback mode re-runs a declined request on the model Anthropic recommends for that refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
AUDITOR_SYSTEM = (
    "You are the nightly quant auditor of a trading bot running on paper. Answer the three numbered questions "
    "of the block in plain, short prose (at most about 250 words per block), using the figures you are given. "
    "Keep apart what the numbers show and what you assume. Do not suggest changing the rule or its parameters "
    "from a single day: point out data, calculation or execution errors."
)


def ping_healthcheck(url: str | None) -> bool:
    """GET ``url``; True if it answered with a 2xx. Never raises and never logs the URL (it may hold a token)."""
    if not url:
        return False
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        logger.warning("Healthcheck: the configured address is not http(s); no ping sent")
        return False
    try:
        with urllib.request.urlopen(url, timeout=PING_TIMEOUT_SECONDS) as response:
            ok = 200 <= response.status < 300
    except (OSError, ValueError) as error:
        logger.warning("Healthcheck: {} did not answer ({})", parsed.hostname, type(error).__name__)
        return False
    if not ok:
        logger.warning("Healthcheck: {} answered {}", parsed.hostname, response.status)
    return ok


class LlmAuditor:
    """Answers the report's LLM questions with Claude and appends the answers to the report."""

    def __init__(self, *, api_key: str | None, model: str, effort: str, client: Any | None = None) -> None:
        """
        Args:
            api_key: Anthropic API key; ``None`` lets the SDK find one (``ANTHROPIC_API_KEY``, a CLI profile).
            model: the model that answers.
            effort: ``output_config.effort``; how hard it thinks, and so what it costs.
            client: an already built client (for tests).
        """
        if client is None:
            import anthropic  # only needed when the auditor is enabled

            client = anthropic.Anthropic(api_key=api_key, timeout=AUDIT_TIMEOUT_SECONDS)
        self._client, self._model, self._effort = client, model, effort

    def audit(self, json_path: Path, markdown_path: Path) -> bool:
        """Adds section 6 to the report of ``json_path``. True if it was written. Never raises."""
        try:
            report = json.loads(json_path.read_text(encoding="utf-8"))
            answers = [self._answer(prompt) for prompt in report.get("llm_prompts", [])]
        except Exception as error:  # noqa: BLE001 - the review is optional; the cycle is already done
            logger.warning("LLM auditor: not completed ({})", type(error).__name__)
            return False
        if not answers:
            return False
        from src.engine.daily_reporter import render_markdown  # the reporter imports nothing from here

        report["llm_audit"] = {"model": self._model, "answers": answers}
        write_atomically(json_path, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                         json_path.with_name(json_path.name + ".tmp"))
        write_atomically(markdown_path, render_markdown(report), markdown_path.with_name(markdown_path.name + ".tmp"))
        logger.info("LLM auditor: review added to the report ({} blocks)", len(answers))
        return True

    def _answer(self, prompt: dict[str, str]) -> dict[str, Any]:
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=16000,
            system=AUDITOR_SYSTEM,
            output_config={"effort": self._effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt["prompt"]}],
        )
        if response.stop_reason == "refusal":
            return {"title": prompt["title"], "answer": None, "refused": True, "served_by": response.model}
        text = "\n\n".join(block.text for block in response.content if block.type == "text").strip()
        return {"title": prompt["title"], "answer": text or None, "refused": False, "served_by": response.model,
                "truncated": response.stop_reason == "max_tokens"}
