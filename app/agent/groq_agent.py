"""Groq-backed conversational agent using the Llama-3.3-70B tool-calling API."""

from __future__ import annotations

import json
import logging
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.data.loader import get_schema
from app.agent.tools import TOOL_SPECS, ToolError, run_tool
from app.storage import store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent trace logger — emits structured, human-readable CLI output
# ---------------------------------------------------------------------------

_DIVIDER      = "─" * 72
_THICK        = "═" * 72
_TOOL_CALL    = "⚙  TOOL CALL"
_TOOL_RESULT  = "↩  TOOL RESULT"
_LLM_THINK    = "🤔 LLM REASONING"
_LLM_FINAL    = "✅ FINAL REPLY"
_LLM_LOOP     = "🔄 LLM TURN"
_USER_IN      = "💬 USER"
_AGENT_START  = "🚀 AGENT SESSION"
_MAX_RESULT_CHARS = 1_200   # truncate large tool results in the log


def _pjson(obj: Any, indent: int = 2, max_chars: int = _MAX_RESULT_CHARS) -> str:
    """Pretty-print JSON, trimming large blobs."""
    try:
        text = json.dumps(obj, indent=indent, default=str)
    except Exception:
        text = str(obj)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n  … (truncated, {len(text)} chars total)"
    return text


def _log_divider(title: str = "") -> None:
    if title:
        pad = max(0, 72 - len(title) - 4)
        logger.info("── %s %s", title, "─" * pad)
    else:
        logger.info(_DIVIDER)


def _log_thick(title: str = "") -> None:
    if title:
        pad = max(0, 72 - len(title) - 4)
        logger.info("══ %s %s", title, "═" * pad)
    else:
        logger.info(_THICK)

try:
    from groq import Groq  # type: ignore
    _HAS_GROQ = True
except Exception:  # pragma: no cover
    _HAS_GROQ = False


SYSTEM_PROMPT = """\
You are a synthetic-data assistant. You help the user generate and validate
conditional synthetic tabular data from the UCI Adult Income dataset.

You have four tools:
  * get_schema          – returns the real dataset's columns, allowed values,
                          and numeric ranges.
  * sample_real         – returns a few real rows, useful as examples.
  * generate            – generates N synthetic rows subject to constraints
                          and returns a generation_id.
  * validate            – validates a generation_id; returns TSTR accuracy,
                          baseline accuracy, KS/JS metrics, fidelity score,
                          and a PASS/FAIL verdict.

Rules you MUST follow:
  1. Before calling `generate` with an unfamiliar constraint, call
     `get_schema` to confirm the column name and allowed values. NEVER
     invent column names or categorical values that are not in the schema.
  2. If the user's request is ambiguous (e.g. "make it realistic", "older
     people"), ask a targeted clarifying question before generating.
  3. After `generate`, call `validate` unless the user explicitly opted out.
  4. Interpret validation output for the user in plain language: mention
     the TSTR accuracy vs baseline, whether the verdict is PASS or FAIL,
     and, if FAIL, point to the worst-offending KS/JS columns and suggest
     constraint tweaks (e.g. widen a range, drop a rare-value filter, or
     try the histogram strategy).
  5. Keep replies concise. Do not repeat tool outputs verbatim; summarize.
  6. Constraint syntax: scalars for equality, lists for membership, and
     objects like {"gte": 40, "lte": 60} for numeric ranges.
"""


def _schema_snapshot_text() -> str:
    s = get_schema()
    # Trim very long categorical lists for the context window.
    snippet = {
        "columns": s.columns,
        "numeric_ranges": {
            c: {"min": r["min"], "max": r["max"]} for c, r in s.value_ranges.items()
        },
        "categorical_values": {
            c: (vals if len(vals) <= 12 else vals[:10] + ["..."])
            for c, vals in s.categorical_values.items()
        },
        "target": {
            "column": s.target_column,
            "class_distribution": s.class_distribution,
        },
    }
    return json.dumps(snippet)


@dataclass
class AgentReply:
    reply: str
    generation_id: Optional[str]
    validation_summary: Optional[Dict[str, Any]]
    tool_calls_made: List[str]


class GroqAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None

    def _ensure_client(self):
        if not _HAS_GROQ:
            raise RuntimeError(
                "`groq` package not available; install it to use the agent."
            )
        if not self.settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. Set it in your environment"
                " or .env file."
            )
        if self._client is None:
            self._client = Groq(api_key=self.settings.groq_api_key)
        return self._client

    # ------------------------------------------------------------------
    def _initial_messages(self) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"Dataset schema snapshot: {_schema_snapshot_text()}",
            },
        ]

    def chat(self, session_id: str, user_message: str) -> AgentReply:
        client = self._ensure_client()
        messages = store.get_session(session_id)
        is_new_session = not messages
        if is_new_session:
            messages = self._initial_messages()

        messages.append({"role": "user", "content": user_message})

        # ── session header ──────────────────────────────────────────────────
        _log_thick(_AGENT_START)
        logger.info("  session_id : %s", session_id)
        logger.info("  model      : %s", self.settings.groq_model)
        logger.info("  new_session: %s", is_new_session)
        _log_divider(_USER_IN)
        logger.info("  %s", user_message)
        _log_thick()

        tool_calls_made: List[str] = []
        last_generation_id: Optional[str] = None
        last_validation: Optional[Dict[str, Any]] = None
        session_start = time.perf_counter()

        # Tool-calling loop with a safety cap to prevent runaway agents.
        for turn in range(8):
            _log_divider(f"{_LLM_LOOP}  #{turn + 1}")
            turn_start = time.perf_counter()

            response = client.chat.completions.create(
                model=self.settings.groq_model,
                messages=messages,
                tools=TOOL_SPECS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1024,
            )
            turn_ms = (time.perf_counter() - turn_start) * 1000

            choice = response.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason
            usage = response.usage

            logger.info(
                "  finish_reason=%s  tokens(prompt=%s completion=%s)  latency=%.0fms",
                finish_reason,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                turn_ms,
            )

            # Log any intermediate text the LLM produced (reasoning/thinking).
            if msg.content and msg.content.strip():
                _log_divider(_LLM_THINK)
                for line in textwrap.wrap(msg.content.strip(), width=70):
                    logger.info("  %s", line)

            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if getattr(msg, "tool_calls", None):
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_entry)

            if not getattr(msg, "tool_calls", None):
                # ── final reply ─────────────────────────────────────────────
                total_ms = (time.perf_counter() - session_start) * 1000
                _log_divider(_LLM_FINAL)
                for line in textwrap.wrap((msg.content or "").strip(), width=70):
                    logger.info("  %s", line)
                _log_thick()
                logger.info(
                    "  tools used: %s  |  total turns: %d  |  wall time: %.0fms",
                    tool_calls_made or "none",
                    turn + 1,
                    total_ms,
                )
                _log_thick()

                store.set_session(session_id, messages)
                return AgentReply(
                    reply=msg.content or "",
                    generation_id=last_generation_id,
                    validation_summary=last_validation,
                    tool_calls_made=tool_calls_made,
                )

            # ── dispatch each tool call ─────────────────────────────────────
            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments or "{}"
                tool_calls_made.append(name)
                store.append_tool_call(
                    session_id, {"name": name, "arguments": args}
                )

                # log the call
                _log_divider(f"{_TOOL_CALL}  →  {name}")
                try:
                    parsed_args = json.loads(args)
                except Exception:
                    parsed_args = args
                logger.info("  id   : %s", tc.id)
                logger.info("  args :\n%s", textwrap.indent(_pjson(parsed_args), "    "))

                # execute
                tool_start = time.perf_counter()
                try:
                    result = run_tool(name, args)
                    tool_content = json.dumps(result, default=str)
                    tool_ok = True
                except ToolError as e:
                    result = {"error": str(e)}
                    tool_content = json.dumps(result)
                    tool_ok = False
                    logger.warning("  ToolError: %s", e)
                except Exception as e:  # noqa: BLE001
                    result = {"error": f"internal error: {e}"}
                    tool_content = json.dumps(result)
                    tool_ok = False
                    logger.exception("  Tool %s crashed", name)
                tool_ms = (time.perf_counter() - tool_start) * 1000

                # log the result
                _log_divider(f"{_TOOL_RESULT}  ←  {name}  ({'ok' if tool_ok else 'ERROR'})  {tool_ms:.0f}ms")
                # For generate: suppress the full rows array, show summary instead
                log_result = result
                if name == "generate" and isinstance(result, dict) and "preview_rows" in result:
                    log_result = {k: v for k, v in result.items() if k != "preview_rows"}
                    log_result["preview_rows"] = f"[{len(result.get('preview_rows', []))} rows shown, full set in store]"
                logger.info("%s", textwrap.indent(_pjson(log_result), "  "))

                # stash interesting bits
                if name == "generate":
                    last_generation_id = result.get("generation_id")
                elif name == "validate" and tool_ok:
                    last_validation = {
                        "verdict": result.get("verdict"),
                        "tstr_accuracy": result.get("tstr_accuracy"),
                        "baseline_accuracy": result.get("baseline_accuracy"),
                        "fidelity_score": result.get("fidelity_score"),
                        "tolerance": result.get("tolerance"),
                        "n_synthetic": result.get("n_synthetic"),
                        "n_real_test": result.get("n_real_test"),
                        "ks_tests": result.get("ks_tests", []),
                        "js_divergences": result.get("js_divergences", []),
                        "notes": result.get("notes", []),
                        "generation_id": result.get("generation_id"),
                    }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": tool_content,
                    }
                )

        # ── safety cap hit — force a final answer ───────────────────────────
        logger.warning("Agent hit 8-turn safety cap; forcing final answer.")
        messages.append(
            {
                "role": "system",
                "content": "Produce a final answer now without calling more tools.",
            }
        )
        response = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        final = response.choices[0].message.content or "(no reply)"
        messages.append({"role": "assistant", "content": final})

        total_ms = (time.perf_counter() - session_start) * 1000
        _log_divider(_LLM_FINAL + "  [forced]")
        for line in textwrap.wrap(final.strip(), width=70):
            logger.info("  %s", line)
        _log_thick()
        logger.info(
            "  tools used: %s  |  total turns: 8 (cap hit)  |  wall time: %.0fms",
            tool_calls_made or "none",
            total_ms,
        )
        _log_thick()

        store.set_session(session_id, messages)
        return AgentReply(
            reply=final,
            generation_id=last_generation_id,
            validation_summary=last_validation,
            tool_calls_made=tool_calls_made,
        )


_agent_singleton: Optional[GroqAgent] = None


def get_agent() -> GroqAgent:
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = GroqAgent()
    return _agent_singleton
