"""Tool definitions exposed to the LLM agent."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import pandas as pd

from app.data.loader import get_schema, sample_rows
from app.services import (
    _validation_to_dict,
    generate,
    validate_rows,
)
from app.storage import store

logger = logging.getLogger(__name__)


TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Return the dataset schema: column names, numeric vs categorical"
                " dtypes, observed numeric ranges, allowed categorical values,"
                " and the target class distribution. ALWAYS call this before"
                " generating or validating if you are unsure about column"
                " names or allowed values."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate",
            "description": (
                "Generate synthetic rows satisfying the given constraints."
                " Constraints is a dict keyed by column name. Values can be a"
                " scalar for equality, a list for membership, or a dict with"
                " 'gte'/'lte'/'gt'/'lt'/'eq' for numeric ranges. Returns a"
                " generation_id you can pass to `validate`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "constraints": {
                        "type": "object",
                        "description": "Column -> constraint map.",
                        "additionalProperties": True,
                    },
                    "n_rows": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100000,
                        "default": 500,
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["copula", "histogram"],
                        "default": "copula",
                    },
                    "random_seed": {"type": "integer", "minimum": 0},
                },
                "required": ["constraints", "n_rows"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate",
            "description": (
                "Run the validation suite on a previously-generated dataset."
                " Returns TSTR accuracy, real-data baseline accuracy, KS/JS"
                " per column, an overall fidelity score, and a PASS/FAIL"
                " verdict."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "generation_id": {
                        "type": "string",
                        "description": "The generation_id returned by `generate`.",
                    }
                },
                "required": ["generation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_real",
            "description": (
                "Return a small sample of real rows from the seed dataset."
                " Useful when the user asks what a row looks like, or when"
                " you need concrete examples of allowed values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "minimum": 1, "maximum": 25, "default": 5}
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


class ToolError(Exception):
    ...


def _schema_payload() -> Dict[str, Any]:
    s = get_schema()
    return {
        "columns": s.columns,
        "numeric_columns": s.numeric_columns,
        "categorical_columns": s.categorical_columns,
        "value_ranges": s.value_ranges,
        "categorical_values": s.categorical_values,
        "class_distribution": s.class_distribution,
        "target_column": s.target_column,
        "n_rows": s.n_rows,
    }


def run_tool(name: str, arguments_json: str) -> Dict[str, Any]:
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        raise ToolError(f"invalid JSON arguments: {e}") from e

    if name == "get_schema":
        return _schema_payload()

    if name == "sample_real":
        n = int(args.get("n", 5))
        return {"rows": sample_rows(n)}

    if name == "generate":
        constraints = args.get("constraints") or {}
        if not isinstance(constraints, dict):
            raise ToolError("`constraints` must be an object")
        n_rows = int(args.get("n_rows", 500))
        strategy = args.get("strategy", "copula")
        random_seed = args.get("random_seed")
        if random_seed is not None:
            random_seed = int(random_seed)
        out = generate(
            constraints=constraints,
            n_rows=n_rows,
            strategy=strategy,
            random_seed=random_seed,
        )
        return {
            "generation_id": out.generation_id,
            "strategy": out.strategy,
            "n_rows_requested": out.n_rows_requested,
            "n_rows_returned": len(out.rows),
            "constraints_applied": out.constraints_applied,
            "constraints_rejected": out.constraints_rejected,
            "random_seed": out.random_seed,
            "preview_rows": out.rows[:3],
        }

    if name == "validate":
        gid = args.get("generation_id")
        if not gid:
            raise ToolError("`generation_id` is required")
        record = store.get_generation(gid)
        if record is None:
            raise ToolError(f"unknown generation_id '{gid}'")
        result, _ = validate_rows(
            record.rows_df,
            generation_id=gid,
            constraints=record.constraints_applied,
        )
        payload = _validation_to_dict(result, gid)
        # Trim KS/JS lists for the model (keep only top-5 worst per metric).
        ks_sorted = sorted(
            payload["ks_tests"], key=lambda m: m["statistic"], reverse=True
        )[:5]
        js_sorted = sorted(
            payload["js_divergences"],
            key=lambda m: m["statistic"],
            reverse=True,
        )[:5]
        payload["ks_tests"] = ks_sorted
        payload["js_divergences"] = js_sorted
        return payload

    raise ToolError(f"unknown tool '{name}'")
