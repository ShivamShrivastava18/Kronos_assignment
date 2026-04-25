"""Constraint parsing, validation, and row filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.data.loader import DatasetSchema


@dataclass
class NumericRange:
    gte: Optional[float] = None
    lte: Optional[float] = None

    def contains(self, arr: np.ndarray) -> np.ndarray:
        mask = np.ones_like(arr, dtype=bool)
        if self.gte is not None:
            mask &= arr >= self.gte
        if self.lte is not None:
            mask &= arr <= self.lte
        return mask

    def clip(self, arr: np.ndarray) -> np.ndarray:
        lo = self.gte if self.gte is not None else -np.inf
        hi = self.lte if self.lte is not None else np.inf
        return np.clip(arr, lo, hi)


@dataclass
class NormalizedConstraints:
    """Constraints normalized against the dataset schema.

    - numeric: column -> NumericRange (closed interval)
    - categorical: column -> set of allowed string values
    - rejected: column -> human-readable reason why it was dropped
    """

    numeric: Dict[str, NumericRange] = field(default_factory=dict)
    categorical: Dict[str, List[str]] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for col, rng in self.numeric.items():
            d: Dict[str, float] = {}
            if rng.gte is not None:
                d["gte"] = rng.gte
            if rng.lte is not None:
                d["lte"] = rng.lte
            out[col] = d
        for col, vals in self.categorical.items():
            out[col] = list(vals) if len(vals) > 1 else vals[0]
        return out

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        for col, rng in self.numeric.items():
            if col not in df.columns:
                continue
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy()
            mask &= rng.contains(arr)
        for col, vals in self.categorical.items():
            if col not in df.columns:
                continue
            mask &= df[col].isin(vals)
        return df.loc[mask].reset_index(drop=True)


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_numeric_constraint(
    col: str,
    raw: Any,
    schema: DatasetSchema,
) -> Tuple[Optional[NumericRange], Optional[str]]:
    rng = schema.value_ranges[col]
    lo_data, hi_data = rng["min"], rng["max"]

    # scalar -> equality
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        if not (lo_data <= v <= hi_data):
            return None, (
                f"value {v} outside observed range [{lo_data}, {hi_data}]"
            )
        return NumericRange(gte=v, lte=v), None

    if isinstance(raw, list):
        vals = [_to_float(x) for x in raw]
        if any(v is None for v in vals) or not vals:
            return None, "list contains non-numeric entries"
        lo = min(vals)
        hi = max(vals)
        return NumericRange(gte=lo, lte=hi), None

    # Pydantic models (e.g. NumericConstraint) get auto-parsed by FastAPI;
    # convert to a plain dict so the rest of the logic works uniformly.
    if isinstance(raw, BaseModel):
        raw = {k: v for k, v in raw.model_dump().items() if v is not None}

    if isinstance(raw, dict):
        gte = _to_float(raw.get("gte")) if raw.get("gte") is not None else None
        lte = _to_float(raw.get("lte")) if raw.get("lte") is not None else None
        gt = _to_float(raw.get("gt")) if raw.get("gt") is not None else None
        lt = _to_float(raw.get("lt")) if raw.get("lt") is not None else None
        eq = _to_float(raw.get("eq")) if raw.get("eq") is not None else None

        # Exclusive -> approximate as inclusive with a tiny nudge. For integer-like
        # columns we snap to the next integer.
        is_intish = col in {
            "age",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "fnlwgt",
        }
        if gt is not None:
            gte = (gt + 1) if is_intish else (gt + 1e-9 if gte is None else max(gte, gt + 1e-9))
        if lt is not None:
            lte = (lt - 1) if is_intish else (lt - 1e-9 if lte is None else min(lte, lt - 1e-9))

        if eq is not None:
            gte = eq
            lte = eq

        if gte is None and lte is None:
            return None, "no usable bound keys (expected gte/lte/gt/lt/eq)"

        if gte is not None and lte is not None and gte > lte:
            return None, f"empty interval: gte={gte} > lte={lte}"

        # Clamp to observed range so downstream sampling doesn't spin forever.
        if gte is not None:
            gte = max(gte, lo_data)
        if lte is not None:
            lte = min(lte, hi_data)
        if gte is not None and lte is not None and gte > lte:
            return None, "interval outside observed data range"
        return NumericRange(gte=gte, lte=lte), None

    if isinstance(raw, str):
        # Try to parse shorthand like ">40", ">=40", "40-60", "40".
        s = raw.strip()
        for op, key in (
            (">=", "gte"),
            ("<=", "lte"),
            (">", "gt"),
            ("<", "lt"),
            ("=", "eq"),
        ):
            if s.startswith(op):
                rest = s[len(op):].strip()
                v = _to_float(rest)
                if v is None:
                    return None, f"could not parse '{raw}'"
                return _normalize_numeric_constraint(col, {key: v}, schema)
        if "-" in s:
            parts = s.split("-", 1)
            a, b = _to_float(parts[0]), _to_float(parts[1])
            if a is None or b is None:
                return None, f"could not parse range '{raw}'"
            return _normalize_numeric_constraint(
                col, {"gte": min(a, b), "lte": max(a, b)}, schema
            )
        v = _to_float(s)
        if v is not None:
            return _normalize_numeric_constraint(col, v, schema)
        return None, f"could not parse '{raw}' as a numeric constraint"

    return None, f"unsupported constraint type {type(raw).__name__}"


def _normalize_categorical_constraint(
    col: str,
    raw: Any,
    schema: DatasetSchema,
) -> Tuple[Optional[List[str]], Optional[str]]:
    allowed = schema.categorical_values[col]
    allowed_set = {a.lower(): a for a in allowed}

    def _match(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return allowed_set.get(s.lower())

    if isinstance(raw, (str, int, float, bool)):
        canon = _match(raw)
        if canon is None:
            return None, (
                f"value '{raw}' not in schema (expected one of {allowed[:10]}"
                f"{'...' if len(allowed) > 10 else ''})"
            )
        return [canon], None

    if isinstance(raw, list):
        out: List[str] = []
        bad: List[Any] = []
        for v in raw:
            canon = _match(v)
            if canon is None:
                bad.append(v)
            else:
                out.append(canon)
        if not out:
            return None, f"no valid values in {raw}; allowed: {allowed[:10]}"
        out = sorted(set(out))
        if bad:
            # Partially valid: keep valid entries, flag the rest. Non-fatal.
            return out, f"ignored unknown values: {bad}"
        return out, None

    if isinstance(raw, dict):
        # Accept {"in": [...]} style if ever sent.
        if "in" in raw and isinstance(raw["in"], list):
            return _normalize_categorical_constraint(col, raw["in"], schema)
        return None, "dict constraints are not supported for categorical columns"

    return None, f"unsupported constraint type {type(raw).__name__}"


def normalize_constraints(
    raw_constraints: Dict[str, Any],
    schema: DatasetSchema,
) -> NormalizedConstraints:
    """Normalize a user constraints dict against the schema. Never raises;
    invalid entries are collected in `.rejected` with a readable reason."""
    out = NormalizedConstraints()
    for col, raw in (raw_constraints or {}).items():
        if col not in schema.columns:
            out.rejected[col] = (
                f"unknown column. Valid columns: {schema.columns}"
            )
            continue

        if col in schema.numeric_columns:
            rng, err = _normalize_numeric_constraint(col, raw, schema)
            if rng is None:
                out.rejected[col] = err or "invalid numeric constraint"
            else:
                out.numeric[col] = rng
        else:
            vals, err = _normalize_categorical_constraint(col, raw, schema)
            if vals is None:
                out.rejected[col] = err or "invalid categorical constraint"
            else:
                out.categorical[col] = vals
                if err:  # partial warning
                    out.rejected[col] = err
    return out
