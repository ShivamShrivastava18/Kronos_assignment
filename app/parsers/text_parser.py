"""Deterministic rule-based parser for natural-language constraint strings.

This is intentionally *not* an LLM. It covers the common phrasing patterns
a user (or an agent) is likely to supply to `/generate/from-text`:

    "age over 40, income >50K, works as Exec-managerial or Prof-specialty"
    "female, between 30 and 50 years old, hours per week at least 35"
    "married people earning <=50K with a bachelors degree"

Unrecognized clauses are collected as warnings so the caller can decide
whether to ask the user to rephrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from app.data.loader import DatasetSchema

# Map common synonyms / phrasings to dataset columns.
COLUMN_SYNONYMS: Dict[str, str] = {
    "age": "age",
    "aged": "age",
    "years old": "age",
    "year old": "age",
    "years of age": "age",

    "workclass": "workclass",
    "work class": "workclass",
    "work sector": "workclass",
    "private sector": "workclass",
    "sector": "workclass",

    "education": "education",
    "degree": "education",

    "education num": "education_num",
    "education_num": "education_num",
    "years of education": "education_num",

    "marital status": "marital_status",
    "marital_status": "marital_status",

    "occupation": "occupation",
    "job": "occupation",
    "profession": "occupation",
    "works as": "occupation",

    "relationship": "relationship",
    "race": "race",

    "sex": "sex",
    "gender": "sex",

    "capital gain": "capital_gain",
    "capital_gain": "capital_gain",
    "capital loss": "capital_loss",
    "capital_loss": "capital_loss",

    "hours per week": "hours_per_week",
    "hours_per_week": "hours_per_week",
    "weekly hours": "hours_per_week",
    "hours a week": "hours_per_week",

    "native country": "native_country",
    "native_country": "native_country",
    "country": "native_country",

    "income": "income",
    "salary": "income",
    "earnings": "income",
    "earning": "income",
    "earn": "income",
}

# Operators ordered so longer matches win.
_OP_PATTERNS: List[Tuple[str, str]] = [
    (">=", "gte"),
    ("<=", "lte"),
    ("=>", "gte"),
    ("=<", "lte"),
    (">", "gt"),
    ("<", "lt"),
    ("==", "eq"),
    ("=", "eq"),
]

_WORD_OPS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(at least|no less than|minimum of|greater than or equal to)\b", re.I), "gte"),
    (re.compile(r"\b(at most|no more than|maximum of|up to|less than or equal to)\b", re.I), "lte"),
    (re.compile(r"\b(greater than|more than|over|above|older than)\b", re.I), "gt"),
    (re.compile(r"\b(less than|under|below|younger than|fewer than)\b", re.I), "lt"),
    (re.compile(r"\b(exactly|equal to|equals|is)\b", re.I), "eq"),
]

_RANGE_RX = re.compile(
    r"\bbetween\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\b", re.I
)
_RANGE_RX2 = re.compile(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)")

_INCOME_WORDS: Dict[str, str] = {
    "high income": ">50K",
    "high-income": ">50K",
    "high earners": ">50K",
    "high earner": ">50K",
    "wealthy": ">50K",
    "rich": ">50K",
    "low income": "<=50K",
    "low-income": "<=50K",
    "low earner": "<=50K",
    "low earners": "<=50K",
    "poor": "<=50K",
}

_GENDER_WORDS: Dict[str, str] = {
    "male": "Male",
    "female": "Female",
    "man": "Male",
    "woman": "Female",
    "men": "Male",
    "women": "Female",
}


@dataclass
class ParseResult:
    constraints: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _find_column(text: str, schema: DatasetSchema) -> Tuple[str, int, int] | None:
    """Find the longest column synonym match in `text`. Returns (column, start, end)."""
    best: Tuple[str, int, int] | None = None
    for phrase, col in COLUMN_SYNONYMS.items():
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", text, re.I):
            if col not in schema.columns:
                continue
            start, end = m.start(), m.end()
            if best is None or (end - start) > (best[2] - best[1]):
                best = (col, start, end)
    return best


def _parse_numeric_clause(
    clause: str, col: str
) -> Dict[str, float] | None:
    m = _RANGE_RX.search(clause)
    if not m:
        m = _RANGE_RX2.search(clause)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return {"gte": min(a, b), "lte": max(a, b)}

    # symbolic operator
    for op_str, key in _OP_PATTERNS:
        if op_str in clause:
            rest = clause.split(op_str, 1)[1]
            nums = re.findall(r"-?\d+(?:\.\d+)?", rest)
            if nums:
                return {key: float(nums[0])}

    # word operator
    for rx, key in _WORD_OPS:
        wm = rx.search(clause)
        if wm:
            rest = clause[wm.end():]
            nums = re.findall(r"-?\d+(?:\.\d+)?", rest)
            if nums:
                return {key: float(nums[0])}

    # bare number => equality
    nums = re.findall(r"-?\d+(?:\.\d+)?", clause)
    if nums:
        return {"eq": float(nums[0])}

    return None


def _parse_categorical_clause(
    clause: str, col: str, schema: DatasetSchema
) -> List[str] | None:
    allowed = schema.categorical_values.get(col, [])
    found: List[str] = []
    # Check each allowed value appears as a whole phrase (case-insensitive).
    # Use word-boundaries only when the value starts/ends with word chars;
    # for values like ">50K" or "<=50K" that start with punctuation, a simple
    # case-insensitive substring search is used instead.
    for v in allowed:
        if not v:
            continue
        escaped = re.escape(v)
        first_char = v[0]
        last_char = v[-1]
        lb = r"\b" if re.match(r"\w", first_char) else r"(?<!\w)"
        rb = r"\b" if re.match(r"\w", last_char) else r"(?!\w)"
        pat = rf"{lb}{escaped}{rb}"
        if re.search(pat, clause, re.I):
            found.append(v)
    if found:
        # Preserve discovery order but de-dup.
        seen: set = set()
        ordered = []
        for v in found:
            if v not in seen:
                ordered.append(v)
                seen.add(v)
        return ordered
    return None


def parse_text_to_constraints(text: str, schema: DatasetSchema) -> ParseResult:
    """Parse a natural-language sentence into a constraints dict.

    The parser is deliberately simple and deterministic; it never invokes an
    LLM. Clauses it cannot interpret are returned as warnings.
    """
    result = ParseResult()
    if not text:
        return result

    lowered = text.strip()

    # --- High-level shortcuts ------------------------------------------------
    for phrase, income in _INCOME_WORDS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered, re.I):
            result.constraints["income"] = income
            break
    for phrase, sex in _GENDER_WORDS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered, re.I):
            result.constraints["sex"] = sex
            break

    # --- Protect "between X and Y" before splitting on bare "and" -----------
    # Replace matched ranges with a placeholder so the "and" inside isn't
    # treated as a clause separator.
    protected = lowered
    _range_placeholder_map: Dict[str, str] = {}
    for m in _RANGE_RX.finditer(lowered):
        token = f"__RANGE_{len(_range_placeholder_map)}__"
        _range_placeholder_map[token] = m.group(0)
        protected = protected.replace(m.group(0), token)

    # --- Split on commas / standalone "and" / semicolons / relative "who" --
    raw_clauses = re.split(r",|;|\band\b|\bwho\b|\bwith\b|\bthat\b", protected, flags=re.I)
    clauses: List[str] = []
    for clause in raw_clauses:
        for token, original in _range_placeholder_map.items():
            clause = clause.replace(token, original)
        clause = clause.strip()
        if clause:
            clauses.append(clause)

    for clause in clauses:
        match = _find_column(clause, schema)
        if match is None:
            # Maybe the clause matches an allowed value directly (e.g. "female",
            # "Exec-managerial"). Try matching categorical values across columns.
            matched = False
            for col in schema.categorical_columns:
                vals = _parse_categorical_clause(clause, col, schema)
                if vals:
                    existing = result.constraints.get(col)
                    if isinstance(existing, list):
                        result.constraints[col] = sorted(set(existing) | set(vals))
                    elif isinstance(existing, str):
                        result.constraints[col] = sorted(set([existing] + vals))
                    else:
                        result.constraints[col] = vals if len(vals) > 1 else vals[0]
                    matched = True
                    break
            if not matched:
                result.warnings.append(f"could not parse clause: '{clause}'")
            continue

        col, _, _ = match
        if col in result.constraints:
            continue
        if col in schema.numeric_columns:
            parsed = _parse_numeric_clause(clause, col)
            if parsed is None:
                result.warnings.append(
                    f"no numeric operator or value found for column '{col}' in: '{clause}'"
                )
                continue
            existing = result.constraints.get(col)
            if isinstance(existing, dict):
                existing.update(parsed)
                result.constraints[col] = existing
            else:
                result.constraints[col] = parsed
        else:
            vals = _parse_categorical_clause(clause, col, schema)
            if not vals:
                result.warnings.append(
                    f"no known value for categorical column '{col}' in: '{clause}'"
                )
                continue
            existing = result.constraints.get(col)
            if isinstance(existing, list):
                result.constraints[col] = sorted(set(existing) | set(vals))
            elif isinstance(existing, str):
                result.constraints[col] = sorted(set([existing] + vals))
            else:
                result.constraints[col] = vals if len(vals) > 1 else vals[0]

    # --- Second pass: scan full text for "<column_word> <op_word> <number>"
    # patterns that the clause loop may have missed (e.g. when the clause had
    # a more prominent match on a different column).
    for phrase, col in COLUMN_SYNONYMS.items():
        if col in result.constraints:
            continue
        if col not in schema.columns or col not in schema.numeric_columns:
            continue
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", lowered, re.I):
            tail = lowered[m.end():]
            parsed = _parse_numeric_clause(tail, col)
            if parsed is not None:
                result.constraints[col] = parsed
                break

    # --- Third pass: bare "over/under/above/below <number>" without a column
    # keyword. If the number falls in a plausible age range and "age" isn't
    # already constrained, assume it refers to age. Only trigger when the
    # operator+number isn't already near another column synonym.
    if "age" not in result.constraints and "age" in schema.columns:
        # Collect numeric values already claimed by other constraints.
        _claimed_values: set = set()
        for v in result.constraints.values():
            if isinstance(v, dict):
                _claimed_values.update(float(x) for x in v.values())
        for rx, key in _WORD_OPS:
            wm = rx.search(lowered)
            if wm:
                rest = lowered[wm.end():]
                nums = re.findall(r"-?\d+(?:\.\d+)?", rest)
                if nums:
                    val = float(nums[0])
                    if val in _claimed_values:
                        continue
                    rng = schema.value_ranges.get("age", {})
                    if rng and rng["min"] <= val <= rng["max"]:
                        result.constraints["age"] = {key: val}
                        break

    return result
