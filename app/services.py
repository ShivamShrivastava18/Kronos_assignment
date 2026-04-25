"""Service layer shared by HTTP routes and the LLM agent tools."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.data.loader import DatasetSchema, get_schema, load_dataset
from app.generators import AVAILABLE_STRATEGIES, BaseGenerator
from app.generators.constraints import NormalizedConstraints, normalize_constraints
from app.storage import GenerationRecord, ValidationRecord, store
from app.validators.validator import ValidationResult, run_validation, _MIN_REFERENCE_ROWS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator cache — fitting the Copula is the expensive step (~1s). We fit
# once per process and reuse.
# ---------------------------------------------------------------------------

_fit_lock = RLock()
_generators: Dict[str, BaseGenerator] = {}


def get_generator(strategy: str) -> BaseGenerator:
    if strategy not in AVAILABLE_STRATEGIES:
        raise ValueError(
            f"unknown strategy '{strategy}'. Available: {list(AVAILABLE_STRATEGIES)}"
        )
    with _fit_lock:
        if strategy in _generators:
            return _generators[strategy]
        logger.info("Fitting %s generator...", strategy)
        schema = get_schema()
        df = load_dataset()
        gen = AVAILABLE_STRATEGIES[strategy](schema).fit(df)
        _generators[strategy] = gen
        return gen


def warm_up() -> None:
    """Called at startup to avoid paying the fit cost on the first request."""
    try:
        get_generator("copula")
    except Exception as e:  # pragma: no cover
        logger.exception("copula warm-up failed: %s", e)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class GenerationOutput:
    generation_id: str
    strategy: str
    n_rows_requested: int
    rows: List[Dict[str, Any]]
    constraints_applied: Dict[str, Any]
    constraints_rejected: Dict[str, str]
    random_seed: Optional[int]
    rows_df: pd.DataFrame


def _coerce_rows_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Convert numpy types to native Python so JSON serialization is clean.
    rows = df.where(pd.notna(df), None).to_dict(orient="records")
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.floating,)):
                r[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                r[k] = bool(v)
    return rows


def generate(
    constraints: Dict[str, Any],
    n_rows: int,
    strategy: str,
    random_seed: Optional[int] = None,
) -> GenerationOutput:
    schema = get_schema()
    normalized = normalize_constraints(constraints, schema)

    gen = get_generator(strategy)
    rng_seed = random_seed

    # Generate with oversampling to compensate for constraint filtering and
    # drop any rows that slipped past (possible for copula with numeric ranges
    # near the marginal tails).
    generated: List[pd.DataFrame] = []
    needed = n_rows
    multiplier = 1.25
    attempts = 0
    current_seed = rng_seed
    while needed > 0 and attempts < 6:
        batch_n = max(needed, int(needed * multiplier)) if attempts else n_rows
        df = gen.generate(batch_n, normalized, random_seed=current_seed)
        df = normalized.filter(df)
        generated.append(df)
        needed = n_rows - sum(len(b) for b in generated)
        attempts += 1
        multiplier *= 1.5
        current_seed = None  # re-randomize batches after the first

    result = pd.concat(generated, ignore_index=True).head(n_rows)
    if len(result) < n_rows:
        logger.warning(
            "Only produced %d/%d rows matching constraints after %d attempts",
            len(result),
            n_rows,
            attempts,
        )

    generation_id = str(uuid.uuid4())
    record = GenerationRecord(
        generation_id=generation_id,
        strategy=strategy,
        n_rows=len(result),
        random_seed=rng_seed,
        constraints_applied=normalized.as_dict(),
        constraints_rejected=normalized.rejected,
        rows_df=result,
    )
    store.put_generation(record)

    return GenerationOutput(
        generation_id=generation_id,
        strategy=strategy,
        n_rows_requested=n_rows,
        rows=_coerce_rows_to_records(result),
        constraints_applied=normalized.as_dict(),
        constraints_rejected=normalized.rejected,
        random_seed=rng_seed,
        rows_df=result,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_rows(
    rows: pd.DataFrame,
    generation_id: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> Tuple[ValidationResult, ValidationRecord]:
    """Run validation, optionally using constraints to build a fair reference set.

    When `constraints` is provided (the dict from GenerationRecord.constraints_applied),
    the real dataset is filtered to rows that satisfy the same constraints. KS/JS
    and the TSTR test set are drawn from this filtered reference so the comparison
    is apples-to-apples: constrained synthetic vs constrained real.
    """
    schema: DatasetSchema = get_schema()
    real = load_dataset().copy()

    reference: Optional[pd.DataFrame] = None
    constrained_columns: set = set()

    if constraints:
        try:
            normalized = normalize_constraints(constraints, schema)
            constrained_columns = set(normalized.numeric) | set(normalized.categorical)
            filtered = normalized.filter(real)
            if len(filtered) >= _MIN_REFERENCE_ROWS:
                reference = filtered
                logger.info(
                    "validate_rows: using %d/%d real rows as constrained reference",
                    len(filtered), len(real),
                )
            else:
                logger.info(
                    "validate_rows: only %d real rows match constraints (< %d); "
                    "using full real dataset",
                    len(filtered), _MIN_REFERENCE_ROWS,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("validate_rows: could not build reference from constraints: %s", exc)

    result: ValidationResult = run_validation(
        real, rows, schema,
        reference=reference,
        constrained_columns=constrained_columns,
    )

    payload = _validation_to_dict(result, generation_id)
    record = ValidationRecord(
        generation_id=generation_id,
        result=payload,
        verdict=result.verdict,
        fidelity_score=result.fidelity_score,
    )
    store.put_validation(record)
    return result, record


def _validation_to_dict(
    result: ValidationResult, generation_id: Optional[str]
) -> Dict[str, Any]:
    return {
        "generation_id": generation_id,
        "tstr_accuracy": result.tstr_accuracy,
        "baseline_accuracy": result.baseline_accuracy,
        "fidelity_score": result.fidelity_score,
        "verdict": result.verdict,
        "tolerance": result.tolerance,
        "ks_tests": [
            {
                "column": m["column"],
                "metric": m["metric"],
                "statistic": m["statistic"],
                "p_value": m.get("p_value"),
            }
            for m in result.ks
        ],
        "js_divergences": [
            {
                "column": m["column"],
                "metric": m["metric"],
                "statistic": m["statistic"],
                "p_value": None,
            }
            for m in result.js
        ],
        "n_synthetic": result.n_synthetic,
        "n_real_test": result.n_real_test,
        "notes": result.notes,
    }


def validation_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compact dict useful for agent replies."""
    return {
        "generation_id": payload.get("generation_id"),
        "verdict": payload["verdict"],
        "tstr_accuracy": round(payload["tstr_accuracy"], 4),
        "baseline_accuracy": round(payload["baseline_accuracy"], 4),
        "fidelity_score": round(payload["fidelity_score"], 4),
    }
