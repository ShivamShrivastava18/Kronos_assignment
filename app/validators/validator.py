"""TSTR classifier + end-to-end validation orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.config import get_settings
from app.data.loader import DatasetSchema, TARGET_COLUMN
from app.validators.metrics import (
    js_per_categorical_column,
    ks_per_numeric_column,
)

# Minimum number of reference rows needed to use the filtered reference
# dataset. Below this, fall back to the full real dataset.
_MIN_REFERENCE_ROWS = 80
# Fidelity threshold used for verdict when TSTR is meaningless (target constrained).
_FIDELITY_PASS_THRESHOLD = 0.65

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier  # type: ignore
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


@dataclass
class _Encoded:
    X: np.ndarray
    y: np.ndarray
    cat_maps: Dict[str, LabelEncoder]
    target_encoder: LabelEncoder
    feature_cols: List[str]


def _build_encoders(
    real: pd.DataFrame, schema: DatasetSchema
) -> Tuple[Dict[str, LabelEncoder], LabelEncoder, List[str]]:
    cat_maps: Dict[str, LabelEncoder] = {}
    feature_cols = [c for c in real.columns if c != TARGET_COLUMN]
    for col in feature_cols:
        if col in schema.categorical_columns:
            le = LabelEncoder()
            # Include "UNKNOWN" sentinel to absorb unseen synthetic categories.
            known = list(schema.categorical_values.get(col, [])) + ["__UNKNOWN__"]
            le.fit(known)
            cat_maps[col] = le
    target_enc = LabelEncoder()
    target_enc.fit(schema.categorical_values[TARGET_COLUMN])
    return cat_maps, target_enc, feature_cols


def _encode(
    df: pd.DataFrame,
    cat_maps: Dict[str, LabelEncoder],
    target_enc: LabelEncoder,
    feature_cols: List[str],
    schema: DatasetSchema,
) -> _Encoded:
    df = df.copy()
    # Ensure target exists.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"synthetic data is missing target column '{TARGET_COLUMN}'"
        )

    # Canonicalize unknown categoricals to the sentinel.
    for col, le in cat_maps.items():
        if col not in df.columns:
            df[col] = "__UNKNOWN__"
            continue
        col_vals = df[col].astype(str).fillna("__UNKNOWN__")
        allowed = set(le.classes_)
        col_vals = col_vals.where(col_vals.isin(allowed), "__UNKNOWN__")
        df[col] = col_vals

    # Coerce numerics.
    for col in schema.numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in feature_cols if c in schema.numeric_columns])
    df = df.dropna(subset=[TARGET_COLUMN])

    X_parts: List[np.ndarray] = []
    for col in feature_cols:
        if col in cat_maps:
            X_parts.append(cat_maps[col].transform(df[col].astype(str)).reshape(-1, 1))
        else:
            X_parts.append(df[col].to_numpy().reshape(-1, 1).astype(float))
    X = np.hstack(X_parts) if X_parts else np.zeros((len(df), 0))

    # Filter target to known classes.
    df = df[df[TARGET_COLUMN].astype(str).isin(target_enc.classes_)]
    y = target_enc.transform(df[TARGET_COLUMN].astype(str))
    # Re-slice X to match the filtered rows.
    X = X[: len(y)]
    return _Encoded(
        X=X, y=y, cat_maps=cat_maps, target_encoder=target_enc, feature_cols=feature_cols
    )


def _make_classifier():
    """Return the best available gradient boosting classifier.

    Preference order: XGBoost > LightGBM > sklearn GradientBoosting.
    The sklearn fallback is slower but has no native-library dependency,
    making it safe for local development on macOS without libomp.
    """
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            n_jobs=2,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
    if _HAS_LGBM:
        return LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=63,
            n_jobs=2,
            random_state=42,
            verbose=-1,
        )
    # Pure-Python fallback — no OpenMP required.
    from sklearn.ensemble import GradientBoostingClassifier

    logger.warning(
        "XGBoost and LightGBM unavailable (missing libomp?). "
        "Falling back to sklearn GradientBoostingClassifier — TSTR may be slower."
    )
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )


@dataclass
class ValidationResult:
    tstr_accuracy: float
    baseline_accuracy: float
    fidelity_score: float
    verdict: str
    tolerance: float
    ks: List[Dict]
    js: List[Dict]
    n_synthetic: int
    n_real_test: int
    notes: List[str]


_BASELINE_CACHE: Dict[str, float] = {}


def _baseline_accuracy(real: pd.DataFrame, schema: DatasetSchema) -> float:
    """Train on real train split, evaluate on real test split. Cached."""
    cache_key = f"{len(real)}:{tuple(real.columns)}"
    if cache_key in _BASELINE_CACHE:
        return _BASELINE_CACHE[cache_key]

    cat_maps, target_enc, feature_cols = _build_encoders(real, schema)
    enc = _encode(real, cat_maps, target_enc, feature_cols, schema)
    X_train, X_test, y_train, y_test = train_test_split(
        enc.X, enc.y, test_size=0.25, random_state=42, stratify=enc.y
    )
    model = _make_classifier()
    model.fit(X_train, y_train)
    acc = float(accuracy_score(y_test, model.predict(X_test)))
    _BASELINE_CACHE[cache_key] = acc
    return acc


def run_validation(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: DatasetSchema,
    *,
    reference: Optional[pd.DataFrame] = None,
    constrained_columns: Optional[Set[str]] = None,
) -> ValidationResult:
    """Run TSTR + KS/JS validation.

    Parameters
    ----------
    real:
        Full real dataset (used to build encoders and the baseline).
    synthetic:
        Synthetic rows to evaluate.
    schema:
        Dataset schema.
    reference:
        Optional filtered subset of `real` that matches the generation
        constraints. When provided (and large enough), KS/JS and the TSTR
        test set are drawn from this subset rather than the full dataset,
        making the comparison fair for constrained generation.
    constrained_columns:
        Set of column names that were explicitly constrained by the user.
        Used to decide the verdict strategy when the target column is locked.
    """
    notes: List[str] = []
    settings = get_settings()
    tolerance = settings.tstr_tolerance
    constrained_columns = constrained_columns or set()

    if len(synthetic) < 50:
        notes.append(
            f"synthetic dataset has only {len(synthetic)} rows; TSTR may be noisy"
        )

    # ── Decide which real subset to compare against ─────────────────────────
    # When the user constrained columns (e.g. sex=Female, age≥40), the fair
    # comparison is real rows that satisfy the same constraints, not the full
    # real dataset that contains the opposite of everything the user filtered.
    use_reference = reference is not None and len(reference) >= _MIN_REFERENCE_ROWS
    compare_df = reference if use_reference else real

    if use_reference:
        notes.append(
            f"comparison uses {len(reference):,} real rows matching your constraints "
            f"(out of {len(real):,} total); KS/JS and TSTR reflect constraint-conditioned fidelity."
        )
    elif reference is not None:
        notes.append(
            f"only {len(reference)} real rows match your constraints (< {_MIN_REFERENCE_ROWS} minimum); "
            "falling back to full real dataset for comparison — metrics may look pessimistic."
        )

    # ── Baseline (always trained on full real, cached) ───────────────────────
    baseline = _baseline_accuracy(real, schema)

    # ── Encoders built from full real so label spaces are complete ───────────
    cat_maps, target_enc, feature_cols = _build_encoders(real, schema)

    # ── Build TSTR test set from the reference (or full real) ────────────────
    enc_compare = _encode(compare_df, cat_maps, target_enc, feature_cols, schema)
    n_classes_compare = len(np.unique(enc_compare.y))

    if n_classes_compare >= 2:
        _, X_test, _, y_test = train_test_split(
            enc_compare.X, enc_compare.y,
            test_size=0.25, random_state=42, stratify=enc_compare.y,
        )
    else:
        # Only one target class in the reference — use all rows as test.
        X_test, y_test = enc_compare.X, enc_compare.y

    # ── Encode synthetic ─────────────────────────────────────────────────────
    try:
        enc_syn = _encode(synthetic, cat_maps, target_enc, feature_cols, schema)
    except ValueError as e:
        raise ValueError(f"synthetic data could not be encoded: {e}") from e

    # ── TSTR ─────────────────────────────────────────────────────────────────
    target_is_constrained = TARGET_COLUMN in constrained_columns
    n_classes_syn = len(np.unique(enc_syn.y))

    if n_classes_syn < 2:
        if target_is_constrained:
            notes.append(
                f"'{TARGET_COLUMN}' was constrained to a single value; "
                "TSTR is not applicable (a model trained on one class has no decision boundary). "
                "Verdict is based on fidelity score instead."
            )
        else:
            notes.append(
                "synthetic data has only one target class; TSTR model cannot learn a "
                "decision boundary — verdict falls back to fidelity score."
            )
        tstr_acc = float(np.mean(y_test == np.unique(enc_syn.y)[0])) if len(y_test) > 0 else 0.0
        tstr_meaningful = False
    else:
        model = _make_classifier()
        model.fit(enc_syn.X, enc_syn.y)
        tstr_acc = float(accuracy_score(y_test, model.predict(X_test)))
        tstr_meaningful = True

    # ── KS / JS — compared against reference (or full real) ─────────────────
    ks_list = ks_per_numeric_column(compare_df, synthetic, schema)
    js_list = js_per_categorical_column(compare_df, synthetic, schema)

    mean_ks = float(np.nanmean([m["statistic"] for m in ks_list]) if ks_list else 0.0)
    mean_js = float(np.nanmean([m["statistic"] for m in js_list]) if js_list else 0.0)

    similarity = 1.0 - 0.5 * (mean_ks + mean_js)  # KS, JS both in [0,1]; lower = better
    tstr_ratio = 1.0 - min(abs(tstr_acc - baseline) / max(baseline, 1e-6), 1.0)

    if tstr_meaningful:
        fidelity = float(np.clip(0.5 * similarity + 0.5 * tstr_ratio, 0.0, 1.0))
        verdict = "PASS" if abs(tstr_acc - baseline) <= tolerance else "FAIL"
    else:
        # TSTR is degenerate — base fidelity and verdict purely on distributional similarity.
        fidelity = float(np.clip(similarity, 0.0, 1.0))
        verdict = "PASS" if fidelity >= _FIDELITY_PASS_THRESHOLD else "FAIL"

    logger.info(
        "validation: tstr=%.4f baseline=%.4f fidelity=%.4f verdict=%s "
        "use_reference=%s n_reference=%s",
        tstr_acc, baseline, fidelity, verdict,
        use_reference, len(reference) if reference is not None else "N/A",
    )

    return ValidationResult(
        tstr_accuracy=tstr_acc,
        baseline_accuracy=baseline,
        fidelity_score=fidelity,
        verdict=verdict,
        tolerance=tolerance,
        ks=ks_list,
        js=js_list,
        n_synthetic=int(len(synthetic)),
        n_real_test=int(len(y_test)),
        notes=notes,
    )
