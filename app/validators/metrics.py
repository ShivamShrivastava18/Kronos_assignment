"""Distribution-level realism metrics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from app.data.loader import DatasetSchema


def ks_per_numeric_column(
    real: pd.DataFrame, synthetic: pd.DataFrame, schema: DatasetSchema
) -> List[Dict[str, float]]:
    """Kolmogorov-Smirnov two-sample test per numeric column.

    Returns a list of dicts with keys {column, metric, statistic, p_value}.
    Lower statistic ⇒ distributions are more similar.
    """
    out: List[Dict[str, float]] = []
    for col in schema.numeric_columns:
        if col not in real.columns or col not in synthetic.columns:
            continue
        a = pd.to_numeric(real[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(synthetic[col], errors="coerce").dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            out.append(
                {
                    "column": col,
                    "metric": "ks",
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                }
            )
            continue
        res = stats.ks_2samp(a, b, alternative="two-sided", method="auto")
        out.append(
            {
                "column": col,
                "metric": "ks",
                "statistic": float(res.statistic),
                "p_value": float(res.pvalue),
            }
        )
    return out


def js_per_categorical_column(
    real: pd.DataFrame, synthetic: pd.DataFrame, schema: DatasetSchema
) -> List[Dict[str, float]]:
    """Jensen-Shannon divergence over each categorical column's PMF.

    Bounded in [0, log 2]; we report the square of `jensenshannon`'s output,
    which is the true divergence (SciPy's function returns the square root
    "distance"). Lower ⇒ more similar.
    """
    out: List[Dict[str, float]] = []
    for col in schema.categorical_columns:
        if col not in real.columns or col not in synthetic.columns:
            continue
        cats = schema.categorical_values.get(col, [])
        if not cats:
            cats = sorted(
                set(real[col].dropna().astype(str)) | set(synthetic[col].dropna().astype(str))
            )
        p = real[col].astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
        q = (
            synthetic[col]
            .astype(str)
            .value_counts()
            .reindex(cats, fill_value=0)
            .to_numpy(dtype=float)
        )
        if p.sum() == 0 or q.sum() == 0:
            out.append(
                {
                    "column": col,
                    "metric": "js",
                    "statistic": float("nan"),
                    "p_value": None,  # type: ignore[dict-item]
                }
            )
            continue
        p = p / p.sum()
        q = q / q.sum()
        dist = stats.entropy(p, 0.5 * (p + q), base=2) + stats.entropy(
            q, 0.5 * (p + q), base=2
        )
        js = 0.5 * dist  # JS divergence in bits, in [0, 1].
        out.append(
            {
                "column": col,
                "metric": "js",
                "statistic": float(js),
                "p_value": None,  # type: ignore[dict-item]
            }
        )
    return out
