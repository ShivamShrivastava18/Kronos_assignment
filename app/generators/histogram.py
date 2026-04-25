"""Conditional histogram sampler — a robust fallback generator.

Idea: fit per-column empirical distributions conditioned on the constrained
columns. For any column not in the constraint set we draw from the marginal
distribution restricted to the subset of training rows that satisfy the
constraints. This guarantees hard-constraint satisfaction and is cheap.

Compared to the copula it under-captures cross-column correlations for the
*free* columns, but it is very reliable for minority constraints where the
copula's covariance estimate becomes noisy.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from app.data.loader import DatasetSchema
from app.generators.base import BaseGenerator
from app.generators.constraints import NormalizedConstraints


_INT_COLS = {
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "fnlwgt",
}


class ConditionalHistogramGenerator(BaseGenerator):
    name = "histogram"

    def __init__(self, schema: DatasetSchema) -> None:
        super().__init__(schema)
        self._df: pd.DataFrame | None = None
        self._modes: Dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "ConditionalHistogramGenerator":
        self._df = df.copy()
        for col in df.columns:
            if col in self.schema.numeric_columns:
                self._modes[col] = float(
                    pd.to_numeric(df[col], errors="coerce").median()
                )
            else:
                self._modes[col] = df[col].mode(dropna=True).iloc[0]
        self._fitted = True
        return self

    def sample(
        self,
        n: int,
        constraints: NormalizedConstraints,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        assert self._df is not None
        subset = constraints.filter(self._df)

        if subset.empty:
            # No real rows matched the constraint. Back off to a per-column
            # draw: respect constraints on constrained columns, use global
            # marginals for the rest.
            subset = self._df

        out: Dict[str, np.ndarray] = {}
        for col in self._df.columns:
            if col in constraints.numeric:
                rng_c = constraints.numeric[col]
                lo = rng_c.gte if rng_c.gte is not None else -np.inf
                hi = rng_c.lte if rng_c.lte is not None else np.inf
                vals = pd.to_numeric(subset[col], errors="coerce").dropna().to_numpy()
                vals = vals[(vals >= lo) & (vals <= hi)]
                if len(vals) == 0:
                    # Uniform over the constrained interval using observed extrema.
                    data_lo = self.schema.value_ranges[col]["min"]
                    data_hi = self.schema.value_ranges[col]["max"]
                    lo2 = max(lo, data_lo) if np.isfinite(lo) else data_lo
                    hi2 = min(hi, data_hi) if np.isfinite(hi) else data_hi
                    sampled = rng.uniform(lo2, hi2, size=n)
                else:
                    sampled = rng.choice(vals, size=n, replace=True)
                if col in _INT_COLS:
                    sampled = np.rint(sampled).astype(int)
                out[col] = sampled

            elif col in constraints.categorical:
                allowed = constraints.categorical[col]
                s = subset[col].dropna()
                s = s[s.isin(allowed)]
                if s.empty:
                    # Fall back to uniform over allowed values.
                    sampled = rng.choice(allowed, size=n, replace=True)
                else:
                    counts = s.value_counts(normalize=True)
                    probs = counts.values.astype(float)
                    cats = counts.index.tolist()
                    sampled = rng.choice(cats, size=n, replace=True, p=probs)
                out[col] = np.asarray(sampled, dtype=object)

            else:
                s = subset[col].dropna()
                if s.empty:
                    s = self._df[col].dropna()
                vals = s.to_numpy()
                sampled = rng.choice(vals, size=n, replace=True)
                if col in self.schema.numeric_columns:
                    sampled = pd.to_numeric(sampled, errors="coerce")
                    sampled = np.where(np.isnan(sampled), self._modes[col], sampled)
                    if col in _INT_COLS:
                        sampled = np.rint(sampled).astype(int)
                out[col] = sampled

        return pd.DataFrame(out, columns=list(self._df.columns))
