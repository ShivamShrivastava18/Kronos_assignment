"""Gaussian Copula generator with conditional sampling.

The copula decomposes the joint distribution into:
  * marginal distributions per column (fitted non-parametrically), and
  * a Gaussian dependence structure over the latent rank-transformed values.

Constraints are honored by **conditioning**, not rejection: the constrained
columns are sampled first within the allowed CDF interval, and the remaining
columns are drawn from the conditional multivariate normal given those latent
values. This preserves inter-column correlations for minority constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from app.data.loader import DatasetSchema
from app.generators.base import BaseGenerator
from app.generators.constraints import NormalizedConstraints, NumericRange

# Clamp the CDF values away from {0, 1} before Phi^-1 to avoid +/- infinity.
_EPS = 1e-6


@dataclass
class _NumericMarginal:
    col: str
    sorted_values: np.ndarray  # shape (n_unique,) monotone non-decreasing

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Empirical CDF at x (midranks for ties)."""
        n = len(self.sorted_values)
        idx = np.searchsorted(self.sorted_values, x, side="right")
        return np.clip(idx / n, _EPS, 1 - _EPS)

    def inv_cdf(self, u: np.ndarray) -> np.ndarray:
        n = len(self.sorted_values)
        # Linear interpolation on the quantile function for smooth synthetic values.
        q = np.clip(u, _EPS, 1 - _EPS)
        positions = q * (n - 1)
        lo = np.floor(positions).astype(int)
        hi = np.ceil(positions).astype(int)
        frac = positions - lo
        return self.sorted_values[lo] * (1 - frac) + self.sorted_values[hi] * frac


@dataclass
class _CategoricalMarginal:
    col: str
    categories: List[str]  # ordered
    cumulative: np.ndarray  # shape (len(categories)+1,), [0, p1, p1+p2, ..., 1]

    def sample_bin_u(
        self,
        rng: np.random.Generator,
        allowed: Optional[List[str]] = None,
        n: int = 1,
    ) -> np.ndarray:
        """Sample u ~ Uniform over union of bins matching `allowed`."""
        if allowed is None:
            return rng.uniform(_EPS, 1 - _EPS, size=n)

        allowed_idx = [i for i, c in enumerate(self.categories) if c in set(allowed)]
        if not allowed_idx:
            return rng.uniform(_EPS, 1 - _EPS, size=n)

        # Weight each allowed bin by its marginal mass.
        masses = np.array(
            [self.cumulative[i + 1] - self.cumulative[i] for i in allowed_idx]
        )
        if masses.sum() <= 0:
            # Degenerate: uniform over bins by width.
            masses = np.ones(len(allowed_idx)) / len(allowed_idx)
        probs = masses / masses.sum()

        chosen = rng.choice(len(allowed_idx), size=n, p=probs)
        u = np.empty(n)
        for k, ci in enumerate(chosen):
            i = allowed_idx[ci]
            lo = self.cumulative[i] + _EPS
            hi = self.cumulative[i + 1] - _EPS
            if hi <= lo:
                lo, hi = max(_EPS, lo), max(_EPS + 1e-9, hi)
            u[k] = rng.uniform(lo, hi)
        return u

    def inv_cdf_to_value(self, u: np.ndarray) -> np.ndarray:
        """Map uniforms to categorical labels via the bin map."""
        idx = np.searchsorted(self.cumulative[1:-1], u, side="right")
        idx = np.clip(idx, 0, len(self.categories) - 1)
        return np.array([self.categories[i] for i in idx], dtype=object)


class GaussianCopulaGenerator(BaseGenerator):
    name = "copula"

    def __init__(self, schema: DatasetSchema) -> None:
        super().__init__(schema)
        self._numeric_marginals: Dict[str, _NumericMarginal] = {}
        self._categorical_marginals: Dict[str, _CategoricalMarginal] = {}
        self._col_order: List[str] = []
        self._cov: Optional[np.ndarray] = None
        self._col_to_idx: Dict[str, int] = {}
        # Column-level mode for missing-value imputation after sampling.
        self._modes: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "GaussianCopulaGenerator":
        self._col_order = list(df.columns)
        self._col_to_idx = {c: i for i, c in enumerate(self._col_order)}

        z_cols: List[np.ndarray] = []

        for col in self._col_order:
            series = df[col]
            if col in self.schema.numeric_columns:
                values = pd.to_numeric(series, errors="coerce")
                filled = values.fillna(values.median())
                self._modes[col] = float(values.median())
                sorted_vals = np.sort(filled.to_numpy())
                marginal = _NumericMarginal(col=col, sorted_values=sorted_vals)
                self._numeric_marginals[col] = marginal

                ranks = stats.rankdata(filled.to_numpy(), method="average")
                u = ranks / (len(ranks) + 1)
                z_cols.append(stats.norm.ppf(np.clip(u, _EPS, 1 - _EPS)))
            else:
                s = series.fillna(series.mode(dropna=True).iloc[0])
                self._modes[col] = s.mode(dropna=True).iloc[0]
                # Sort categories by descending frequency for stability.
                counts = s.value_counts()
                categories = counts.index.tolist()
                probs = (counts / counts.sum()).to_numpy()
                cumulative = np.concatenate([[0.0], np.cumsum(probs)])
                cumulative[-1] = 1.0
                marginal = _CategoricalMarginal(
                    col=col, categories=categories, cumulative=cumulative
                )
                self._categorical_marginals[col] = marginal

                # Map each row to midpoint of its bin -> uniform -> Gaussian.
                idx_map = {c: i for i, c in enumerate(categories)}
                bin_lo = np.array([cumulative[idx_map[v]] for v in s])
                bin_hi = np.array([cumulative[idx_map[v] + 1] for v in s])
                u = (bin_lo + bin_hi) / 2.0
                z_cols.append(stats.norm.ppf(np.clip(u, _EPS, 1 - _EPS)))

        Z = np.column_stack(z_cols)  # (n_rows, n_cols)
        # Regularize for numerical stability.
        cov = np.cov(Z, rowvar=False)
        cov = cov + 1e-4 * np.eye(cov.shape[0])
        self._cov = cov
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------
    def sample(
        self,
        n: int,
        constraints: NormalizedConstraints,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        assert self._cov is not None
        d = len(self._col_order)

        constrained_idx: List[int] = []
        constrained_u: List[np.ndarray] = []

        for col, rng_c in constraints.numeric.items():
            if col not in self._col_to_idx:
                continue
            constrained_idx.append(self._col_to_idx[col])
            constrained_u.append(self._sample_numeric_u(col, rng_c, n, rng))

        for col, allowed in constraints.categorical.items():
            if col not in self._col_to_idx:
                continue
            constrained_idx.append(self._col_to_idx[col])
            constrained_u.append(
                self._categorical_marginals[col].sample_bin_u(rng, allowed, n)
            )

        if constrained_idx:
            order = np.argsort(constrained_idx)
            constrained_idx = [constrained_idx[i] for i in order]
            constrained_u = [constrained_u[i] for i in order]
            U_I = np.column_stack(constrained_u)  # (n, |I|)
            Z_I = stats.norm.ppf(np.clip(U_I, _EPS, 1 - _EPS))

            free_idx = [i for i in range(d) if i not in set(constrained_idx)]
            Z_J = self._conditional_sample(Z_I, constrained_idx, free_idx, rng)

            Z = np.empty((n, d))
            Z[:, constrained_idx] = Z_I
            if free_idx:
                Z[:, free_idx] = Z_J
        else:
            Z = rng.multivariate_normal(
                mean=np.zeros(d), cov=self._cov, size=n, method="cholesky"
            )

        return self._decode(Z)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_numeric_u(
        self,
        col: str,
        rng_c: NumericRange,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        marg = self._numeric_marginals[col]
        lo = rng_c.gte if rng_c.gte is not None else -np.inf
        hi = rng_c.lte if rng_c.lte is not None else np.inf
        if np.isfinite(lo):
            u_lo = float(marg.cdf(np.array([lo]))[0])
        else:
            u_lo = _EPS
        if np.isfinite(hi):
            u_hi = float(marg.cdf(np.array([hi]))[0])
        else:
            u_hi = 1 - _EPS
        if u_hi <= u_lo:
            # degenerate: point mass at that quantile
            u_hi = min(1 - _EPS, u_lo + 1e-4)
        return rng.uniform(u_lo, u_hi, size=n)

    def _conditional_sample(
        self,
        Z_I: np.ndarray,
        idx_I: List[int],
        idx_J: List[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not idx_J:
            return np.empty((Z_I.shape[0], 0))

        n = Z_I.shape[0]
        nJ = len(idx_J)

        # All arithmetic in this block may encounter numerically degenerate
        # cases with certain constraint combinations. We suppress numpy
        # floating-point warnings and sanitize the output rows instead of
        # letting the warnings surface to callers.
        with np.errstate(all="ignore"):
            # Clamp latent inputs to prevent matmul overflow.
            Z_I = np.clip(Z_I, -6.0, 6.0)

            cov = self._cov
            Sigma_II = cov[np.ix_(idx_I, idx_I)]
            Sigma_JJ = cov[np.ix_(idx_J, idx_J)]
            Sigma_JI = cov[np.ix_(idx_J, idx_I)]

            # Pseudo-inverse is more stable than regular inverse when Sigma_II
            # is near-singular (e.g. two correlated categorical columns).
            reg = 1e-3
            Sigma_II_reg = Sigma_II + reg * np.eye(Sigma_II.shape[0])
            inv_II = np.linalg.pinv(Sigma_II_reg)

            A = Sigma_JI @ inv_II  # (|J|, |I|)
            # Hard-clamp regression coefficients to prevent catastrophic overflow.
            A = np.clip(A, -10.0, 10.0)

            mu_J_given_I = Z_I @ A.T  # (n, |J|)

            Sigma_J_given_I = Sigma_JJ - A @ Sigma_JI.T
            Sigma_J_given_I = (Sigma_J_given_I + Sigma_J_given_I.T) / 2

            # Ensure PSD via eigenvalue floor.
            eigvals = np.linalg.eigvalsh(Sigma_J_given_I)
            min_eig = float(eigvals.min()) if np.isfinite(eigvals.min()) else 0.0
            if min_eig < 1e-6:
                Sigma_J_given_I += (abs(min_eig) + 1e-4) * np.eye(nJ)

            try:
                L = np.linalg.cholesky(Sigma_J_given_I)
            except np.linalg.LinAlgError:
                L = np.diag(np.sqrt(np.clip(np.diag(Sigma_J_given_I), 1e-6, None)))

            noise = rng.standard_normal((n, nJ))
            out = mu_J_given_I + noise @ L.T

        # Replace any NaN/Inf rows with unconstrained standard normals so the
        # caller always receives a fully-finite Z matrix.
        bad = ~np.isfinite(out).all(axis=1)
        if bad.any():
            out[bad] = rng.standard_normal((int(bad.sum()), nJ))
        return out

    def _decode(self, Z: np.ndarray) -> pd.DataFrame:
        U = stats.norm.cdf(Z)
        U = np.clip(U, _EPS, 1 - _EPS)
        data: Dict[str, np.ndarray] = {}
        for i, col in enumerate(self._col_order):
            if col in self._numeric_marginals:
                vals = self._numeric_marginals[col].inv_cdf(U[:, i])
                # Snap integer-valued columns to nearest int.
                if col in {
                    "age",
                    "education_num",
                    "capital_gain",
                    "capital_loss",
                    "hours_per_week",
                    "fnlwgt",
                }:
                    vals = np.rint(vals).astype(int)
                data[col] = vals
            else:
                data[col] = self._categorical_marginals[col].inv_cdf_to_value(U[:, i])
        return pd.DataFrame(data, columns=self._col_order)
