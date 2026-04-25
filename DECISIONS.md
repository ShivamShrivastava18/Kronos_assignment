# Design Decisions & Trade-offs

## 1. Gaussian Copula — conditional sampling vs. rejection sampling

**Decision:** The copula uses closed-form conditional multivariate-normal sampling
rather than rejection sampling.

For a constrained column *I*, we draw latent scores *Z_I* uniformly inside the
CDF interval corresponding to the constraint, then compute the analytical
conditional distribution *Z_J | Z_I* for the remaining columns. This preserves
realistic inter-column correlations even under minority constraints (e.g.
`income >50K, age >60`) where pure rejection sampling would require generating
10–100× more rows and still produce distribution collapse.

**Trade-off:** The approximation is only exact when the joint is truly Gaussian.
The UCI dataset's marginals are highly non-Gaussian (skewed numerics, categorical
columns), so we apply a rank-based normalizing transform (empirical CDF → Φ⁻¹).
This is the standard Gaussian copula construction and works well in practice, but
it cannot capture tail dependences beyond what Pearson correlation captures.

## 2. Two generation strategies

**Copula** (default): Better correlation fidelity; correct choice when the user
wants realistic inter-column relationships. Slower to fit once (~1 s).

**Histogram**: Bootstrap-resamples real rows that match the constraints. By
construction it can only produce rows that were present in the training data
(modulo the continuous numeric sampling), so it can never extrapolate. It is
fast, deterministic, and a natural fallback when the copula produces poor TSTR
scores for extreme constraints.

## 3. TSTR baseline caching

The baseline accuracy (train-on-real, test-on-real) is computed once per process
and cached. Re-computing it for every `/validate` call would double the
latency. The per-session nature of the in-memory store means the baseline
resets on container restart, which is acceptable for this scope.

## 4. Classifier choice: XGBoost with LightGBM fallback

XGBoost is the default TSTR classifier because it handles mixed types robustly
with `tree_method="hist"` and is fast enough for a 32K-row dataset at 150
estimators. LightGBM is retained as a fallback in case the `xgboost` wheel is
unavailable on the target architecture.

## 5. Deterministic NL text parser

The spec explicitly requires `/generate/from-text` to use a rule-based parser,
not an LLM. The parser covers operator synonyms (≥ / "at least" / "no less
than"), range phrasings ("between 30 and 50"), gender shortcuts, and income
shortcuts. Unrecognised clauses are returned as warnings rather than silently
dropped so the caller can surface them.

## 6. In-memory storage

Generations, validation results, and agent sessions are stored in Python dicts
behind an `RLock`. This is appropriate for a single-worker `uvicorn` process
(as started by `docker-compose up`) and eliminates external dependencies. For
multi-worker deployments these stores would be swapped for Redis.

## 7. Known limitations

- **Memory**: Large `n_rows` (>50 K) may exhaust container RAM; the API caps
  at 100 K rows.
- **Copula fit time**: ~1 s at startup; the service warms up generators in the
  lifespan hook so the first request is instant.
- **TSTR noise**: With fewer than ~200 synthetic rows the TSTR accuracy estimate
  is noisy; the API surfaces a warning in `notes`.
- **Text parser coverage**: The rule-based parser cannot handle negations
  ("not in New York") or complex boolean logic ("age > 40 OR income >50K").
  These cases are best handled via the structured `/generate` endpoint or by
  asking the agent to decompose the request.
