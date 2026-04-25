"""UCI Adult Income dataset loader with local caching."""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from app.config import get_settings

logger = logging.getLogger(__name__)

# UCI adult.data has no header row; these are the canonical column names.
COLUMN_NAMES: List[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_COLUMNS: List[str] = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLUMNS: List[str] = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "income",
]

TARGET_COLUMN = "income"


@dataclass(frozen=True)
class DatasetSchema:
    """Machine-readable schema used by validators and the LLM agent."""

    columns: List[str]
    dtypes: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    value_ranges: Dict[str, Dict[str, float]]  # numeric -> {min, max, mean, std}
    categorical_values: Dict[str, List[str]]  # categorical -> allowed values
    class_distribution: Dict[str, float]  # income -> proportion
    n_rows: int
    target_column: str = TARGET_COLUMN


def _cache_path() -> str:
    settings = get_settings()
    os.makedirs(settings.data_dir, exist_ok=True)
    return os.path.join(settings.data_dir, "adult.data")


def ensure_dataset() -> str:
    """Download the UCI Adult dataset if not already cached. Returns local path."""
    path = _cache_path()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    settings = get_settings()
    logger.info("Downloading UCI Adult dataset from %s", settings.dataset_url)
    req = urllib.request.Request(
        settings.dataset_url,
        headers={"User-Agent": "synthgen/0.1 (+https://archive.ics.uci.edu)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        body = resp.read()
    with open(path, "wb") as f:
        f.write(body)
    return path


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and standardize missing value markers."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    # UCI uses "?" to denote missing values in categorical cells.
    df = df.replace({"?": np.nan})
    # Drop trailing dot that sometimes appears in income ("<=50K.")
    if "income" in df.columns:
        df["income"] = df["income"].str.rstrip(".")
    return df


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Load and clean the dataset once per process."""
    path = ensure_dataset()
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        na_values=["?"],
        engine="python",
    )
    df = _clean(df)
    # Drop rows that are completely blank (UCI file sometimes has a trailing newline).
    df = df.dropna(how="all").reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def get_schema() -> DatasetSchema:
    df = load_dataset()

    value_ranges: Dict[str, Dict[str, float]] = {}
    for col in NUMERIC_COLUMNS:
        s = df[col].dropna()
        value_ranges[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
        }

    categorical_values: Dict[str, List[str]] = {}
    for col in CATEGORICAL_COLUMNS:
        vals = sorted([v for v in df[col].dropna().unique().tolist()])
        categorical_values[col] = vals

    class_counts = df[TARGET_COLUMN].value_counts(normalize=True, dropna=False)
    class_distribution = {str(k): float(v) for k, v in class_counts.items()}

    dtypes = {
        col: ("numeric" if col in NUMERIC_COLUMNS else "categorical")
        for col in df.columns
    }

    return DatasetSchema(
        columns=list(df.columns),
        dtypes=dtypes,
        numeric_columns=list(NUMERIC_COLUMNS),
        categorical_columns=list(CATEGORICAL_COLUMNS),
        value_ranges=value_ranges,
        categorical_values=categorical_values,
        class_distribution=class_distribution,
        n_rows=int(len(df)),
    )


def sample_rows(n: int = 5, random_seed: int | None = None) -> List[Dict]:
    df = load_dataset()
    n = max(1, min(n, len(df)))
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(df), size=n, replace=False)
    rows = df.iloc[idx].where(pd.notna(df.iloc[idx]), None)
    return rows.to_dict(orient="records")
