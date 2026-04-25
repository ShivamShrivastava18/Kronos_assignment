"""Base class for synthetic data generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from app.data.loader import DatasetSchema
from app.generators.constraints import NormalizedConstraints


class BaseGenerator(ABC):
    """Subclasses must implement `.fit()` and `.sample()`."""

    name: str = "base"

    def __init__(self, schema: DatasetSchema) -> None:
        self.schema = schema
        self._fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseGenerator": ...

    @abstractmethod
    def sample(
        self,
        n: int,
        constraints: NormalizedConstraints,
        rng: np.random.Generator,
    ) -> pd.DataFrame: ...

    def generate(
        self,
        n: int,
        constraints: NormalizedConstraints,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name} generator is not fitted")
        rng = np.random.default_rng(random_seed)
        return self.sample(n, constraints, rng)
