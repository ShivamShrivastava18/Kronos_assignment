"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Conditional Synthetic Tabular Data Generator"
    app_version: str = "0.1.0"

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # Dataset
    dataset_url: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    data_dir: str = os.path.join(os.getcwd(), "data")

    # Validation
    tstr_tolerance: float = 0.05  # +/- 5% of real baseline accuracy => PASS
    validation_n_synthetic: int = 2000

    log_level: str = "info"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
