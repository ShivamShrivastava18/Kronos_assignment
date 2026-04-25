"""Synthetic data generation engines."""

from app.generators.base import BaseGenerator
from app.generators.copula import GaussianCopulaGenerator
from app.generators.histogram import ConditionalHistogramGenerator

AVAILABLE_STRATEGIES = {
    "copula": GaussianCopulaGenerator,
    "histogram": ConditionalHistogramGenerator,
}

__all__ = [
    "BaseGenerator",
    "GaussianCopulaGenerator",
    "ConditionalHistogramGenerator",
    "AVAILABLE_STRATEGIES",
]
