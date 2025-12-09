"""Thin wrapper for backwards compatibility. The real implementation lives in the pipeline package."""

from .pipeline import run_query_generation

__all__ = ["run_query_generation"]
