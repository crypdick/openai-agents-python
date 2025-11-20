"""Centralized Ray setup and initialization."""

import os
import sys
from pathlib import Path
from typing import Any

try:
    import ray

    from agents.tracing.ray_exporter import TRACING_AGGREGATOR_NAME, TracingAggregator
except ImportError:
    ray = None  # type: ignore[assignment]

_ray_initialized = False


def _get_project_root() -> str:
    # __file__ = <repo>/src/agents/setup_ray.py
    # parents[0] = src/agents, parents[1] = src, parents[2] = repo root
    return str(Path(__file__).resolve().parents[2])


def ensure_ray_initialized() -> Any:
    if not ray:
        # TODO: log warning
        return

    global _ray_initialized
    if not _ray_initialized:
        if ray.is_initialized():
            pass
        else:
            ray.init(
                runtime_env={
                    "working_dir": _get_project_root(),
                }
            )

        _ray_initialized = True

    # Get or create the tracing aggregator
    aggregator = TracingAggregator.options(  # type: ignore[attr-defined]
        name=TRACING_AGGREGATOR_NAME, get_if_exists=True
    ).remote()
    return aggregator


def use_ray() -> bool:
    """Return True if Ray backend should be used (env-controlled) and Ray is available."""
    return (ray is not None) and (os.environ.get("RAY_BACKEND") == "1")
