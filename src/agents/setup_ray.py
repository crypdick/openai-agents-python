"""Centralized Ray setup and initialization."""

from typing import Any

try:
    import ray

    from agents.tracing.ray_exporter import TRACING_AGGREGATOR_NAME, TracingAggregator
except ImportError:
    ray = None

_ray_initialized = False


def ensure_ray_initialized() -> Any:
    if not ray:
        # TODO: log warning
        return

    global _ray_initialized
    if not _ray_initialized:
        if ray.is_initialized():
            pass
        else:
            ray.init()

        _ray_initialized = True

        # Ensure the tracing aggregator is running if Ray is available
        aggregator = TracingAggregator.options(
            name=TRACING_AGGREGATOR_NAME, get_if_exists=True
        ).remote()
        return aggregator

    else:
        return ray.get_actor(TRACING_AGGREGATOR_NAME)
