from typing import Any

import ray

from .processor_interface import TracingProcessor
from .processors import BatchTraceProcessor, default_exporter
from .provider import DefaultTraceProvider
from .setup import get_trace_provider, set_trace_provider
from .spans import Span
from .traces import Trace

TRACING_AGGREGATOR_NAME = "openai_agents_tracing_aggregator"


class RemoteSpan:
    """A wrapper for span data that behaves like a Span for export purposes."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def export(self) -> dict[str, Any]:
        return self.data


class RemoteTrace:
    """A wrapper for trace data that behaves like a Trace for export purposes."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def export(self) -> dict[str, Any]:
        return self.data


@ray.remote(num_cpus=0)
class TracingAggregator:
    """
    A Ray actor that aggregates traces and spans from Ray workers and exports them.
    This actor should be created on the driver and its handle passed to workers.
    """

    def __init__(self):
        # We use the standard BatchTraceProcessor on the driver side.
        # It will use the default exporter (BackendSpanExporter) which reads env vars.
        self.processor = BatchTraceProcessor(exporter=default_exporter())

    def process_span(self, span_data: dict[str, Any]):
        """Receive a span from a worker and queue it for export."""
        self.processor.on_span_end(RemoteSpan(span_data))  # type: ignore

    def process_trace(self, trace_data: dict[str, Any]):
        """Receive a trace from a worker and queue it for export."""
        self.processor.on_trace_start(RemoteTrace(trace_data))  # type: ignore

    def flush(self):
        """Force flush the underlying processor."""
        self.processor.force_flush()

    def shutdown(self):
        """Shutdown the underlying processor."""
        self.processor.shutdown()

    def __ray_terminate__(self):
        """Flush any pending traces when the actor is gracefully terminated."""
        try:
            self.flush()
            self.shutdown()
        except Exception:
            pass


class RemoteTracingProcessor(TracingProcessor):
    """
    A TracingProcessor that forwards spans and traces to a remote TracingAggregator actor.
    This is intended to be used in Ray workers.
    """

    def __init__(self):
        self.aggregator = ray.get_actor(TRACING_AGGREGATOR_NAME)

    def on_trace_start(self, trace: Trace) -> None:
        data = trace.export()
        if data:
            self.aggregator.process_trace.remote(data)

    def on_trace_end(self, trace: Trace) -> None:
        # Traces are sent on start in the batch processor model
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        # Spans are sent on end
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        data = span.export()
        if data:
            self.aggregator.process_span.remote(data)

    def shutdown(self) -> None:
        # We don't shut down the aggregator from a worker
        pass

    def force_flush(self) -> None:
        # We don't force flush the aggregator from a worker
        pass





def setup_distributed_tracing():
    """
    Setup distributed tracing for Ray functions.
    """
    try:
        provider = get_trace_provider()
    except RuntimeError:
        # Provider not set, set a default one
        provider = DefaultTraceProvider()
        set_trace_provider(provider)

    processor = RemoteTracingProcessor()
    provider.set_processors([processor])

