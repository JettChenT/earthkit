from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from .modal_otel import ModalInstrumentor, OTEL_DEPS
from .cfig import ENVS
import modal

if modal.is_local():
    import os
    os.environ.update(ENVS)

resource = Resource(attributes={
    "service.name": "modal-otel"
})

tracer_provider = TracerProvider(
    resource=resource
)

otlp_exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)

def instrument():
    ModalInstrumentor().instrument(tracer_provider=tracer_provider)

def get_tracer():
    return trace.get_tracer(
        __name__,
        schema_url="https://opentelemetry.io/schemas/1.11.0",
        tracer_provider=tracer_provider
    )

