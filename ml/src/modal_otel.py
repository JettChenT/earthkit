from typing import Collection, Dict, Optional
import modal
from modal.app import FunctionInfo
import modal.client
import modal_proto.api_grpc
from opentelemetry import context
from opentelemetry.context import attach, detach
from opentelemetry.propagate import inject, extract
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.propagators import get_global_response_propagator
from opentelemetry.trace import SpanKind, TracerProvider, get_tracer
from opentelemetry.propagate import inject
import inspect
from modal import App, Function
from modal.functions import _Invocation
import logging
from functools import wraps

_logger = logging.getLogger(__name__)

OTEL_DEPS = ["opentelemetry-distro", "opentelemetry-exporter-otlp"]

class ModalInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return ["modal"]

    @staticmethod
    def instrument_app(app: App, tracer_provider: Optional[TracerProvider] = None):
        if app not in _InstrumentedApp.is_traced:
            _InstrumentedApp.providers[app] = tracer_provider
            object.__setattr__(app, "function", _InstrumentedApp.function.__get__(app))
            _InstrumentedApp.is_traced.add(app)
    
    @staticmethod
    def uninstrument_app(app: App):
        if hasattr(app, "_original_function"):
            app.function = _InstrumentedApp.original_function.__get__(app)
            _InstrumentedApp.providers.pop(app)
            _InstrumentedApp.is_traced.remove(app)
        else:
            _logger.warning("Attempt to uninstrument an app that is not instrumented")

    def _instrument(self, **kwargs):
        self.original_app_cls = modal.App
        self.original_stub_cls = modal.Stub
        self.original_invocation = modal.functions._Invocation
        _InstrumentedApp._tracer_provider = kwargs.get("tracer_provider")
        modal.App = _InstrumentedApp
        modal.Stub = _InstrumentedApp
        modal.functions._Invocation = InstrumentedInvocation
        return super()._instrument(**kwargs)
    
    def _uninstrument(self, **kwargs):
        modal.App = self.original_app_cls
        modal.Stub = self.original_stub_cls
        modal.functions._Invocation = self.original_invocation
        _InstrumentedApp._tracer_provider = None

class _InstrumentedApp(App):
    _tracer_provider = None
    is_traced: set[App] = set()
    providers: Dict[App, TracerProvider|None] = {}
    original_function = App.function

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def function(self, *args, **kwargs):
        if self in _InstrumentedApp.providers:
            tracer_provider = _InstrumentedApp.providers[self]
        else:
            tracer_provider = _InstrumentedApp._tracer_provider
        tracer = get_tracer(
            __name__,
            tracer_provider=tracer_provider,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        orig_bind = _InstrumentedApp.original_function.__get__(self)
        in_wrapped = orig_bind(*args, **kwargs)
        serialized = kwargs.get("serialized", False)
        name_override = kwargs.get("name_override", None)
        @wraps(in_wrapped)
        def wr_wrapped(f, _cls=None):
            f_info = FunctionInfo(
                f, serialized=serialized, name_override=name_override, cls=_cls
            )
            span_name = f"{self.name}.{f_info.get_tag()}"
            if inspect.iscoroutinefunction(f):
                @wraps(f)
                async def wrapped_f(*args, **kwargs):
                    ctx = extract_ctx(kwargs)
                    with tracer.start_as_current_span(
                        span_name,
                        kind=SpanKind.SERVER,
                        context=ctx,
                    ) as span:
                        return await f(*args, **kwargs)
            else:
                @wraps(f)
                def wrapped_f(*args, **kwargs):
                    ctx = extract_ctx(kwargs)
                    with tracer.start_as_current_span(
                        span_name,
                        kind=SpanKind.SERVER,
                        context=ctx,
                    ) as span:
                        return f(*args, **kwargs)
            return in_wrapped(wrapped_f, _cls)
        return wr_wrapped

class InstrumentedInvocation(_Invocation):
    @staticmethod
    async def create(function_id: str, args, kwargs, client) -> _Invocation:
        carrier = {}
        inject(carrier)
        kwargs["_otel_context"] = carrier
        return await _Invocation.create(function_id, args, kwargs, client)

def extract_ctx(kwargs):
    if "_otel_context" in kwargs:
        return extract(kwargs.pop("_otel_context"))
    return None

__all__ = ["ModalInstrumentor", "OTEL_DEPS"]
