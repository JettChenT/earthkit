from typing import Collection, Dict, Optional
import modal
from modal.app import FunctionInfo
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import SpanKind, TracerProvider, get_tracer
from modal import App, Function
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
        _InstrumentedApp._tracer_provider = kwargs.get("tracer_provider")
        modal.App = _InstrumentedApp
        modal.Stub = _InstrumentedApp
        return super()._instrument(**kwargs)
    
    def _uninstrument(self, **kwargs):
        modal.App = self.original_app_cls
        modal.Stub = self.original_stub_cls
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
            @wraps(f)
            def wrapped_f(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name,
                    kind=SpanKind.SERVER,
                ) as span:
                    return f(*args, **kwargs)
            return in_wrapped(wrapped_f, _cls)
        return wr_wrapped

__all__ = ["ModalInstrumentor", "OTEL_DEPS"]
