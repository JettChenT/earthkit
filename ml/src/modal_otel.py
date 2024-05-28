from typing import Collection, Dict, Optional
from functools import wraps
import inspect
import modal
from modal.app import FunctionInfo
import modal.client
import modal_proto.api_pb2
from opentelemetry.propagate import inject, extract
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import SpanKind, TracerProvider, get_tracer
from opentelemetry.propagate import inject
from modal import App
from modal.functions import _create_input, _Function, _Invocation
from modal.partial_function import _PartialFunction, _PartialFunctionFlags, _method, synchronize_api
import logging


# TODO: Invoke for remote calls instead

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
        self.original_function = modal.functions._Function
        self.original_create_input = modal.functions._create_input
        self.original_method = modal.partial_function._method
        _InstrumentedApp._tracer_provider = kwargs.get("tracer_provider")
        modal.App = _InstrumentedApp
        modal.Stub = _InstrumentedApp
        modal.functions._Function = _InstrumentedFunction
        modal.Function = modal.functions.Function = synchronize_api(_InstrumentedFunction)
        modal.functions._create_input = _create_input_otel
        modal.partial_function._method = patched_method
        modal.method = modal.partial_function.method = synchronize_api(patched_method)
        return super()._instrument(**kwargs)
    
    def _uninstrument(self, **kwargs):
        modal.App = self.original_app_cls
        modal.Stub = self.original_stub_cls
        modal.functions._create_input = self.original_create_input
        modal.partial_function._method = self.original_method
        modal.functions._Function = self.original_function
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
            wrapped_f = wrap_function_span(f, tracer, span_name)
            return in_wrapped(wrapped_f, _cls)
        return wr_wrapped

def patched_method(*args, **kwargs):
    in_wrapper = _method(*args, **kwargs)
    @wraps(in_wrapper)
    def wr_wrapped(f):
        tracer = get_tracer(
            __name__,
            tracer_provider=_InstrumentedApp._tracer_provider,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        span_name = f.__name__
        wrapped_f = wrap_function_span(f, tracer, span_name, is_method=True)
        print(f"wrapped {wrapped_f.__name__}")
        return in_wrapper(wrapped_f)
    return wr_wrapped

def wrap_function_span(f, tracer, span_name, is_method=False):
    def get_span_name(args):
        if not is_method:
            return span_name
        return f"{args[0].__class__.__name__}.{span_name}"
    if inspect.isasyncgenfunction(f):
        @wraps(f)
        async def wrapped_f(*args, **kwargs):
            ctx = extract_ctx(kwargs)
            with tracer.start_as_current_span(
                get_span_name(args),
                kind=SpanKind.SERVER,
                context=ctx,
            ) as span:
                async for result in f(*args, **kwargs):
                    yield result
    elif inspect.iscoroutinefunction(f):
        @wraps(f)
        async def wrapped_f(*args, **kwargs):
            ctx = extract_ctx(kwargs)
            with tracer.start_as_current_span(
                get_span_name(args),
                kind=SpanKind.SERVER,
                context=ctx,
            ) as span:
                return await f(*args, **kwargs)
    elif inspect.isgeneratorfunction(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            ctx = extract_ctx(kwargs)
            with tracer.start_as_current_span(
                get_span_name(args),
                kind=SpanKind.SERVER,
                context=ctx,
            ) as span:
                for result in f(*args, **kwargs):
                    yield result
    elif inspect.isfunction(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            ctx = extract_ctx(kwargs)
            with tracer.start_as_current_span(
                get_span_name(args),
                kind=SpanKind.SERVER,
                context=ctx,
            ) as span:
                return f(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported function type: {type(f)}")
    return wrapped_f

def g_tracer():
    return get_tracer(
        __name__,
        tracer_provider=_InstrumentedApp._tracer_provider,
        schema_url="https://opentelemetry.io/schemas/1.11.0",
    )

class _InstrumentedFunction(_Function):
    async def _call_function(self, args, kwargs):
        tracer = g_tracer()
        with tracer.start_as_current_span(
            f"invocation_function.{self.info.module_name}.{self.info.function_name}",
            kind=SpanKind.CLIENT,
        ) as span:
            return await super()._call_function(args, kwargs)


@wraps(_create_input)
async def _create_input_otel(args, kwargs, client, idx: Optional[int] = None):
    carrier = {}
    inject(carrier)
    kwargs["_otel_context"] = carrier
    return await _create_input(args, kwargs, client, idx)

def extract_ctx(kwargs):
    if "_otel_context" in kwargs:
        return extract(kwargs.pop("_otel_context"))
    return None

__all__ = ["ModalInstrumentor", "OTEL_DEPS"]
