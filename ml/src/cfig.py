import os
import dotenv

if not os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL"):
    dotenv.load_dotenv()

assert os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL")

ENVS = {
   "OTEL_EXPORTER_OTLP_PROTOCOL": os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"],
   "OTEL_EXPORTER_OTLP_ENDPOINT": os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"],
   "OTEL_EXPORTER_OTLP_HEADERS": os.environ["OTEL_EXPORTER_OTLP_HEADERS"],
   "AXIOM_DATASET_NAME": os.environ["AXIOM_DATASET_NAME"],
   "AXIOM_API_KEY": os.environ["AXIOM_API_KEY"],
}