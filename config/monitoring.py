from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Histogram, generate_latest
import time

# Métriques personnalisées
REQUESTS_PROCESSING_TIME = Histogram(
    "fastapi_requests_duration_seconds",
    "Temps de traitement des requêtes HTTP",
    ["method", "endpoint"]
)

DB_OPERATIONS_PROCESSING_TIME = Histogram(
    "fastapi_db_operations_duration_seconds",
    "Temps de traitement des opérations de base de données",
    ["operation_type"]
)

FAILED_OPERATIONS_COUNTER = Counter(
    "fastapi_failed_operations_total",
    "Nombre total d'opérations échouées",
    ["operation_type"]
)

USER_OPERATIONS_COUNTER = Counter(
    "fastapi_user_operations_total",
    "Nombre total d'opérations sur les utilisateurs",
    ["operation_type"]
)

def init_monitoring(app):
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="fastapi_inprogress",
        inprogress_labels=True,
    )

    # Ajout des métriques par défaut
    instrumentator.add(
        metrics.latency(
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
            metric_namespace="fastapi",
            metric_subsystem="http",
        )
    )
    instrumentator.add(
        metrics.request_size(
            metric_namespace="fastapi",
            metric_subsystem="http",
        )
    )
    instrumentator.add(
        metrics.response_size(
            metric_namespace="fastapi",
            metric_subsystem="http",
        )
    )

    # Ajouter un endpoint metrics explicite
    @app.get("/metrics")
    async def metrics():
        return generate_latest()

    return instrumentator.instrument(app) 