# Dependencias propias
from modulo_ia.config import settings as CONFIG

# Dependencias de terceros
from mlflow.tracking import MlflowClient
from loguru import logger as LOGGER

# Configuraciones.
TRACKING_URI = CONFIG.mlflow.tracking_uri
RUN_ID = "b7218a073a3942339689b2e7f4e0b543"  # ID a interactuar.


def finish_run():
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Forzamos el estado a FINISHED
    client.set_terminated(run_id=RUN_ID, status="FINISHED")
    LOGGER.success(f"Run {RUN_ID} actualizado a FINISHED")