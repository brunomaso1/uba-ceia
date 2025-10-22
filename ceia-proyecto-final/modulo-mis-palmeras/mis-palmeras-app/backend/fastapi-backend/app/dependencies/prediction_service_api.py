# Dependencias locales
from .in_memory_store_api import InMemoryStore, get_store_api
from ..services.predict_service import PredictionService

# Dependencias de terceros
from fastapi import Depends


def get_prediction_service(
    store_api: InMemoryStore = Depends(get_store_api),
) -> PredictionService:
    """
    Crea una nueva instancia de PredictionService por request.
    """
    return PredictionService(store_api)
