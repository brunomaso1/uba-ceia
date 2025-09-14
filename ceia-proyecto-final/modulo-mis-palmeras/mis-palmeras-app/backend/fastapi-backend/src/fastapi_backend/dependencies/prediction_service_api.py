from fastapi import Depends
from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore, get_store_api
from fastapi_backend.services.predict_service import PredictionService

def get_prediction_service(
    store_api: InMemoryStore = Depends(get_store_api),
) -> PredictionService:
    """
    Crea una nueva instancia de PredictionService por request.
    """
    return PredictionService(store_api)