from fastapi import APIRouter, status, HTTPException
from fastapi.params import Depends
from fastapi.responses import JSONResponse

from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore, get_store_api
from fastapi_backend.enums.predictions_status_enum import PredictionsStatusEnum
from fastapi_backend.schemas.responses_types.generate_sync_predictions_response import GenerateSyncPredictionsResponse
from fastapi_backend.services.predict_service import PredictionService


router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post(
    "/generate_sync_prediction/{image_id}",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Predicción en proceso."},
        404: {"description": "Imagen no encontrada."},
        500: {"description": "Error interno del servidor."},
    },
    response_model=GenerateSyncPredictionsResponse,
)
def generate_sync_predictions(
    image_id: int, store_api: InMemoryStore = Depends(get_store_api)
) -> GenerateSyncPredictionsResponse:
    """Get predictions for a specific image by ID."""
    if image_id < 1 or image_id > store_api.get_length():
        raise HTTPException(status_code=404, detail="ID de imagen fuera del rango.")

    store_entry = store_api.get(image_id)
    if store_entry is None:
        raise HTTPException(status_code=404, detail="No se encontró la imagen con el ID proporcionado.")

    if store_entry.image is None:
        raise HTTPException(status_code=404, detail="No se encontró la imagen para el ID proporcionado. Ya la subió?")
    if store_entry.jgw is None:
        raise HTTPException(
            status_code=404, detail="No se encontró el JGW para la imagen con el ID proporcionado. Ya lo subió?"
        )

    predictions = PredictionService(store_api)
    result = predictions.generate_sync_predictions(image_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Error al generar las predicciones.")

    return GenerateSyncPredictionsResponse(
        id=image_id, status=PredictionsStatusEnum.COMPLETED, message="Predicciones generadas correctamente."
    )


@router.post(
    "/generate_async_prediction/{image_id}",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Predicción en proceso."},
        404: {"description": "Imagen no encontrada."},
    },
)
async def generate_async_predictions(image_id: int, store_api: InMemoryStore = Depends(get_store_api)) -> JSONResponse:
    """Get predictions for a specific image by ID."""
    if image_id < 1 or image_id > store_api.get_length():
        raise HTTPException(status_code=404, detail="ID de imagen fuera del rango.")

    store_entry = store_api.get(image_id)
    if store_entry is None:
        raise HTTPException(status_code=404, detail="No se encontró la imagen con el ID proporcionado.")

    if store_entry.image is None:
        raise HTTPException(status_code=404, detail="No se encontró la imagen para el ID proporcionado. Ya la subió?")
    if store_entry.jgw is None:
        raise HTTPException(
            status_code=404, detail="No se encontró el JGW para la imagen con el ID proporcionado. Ya lo subió?"
        )

    # TODO: Make this inyectable.
    predictions = PredictionService(store_api, image_id)
    job_id = await predictions.generate_async_predictions()
    return JSONResponse(
        content={"message": f"Job {job_id} started for image ID {image_id}.", "status": "started", "job_id": job_id}
    )
