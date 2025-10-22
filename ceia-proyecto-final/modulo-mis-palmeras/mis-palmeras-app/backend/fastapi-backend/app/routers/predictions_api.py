# Dependencias del sistema
from typing import Annotated

# Dependencias locales
from ..dependencies.in_memory_store_api import InMemoryStore, get_store_api
from ..dependencies.prediction_service_api import get_prediction_service
from ..schemas.data_types.models_types import ModelType
from ..schemas.data_types.predictions_status_type import PredictionsStatusType
from ..schemas.responses_types.generate_sync_predictions_response import GenerateSyncPredictionsResponse
from ..services.predict_service import PredictionService

# Dependencias de terceros
from fastapi import APIRouter, status, HTTPException
from fastapi.params import Depends
from fastapi.responses import JSONResponse

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
    image_id: Annotated[
        int,
        "Identificador de la imagen para la cual se desean generar las predicciones. "
        "Este ID debe corresponder a una imagen previamente almacenada en el sistema, usualmente devuelta por el endpoint de subida de imágenes.",
    ],
    model_type: Annotated[ModelType, "Tipo de modelo a utilizar para las predicciones."] = ModelType.RPW_DETECTION,
    store_api: InMemoryStore = Depends(get_store_api),
    prediction_service: PredictionService = Depends(get_prediction_service),
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

    result = prediction_service.generate_sync_predictions(image_id, model_type)
    if result is None:
        raise HTTPException(status_code=500, detail="Error al generar las predicciones.")

    return GenerateSyncPredictionsResponse(
        id=image_id, status=PredictionsStatusType.COMPLETED, message="Predicciones generadas correctamente."
    )


@router.post(
    "/generate_async_prediction/{image_id}",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Predicción en proceso."},
        404: {"description": "Imagen no encontrada."},
        500: {"description": "Error interno del servidor."},
    },
)
async def generate_async_predictions(
    image_id: Annotated[
        int,
        "Identificador de la imagen para la cual se desean generar las predicciones. "
        "Este ID debe corresponder a una imagen previamente almacenada en el sistema, usualmente devuelta por el endpoint de subida de imágenes.",
    ],
    model_type: Annotated[ModelType, "Tipo de modelo a utilizar para las predicciones."] = ModelType.PALM_DETECTION,
    store_api: InMemoryStore = Depends(get_store_api),
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> JSONResponse:
    """Get predictions for a specific image by ID."""
    raise HTTPException(status_code=501, detail="Predicciones asíncronas no implementadas aún.")

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

    job_id = await prediction_service.generate_async_predictions(image_id, model_type)
    return JSONResponse(
        content={"message": f"Job {job_id} started for image ID {image_id}.", "status": "started", "job_id": job_id}
    )
