from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore, get_store_api
from fastapi_backend.services.zip_service import ZipService


router = APIRouter(prefix="/zip", tags=["zip"])


@router.get(
    "/download/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Zip descargada correctamente."},
        404: {"description": "Predicción no ejecutada."},
        500: {"description": "Error al codificar la Zip."},
    },
)
async def download_zip(image_id: int, store_api: InMemoryStore = Depends(get_store_api)) -> StreamingResponse:
    if image_id < 1 or image_id > store_api.get_length():
        raise HTTPException(status_code=404, detail="ID de imagen fuera del rango.")

    store_entry = store_api.get(image_id)
    if store_entry is None:
        raise HTTPException(
            status_code=404,
            detail="No se encontró la imagen con el ID proporcionado.",
        )
    if store_entry.predictions is None:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron predicciones para la imagen con el ID proporcionado. Ya las generó?",
        )

    zipService = ZipService(store_api)
    return zipService.download_zip(image_id)
