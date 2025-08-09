from fastapi import APIRouter, Depends, UploadFile, status, HTTPException
from loguru import logger

from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore, get_store_api
from fastapi_backend.schemas.data_types.jgw_data_type import JGWDataType
from fastapi_backend.schemas.responses_types.upload_image_response import UploadImageResponse
from fastapi_backend.schemas.responses_types.upload_jgw_data_response import UploadJGWDataResponse

router = APIRouter(prefix="/jgw", tags=["jgw"])


@router.post(
    "/upload/{image_id}",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "JGW subidos correctamente."},
        400: {"description": "Solicitud incorrecta."},
        500: {"description": "Error interno del servidor."},
    },
    response_model=UploadJGWDataResponse,
)
async def upload_jgw(
    image_id: int, jgw_data: JGWDataType, store_api: InMemoryStore = Depends(get_store_api)
) -> UploadJGWDataResponse:
    """Upload a JGW JSON for a specific image by ID."""
    if image_id < 1 or image_id > store_api.get_length():
        raise HTTPException(status_code=404, detail="ID de imagen fuera del rango.")

    store_entry = store_api.get(image_id)
    if store_entry is None:
        raise HTTPException(status_code=404, detail="No se encontró la imagen con el ID proporcionado.")
    if not jgw_data:
        raise HTTPException(status_code=400, detail="No se proporcionó ningún dato JGW.")
    try:
        # Validate the JGW data
        if not all(
            isinstance(value, (float, int))
            for value in [
                jgw_data.x_pixel_size,
                jgw_data.x_rotation,
                jgw_data.y_rotation,
                jgw_data.y_pixel_size,
                jgw_data.x_origin,
                jgw_data.y_origin,
            ]
        ):
            raise HTTPException(status_code=400, detail="Los datos JGW deben ser valores numéricos válidos.")

        store_entry.jgw = jgw_data
        logger.info(f"JGW subido para la imagen ID: {image_id}")
        return UploadJGWDataResponse(id=image_id, jgw_file_name=f"jgw_{image_id}.json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar los datos JGW: {str(e)}")


@router.get(
    "/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "JGW obtenido correctamente."},
        404: {"description": "Imagen no encontrada."},
    },
    response_model=JGWDataType,
)
async def download_jgw(image_id: int, store_api: InMemoryStore = Depends(get_store_api)) -> JGWDataType:
    """Get JGW data for a specific image by ID."""
    if image_id < 1 or image_id > store_api.get_length():
        raise HTTPException(status_code=404, detail="ID de imagen fuera del rango.")

    store_entry = store_api.get(image_id)
    if store_entry is None:
        raise HTTPException(status_code=404, detail="No se encontró la imagen con el ID proporcionado.")

    if store_entry.jgw is None:
        raise HTTPException(
            status_code=404, detail="No se encontró el JGW para la imagen con el ID proporcionado. Ya lo subió?"
        )

    return store_entry.jgw
