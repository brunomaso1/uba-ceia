import io
import uuid
from fastapi import APIRouter, Depends, UploadFile, Response, HTTPException, status
import cv2
from fastapi.responses import StreamingResponse
import numpy as np
from loguru import logger

from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore, get_store_api
from fastapi_backend.errors.errors_codes import ERROR_CODES
from fastapi_backend.schemas.data_types.store_data_type import StoreDataType
from fastapi_backend.schemas.responses_types.upload_image_response import UploadImageResponse


router = APIRouter(prefix="/image", tags=["image"])


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Imagen subida correctamente."},
        400: {"description": "Solicitud incorrecta."},
        500: {"description": "Error interno del servidor."},
    },
    response_model=UploadImageResponse,
)
async def upload_image(file: UploadFile, store_api: InMemoryStore = Depends(get_store_api)) -> UploadImageResponse:
    if not file:
        raise HTTPException(status_code=400, detail="No se subió ningún archivo.")
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Use PNG o JPG.")
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado.")
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Error al decodificar la imagen.")

        image_storage_entry = StoreDataType(
            image=image,
            name=file.filename,
            uuid=str(uuid.uuid4()),
        )
        store_id = store_api.add(image_storage_entry)  # Add to the in-memory store
        logger.info(f"Imagen subida: {file.filename}, ID: {store_id}")
        return UploadImageResponse(id=image_storage_entry.id, name=image_storage_entry.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")


@router.get(
    "/download/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Imagen descargada correctamente."},
        404: {"description": "Imagen no encontrada."},
        500: {"description": "Error al codificar la imagen."},
    },
)
async def download_image(image_id: int, store_api: InMemoryStore = Depends(get_store_api)) -> StreamingResponse:
    """Download an image by its ID."""
    if image_id < 1 or image_id > store_api.get_length():
        raise HTTPException(status_code=404, detail="ID de imagen fuera del rango.")

    store_data_type = store_api.get(image_id)
    if store_data_type is None:
        raise HTTPException(status_code=404, detail="No se encontró la imagen con el ID proporcionado.")

    image = store_data_type.image
    if image is None:
        raise HTTPException(status_code=500, detail=ERROR_CODES[3])

    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise HTTPException(status_code=500, detail=ERROR_CODES[4])

    encoded_image_bytes = encoded_image.tobytes()

    def iter_bytes():
        chunk_size = 64 * 1024  # 64 KB
        for i in range(0, len(encoded_image_bytes), chunk_size):
            yield encoded_image_bytes[i : i + chunk_size]

    return StreamingResponse(
        iter_bytes(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=image_{image_id}.jpg"},
    )


@router.get(
    "/downloadAnnotated/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Imagen anotada descargada correctamente."},
        404: {"description": "Imagen no encontrada."},
        500: {"description": "Error al codificar la imagen."},
    },
)
async def download_annotated_image(
    image_id: int, store_api: InMemoryStore = Depends(get_store_api)
) -> StreamingResponse:
    """Download an image by its ID."""
    if image_id < 1 or image_id > store_api.get_length():
        raise HTTPException(status_code=404, detail="ID de imagen fuera del rango.")

    store_data_type = store_api.get(image_id)
    if store_data_type is None:
        raise HTTPException(status_code=404, detail="No se encontró la imagen con el ID proporcionado.")
    if store_data_type.predictions is None:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron predicciones para la imagen con el ID proporcionado. Ya las generó?",
        )

    image_buffer = store_data_type.encoded_annotated_image_buffer
    if image_buffer is None:
        raise HTTPException(status_code=500, detail=ERROR_CODES[5])

    return StreamingResponse(
        image_buffer,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=annotated_image_{image_id}.jpg"},
    )
