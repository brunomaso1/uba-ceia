import json
from fastapi import APIRouter, Depends, Response, status, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore, get_store_api
from fastapi_backend.errors.errors_codes import ERROR_CODES

router = APIRouter(prefix="/geolocalization", tags=["geolocalization"])


@router.get(
    "/downloadKML/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "KML descargado correctamente."},
        404: {"description": "Imagen no encontrada."},
        500: {"description": "Error al generar el KML."},
    },
    response_class=Response,
)
async def download_kml(image_id: int, store_api: InMemoryStore = Depends(get_store_api)) -> Response:
    """Download a KML file for an image by its ID."""
    if image_id < 1 or image_id > store_api.get_length():
        raise HTTPException(status_code=404, detail="ID de imagen fuera del rango.")

    store_entry = store_api.get(image_id)
    if store_entry is None:
        raise HTTPException(
            status_code=404,
            detail="No se encontró la imagen con el ID proporcionado. Se ha ejecutado la predicción para dicho identificador?",
        )

    if store_entry.predictions is None:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron predicciones para la imagen con el ID proporcionado. Ya las generó?",
        )

    kml = store_entry.kml
    if kml is None:
        logger.error(f"Error al generar el KML para la imagen ID: {image_id}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_CODES[1],
        )

    return Response(
        content=kml,
        media_type="application/vnd.google-earth.kml+xml",
        headers={"Content-Disposition": f"attachment; filename=predictions_{image_id}.kml"},
    )


@router.get(
    "/downloadGeoJSON/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "GeoJSON descargado correctamente."},
        404: {"description": "Imagen no encontrada."},
        500: {"description": "Error al generar el GeoJSON."},
    },
)
async def download_geojson(image_id: int, store_api: InMemoryStore = Depends(get_store_api)) -> Response:
    """Download a GeoJSON file for an image by its ID."""
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

    geojson = store_entry.predictions.__geo_interface__
    if geojson is None:
        logger.error(f"Error al generar el GeoJSON para la imagen ID: {image_id}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_CODES[2],
        )
    
    # Serializar el GeoJSON a una cadena JSON
    geojson_String = json.dumps(geojson)

    return Response(
        content=geojson_String,
        media_type="application/geo+json",
        headers={"Content-Disposition": f"attachment; filename=predictions_{image_id}.geojson"},
    )
