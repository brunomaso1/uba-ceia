from fastapi import APIRouter, status

from fastapi_backend.config import API_VERSION
from fastapi_backend.routers import geolocalization_api, image_api, jgw_api, predictions_api, zip_api
from fastapi_backend.schemas.responses_types.health_check_response import HealthCheckResponse

router = APIRouter(prefix=f"/api{API_VERSION}")

router.include_router(geolocalization_api.router)
router.include_router(image_api.router)
router.include_router(jgw_api.router)
router.include_router(predictions_api.router)
router.include_router(zip_api.router)


@router.get(
    "/health",
    tags=["healthcheck"],
    summary="Health Check",
    description="Check the health status of the API.",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheckResponse,
)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint to verify the API is running."""
    return HealthCheckResponse(status="Ok")
