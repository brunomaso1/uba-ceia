from pydantic import BaseModel

from fastapi_backend.enums.predictions_status_enum import PredictionsStatusEnum


class GenerateSyncPredictionsResponse(BaseModel):
    id: int
    status: PredictionsStatusEnum
    message: str
