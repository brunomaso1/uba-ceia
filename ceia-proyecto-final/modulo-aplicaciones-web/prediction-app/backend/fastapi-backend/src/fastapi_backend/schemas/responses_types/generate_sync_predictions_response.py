from pydantic import BaseModel

from fastapi_backend.schemas.data_types.predictions_status_type import PredictionsStatusType


class GenerateSyncPredictionsResponse(BaseModel):
    id: int
    status: PredictionsStatusType
    message: str
