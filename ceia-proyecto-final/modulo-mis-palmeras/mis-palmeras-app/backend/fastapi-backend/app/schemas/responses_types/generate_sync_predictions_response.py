# Dependencias locales
from ..data_types.predictions_status_type import PredictionsStatusType

# Dependencias de terceros
from pydantic import BaseModel


class GenerateSyncPredictionsResponse(BaseModel):
    id: int
    status: PredictionsStatusType
    message: str
