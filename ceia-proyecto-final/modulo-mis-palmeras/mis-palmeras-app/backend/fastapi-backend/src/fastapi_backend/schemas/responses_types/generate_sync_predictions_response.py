from pydantic import BaseModel

# Local imports
from ..data_types.predictions_status_type import PredictionsStatusType
class GenerateSyncPredictionsResponse(BaseModel):
    id: int
    status: PredictionsStatusType
    message: str
