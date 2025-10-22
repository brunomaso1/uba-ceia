# Dependencias del sistema
import io

# Dependencias locales
from ..data_types.jgw_data_type import JGWDataType

# Dependencias de terceros
from geopandas import GeoDataFrame
import numpy as np
from pydantic import BaseModel


class StoreDataType(BaseModel):
    """Base class for data types that can be stored in the database."""

    id: int | None = None
    image: np.ndarray | None = None
    uuid: str | None = None
    name: str
    jgw: JGWDataType | None = None
    predictions: GeoDataFrame | None = None
    kml: str | None = None
    # annotated_image: np.ndarray | None = None
    encoded_annotated_image_buffer: io.BytesIO | None = None

    class Config:
        arbitrary_types_allowed = True
