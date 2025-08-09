from pydantic import BaseModel


class JGWDataType(BaseModel):
    """Data type for JGW files."""

    x_pixel_size: float
    x_rotation: float
    y_rotation: float
    y_pixel_size: float
    x_origin: float
    y_origin: float
