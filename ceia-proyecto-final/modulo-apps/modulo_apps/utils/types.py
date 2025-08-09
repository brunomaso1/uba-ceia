from typing import Literal
from pydantic import BaseModel, Field


class Patch(BaseModel):
    patch_name: str
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    width: int
    height: int


class Metadata(BaseModel):
    pic_name: str
    image_shape: tuple[int, int] = Field(..., description="Forma de la imagen (alto, ancho)")
    patches: list[Patch]

AnnotationType = Literal["images", "patches", "cvat"]
