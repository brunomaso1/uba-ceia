from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, computed_field


class Patch(BaseModel):
    patch_id: int
    patch_name: str
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    width: int
    height: int
    is_white: bool
    white_threeshold_percent: int
    white_threeshold_value: int


class JGWData(BaseModel):
    x_pixel_size: float
    y_rotation: float
    x_rotation: float
    y_pixel_size: float
    x_origin: float
    y_origin: float


class Metadata(BaseModel):
    pic_name: str
    image_shape: tuple[int, int] = Field(..., description="Forma de la imagen (alto, ancho)")
    patches: list[Patch]


class DownloadFileMetadata(BaseModel):
    file_download_id: str | None = None
    js_name: str | None = None
    title: str
    date_captured: datetime
    group_id: str | None = None
    height: int | None = None
    width: int | None = None
    downloaded_date: datetime | None = None
    jgw_data: JGWData | None = None
    patches: list[Patch] | None = None
    has_patches: bool = None
    url_generate_zip: str | None = None
    
    @computed_field
    @property
    def image_name(self) -> str | None:
        return self.file_download_id


AnnotationType = Literal["images", "patches", "cvat"]
