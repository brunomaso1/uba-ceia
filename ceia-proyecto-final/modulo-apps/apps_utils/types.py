from typing import Annotated, Any, Dict, List, Tuple, TypedDict


class Patch(TypedDict):
    patch_name: str
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    width: int
    height: int


class Metadata(TypedDict):
    pic_name: str
    image_shape: Annotated[Tuple[int, int], "Forma de la imagen (alto, ancho)"]
    patches: List[Patch]
