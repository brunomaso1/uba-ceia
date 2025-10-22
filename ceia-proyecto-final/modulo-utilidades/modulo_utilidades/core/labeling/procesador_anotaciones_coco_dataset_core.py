# Dependencias del sistema.
import datetime, json
from pathlib import Path
from typing import Any, Optional

# Dependencias propias.
from modulo_utilidades.config import settings as CONFIG

# Dependencias de terceros.
from loguru import logger as LOGGER
from supervision import Detections

# Configuraciones
DOWNLOAD_COCO_ANNOTATIONS_FOLDER: Path = CONFIG.folders.download_coco_annotations_folder
COCO_DUMP = CONFIG.coco_dataset.model_dump()
COCO_DATASET_INFO_DICT: dict[str, Any] = COCO_DUMP["info"]
COCO_DATASET_LICENSES_DICT: list[dict[str, Any]] = COCO_DUMP["licenses"]
COCO_DATASET_CATEGORIES_DICT: list[dict[str, Any]] = COCO_DUMP["categories"]


def get_image_id_from_annotations(image_name: str, coco_annotations: dict[str, Any]) -> Optional[int]:
    """
    Obtiene el ID de una imagen a partir de las anotaciones en formato COCO.

    Busca el ID de la imagen en las anotaciones COCO utilizando el nombre de la imagen.
    Si el nombre de la imagen no tiene extensión, se le añade ".jpg" para la búsqueda,
    dado que el formato COCO suele incluir la extensión en el campo "file_name".
    Si no se encuentra el ID, se lanza una excepción.

    Args:
        image_name (str): Nombre de la imagen (con o sin extensión .jpg).
        coco_annotations (dict[str, Any]): Diccionario con las anotaciones en formato COCO.

    Raises:
        ValueError: Si no se encuentra el ID de la imagen en las anotaciones.

    Returns:
        Optional[int]: ID de la imagen si se encuentra, de lo contrario None.
    """
    if not coco_annotations["annotations"]:
        LOGGER.warning("No hay anotaciones en el archivo COCO.")
        return None

    image_id = next(
        (
            img["id"]
            for img in coco_annotations["images"]
            if img["file_name"] == image_name or img["file_name"] == f"{image_name}.jpg"
        ),
        None,
    )

    if not image_id:
        raise ValueError(f"No se encontró el id de la imagen {image_name} en las anotaciones.")
    return image_id


def create_coco_annotations_from_detections(
    detections: Detections,
    image_size_hw: tuple[int, int],
    pic_name: str,
    categories: Optional[list[dict]] = None,
    output_file_path: Optional[Path] = None,
) -> dict[str, Any]:
    if categories is None:
        categories = COCO_DATASET_CATEGORIES_DICT
        LOGGER.debug(
            "No se proporcionó un mapa de categorías. Se utilizará el mapa de categorías predeterminado del dataset COCO. Categorías: {categories}"
        )

    coco_annotations = {
        "info": COCO_DATASET_INFO_DICT,
        "licenses": COCO_DATASET_LICENSES_DICT,
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    category_map = {cat["id"]: cat["name"] for cat in coco_annotations["categories"]}
    image_height, image_width = image_size_hw

    image = {
        "id": 1,
        "width": image_width,
        "height": image_height,
        "file_name": f"{pic_name}.jpg",
        "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    coco_annotations["images"] = [image]

    if detections.is_empty():
        LOGGER.warning("No se encontraron resultados de detección de objetos.")
        return coco_annotations

    annotations = []
    for index in range(len(detections)):
        id = index + 1
        category_id = int(detections.class_id[index])
        category_name = category_map.get(category_id, None)
        if category_id is None:
            LOGGER.warning(f"Categoría '{category_name}' no encontrada en el mapa de categorías.")
            continue

        x_min, y_min, x_max, y_max = map(float, detections.xyxy[index])
        ancho = x_max - x_min
        alto = y_max - y_min
        area = ancho * alto

        conf = float(detections.confidence[index])
        annotation = {
            "id": id,
            "image_id": image["id"],
            "category_id": category_id,
            "bbox": [x_min, y_min, ancho, alto],
            "area": area,
            "iscrowd": 0,
            "attributes": {
                "occluded": False,
                "rotation": 0.0,
            },
            "confidence": conf,
        }
        annotations.append(annotation)

    coco_annotations["annotations"] = annotations

    if output_file_path:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(coco_annotations, f, indent=4, ensure_ascii=False)
        LOGGER.success(f"Anotaciones COCO guardadas en {output_file_path}")

    return coco_annotations
