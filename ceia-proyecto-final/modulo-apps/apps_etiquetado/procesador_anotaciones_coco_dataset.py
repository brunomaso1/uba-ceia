import json
from pathlib import Path
from typing import Any, Dict, Optional


from apps_utils.logging import Logging

LOGGER = Logging().logger


def load_annotations_from_file(file_path: Path) -> Dict[str, Any]:
    """
    Carga anotaciones desde un archivo JSON.

    Args:
        file_path (Path): Ruta al archivo JSON.

    Returns:
        Dict[str, Any]: Anotaciones cargadas desde el archivo JSON.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        json.JSONDecodeError: Si el archivo no es un JSON válido.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        return annotations
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error al decodificar JSON en {file_path}: {e.msg}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"Error inesperado al cargar el archivo {file_path}: {e}")


def get_image_id_from_annotations(image_name: str, coco_annotations: Dict[str, Any]) -> Optional[int]:
    """
    Obtiene el ID de una imagen a partir de las anotaciones en formato COCO.

    Busca el ID de la imagen en las anotaciones COCO utilizando el nombre de la imagen.
    Si el nombre de la imagen no tiene extensión, se le añade ".jpg" para la búsqueda,
    dado que el formato COCO suele incluir la extensión en el campo "file_name".
    Si no se encuentra el ID, se lanza una excepción.

    Args:
        image_name (str): Nombre de la imagen (con o sin extensión .jpg).
        coco_annotations (Dict[str, Any]): Diccionario con las anotaciones en formato COCO.

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


def parse_class_annotations_to(coco_annotations: Dict[str, Any], class_name: str) -> Dict[str, Any]:
    """Convierte todas las anotaciones de una clase de un archivo de anotaciones COCO a otra clase especificada."""
    if not coco_annotations["annotations"]:
        LOGGER.warning("No hay anotaciones en el archivo COCO.")
        return coco_annotations

    category = {
        "id": 1,
        "name": class_name,
        "supercategory": "",
    }

    coco_annotations["categories"] = [category]
    for annotation in coco_annotations["annotations"]:
        annotation["category_id"] = 1

    return coco_annotations


def merge_coco_annotations(
    coco_annotations: Dict[str, Any],
    new_annotations: Dict[str, Any],
) -> Dict[str, Any]:
    """Fusiona dos conjuntos de anotaciones COCO."""
    raise NotImplementedError(
        "Esta función no está implementada. Se debe implementar la lógica de fusión de anotaciones COCO."
    )
    # if not coco_annotations["annotations"]:
    #     LOGGER.warning("No hay anotaciones en el archivo COCO.")
    #     return coco_annotations

    # # Merge categories
    # existing_category_ids = {cat["id"] for cat in coco_annotations["categories"]}
    # new_category_ids = {cat["id"] for cat in new_annotations["categories"]}

    # for category in new_annotations["categories"]:
    #     if category["id"] not in existing_category_ids:
    #         coco_annotations["categories"].append(category)

    # # Merge annotations
    # for annotation in new_annotations["annotations"]:
    #     if annotation["category_id"] not in existing_category_ids:
    #         annotation["category_id"] = 1  # Asignar a la primera categoría si no existe
    #     coco_annotations["annotations"].append(annotation)

    # return coco_annotations
