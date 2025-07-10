import datetime
import numpy as np
import os, sys, yaml, shutil

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from loguru import logger as LOGGER
from modulo_ia.config import config as CONFIG

from ultralytics.engine.results import Results
from supervision import Detections

from deprecated import deprecated


def filter_results_by_confidence(
    results: List[Dict[str, Any]],
    min_confidence: float = 0.5,
) -> Results:
    """
    Filtra los resultados de detección de objetos para eliminar aquellas detecciones
    con una confianza inferior al umbral especificado.

    Args:
        results (List[Dict[str, Any]]): Lista de resultados de detección, donde cada resultado
                                        es un diccionario que contiene información sobre las
                                        cajas delimitadoras, categorías y confianza.
        min_confidence (float): Umbral mínimo de confianza para filtrar las detecciones.
                                Por defecto es 0.5.

    Returns:
        Results: Resultados filtrados que cumplen con el umbral de confianza.
    """
    if len(results) > 1:
        raise ValueError(
            "Se detectó un resultado para varias imágenes. Asegúrate de que solo haya el resultado de una imagen."
        )
    result = results[0]

    # Filtrar los índices según el umbral
    filtered_indices = [i for i, conf in enumerate(result.boxes.conf) if conf.item() >= min_confidence]

    # Filtrar las cajas y otros atributos
    filtered_boxes = result.boxes[filtered_indices]

    # Crear un nuevo objeto Results con las mismas propiedades pero solo con las detecciones filtradas
    results_filtered = Results(
        orig_img=result.orig_img,
        path=result.path,
        names=result.names,
    )
    results_filtered.boxes = filtered_boxes
    results_filtered.orig_shape = result.orig_shape
    results_filtered.speed = result.speed
    results_filtered.save_dir = result.save_dir

    return [results_filtered]


@deprecated(
    version="1.0.0",
    reason="Esta función está obsoleta y será eliminada en futuras versiones. Usa create_coco_annotations_from_detections en su lugar.",
)
def create_coco_annotations_from_yolo_result(
    results: List[Dict[str, Any]],
    pic_name: str,
) -> Dict[str, Any]:
    """
    Convierte los resultados de detección de objetos en formato YOLO a anotaciones en formato COCO.

    Esta función toma una imagen y sus resultados de detección, y genera un diccionario
    que sigue la estructura del formato COCO. Se espera que los resultados contengan
    información sobre las cajas delimitadoras, categorías y confianza de las detecciones.
    Solamente se procesa una imagen a la vez.

    Args:
        image (np.ndarray): Imagen en formato numpy array sobre la que se realizó la detección.
        results (Dict[str, Any]): Resultados de la detección de objetos, típicamente una lista de predicciones YOLO.
        pic_name (str): Nombre base del archivo de la imagen (sin extensión).

    Raises:
        ValueError: Si los resultados contienen detecciones para más de una imagen.

    Returns:
        Dict[str, Any]: Diccionario con las anotaciones en formato COCO, incluyendo info, licenses, categories, images y annotations.
    """
    bboxes_result = results.boxes.cpu()
    names = results.names
    coco_annotations = {
        "info": CONFIG.coco_dataset.to_dict()["info"],
        "licenses": CONFIG.coco_dataset.to_dict()["licenses"],
        "categories": CONFIG.coco_dataset.to_dict()["categories"],
        "images": [],
        "annotations": [],
    }

    category_map = {cat["name"]: cat["id"] for cat in coco_annotations["categories"]}
    image_height, image_width = results.orig_shape

    image = {
        "id": 1,
        "width": image_width,
        "height": image_height,
        "file_name": f"{pic_name}.jpg",
        "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    coco_annotations["images"] = [image]

    if not results.boxes:
        LOGGER.warning("No se encontraron resultados de detección de objetos.")
        return coco_annotations

    annotations = []
    for index, bbox_result in enumerate(bboxes_result):
        id = index + 1
        category_name = names[bbox_result.cls.int().item()]
        category_id = category_map.get(category_name, None)
        if category_id is None:
            LOGGER.warning(f"Categoría '{category_name}' no encontrada en el mapa de categorías.")
            continue

        x_min, y_min, x_max, y_max = bbox_result.xyxy[0].numpy()
        ancho = x_max - x_min
        alto = y_max - y_min
        area = ancho * alto

        conf = bbox_result.conf[0].numpy()

        # Casteamos a float para evitar problemas de serialización
        conf = float(conf)
        x_min, y_min, ancho, alto = map(float, [x_min, y_min, ancho, alto])
        area = float(area)

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

    return coco_annotations
