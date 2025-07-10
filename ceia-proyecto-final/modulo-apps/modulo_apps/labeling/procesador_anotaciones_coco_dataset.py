import datetime, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as LOGGER
from modulo_apps.config import config as CONFIG
from modulo_apps.utils.types import Metadata

from supervision import Detections

import modulo_apps.labeling.convertor_cordenadas as ConvertorCordenadas

DOWNLOAD_COCO_ANNOTATIONS_FOLDER = CONFIG.folders.download_coco_annotations_folder


def load_annotations_from_path(file_path: Path) -> Dict[str, Any]:
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
    """
    Modifica las anotaciones de un archivo COCO para asignar todas las anotaciones a una única categoría.
    Args:
        coco_annotations (Dict[str, Any]): Diccionario que representa las anotaciones en formato COCO.
        class_name (str): Nombre de la clase que se asignará a todas las anotaciones.
    Returns:
        Dict[str, Any]: Diccionario modificado con las anotaciones actualizadas y una única categoría.
    Advertencias:
        - Si el archivo COCO no contiene anotaciones, se registra una advertencia y se devuelve el diccionario sin modificaciones.
    """

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


def merge_patches_coco_annotations_for_image(
    metadata: Metadata,
    coco_annotations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fusiona múltiples anotaciones COCO en una sola anotación, adaptando los cuadros delimitadores (bboxes)
    de los parches a las coordenadas de la imagen completa utilizando los metadatos proporcionados.

    Args:
        metadata (Metadata): Metadatos de la imagen objetivo, incluyendo el tamaño y la información de los parches.
        coco_annotations (List[Dict[str, Any]]): Lista de diccionarios de anotaciones COCO, cada uno correspondiente a un parche.

    Returns:
        Dict[str, Any]: Diccionario de anotaciones COCO fusionadas y adaptadas a la imagen completa.

    Ejemplo de uso:

        >>> patch_name1 = "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_0"
        >>> coco_annotations1_path = ProcesadorMongoDB.download_annotations_as_coco_from_mongodb(
        ...     field_name="cvat",
        ...     patches_names=[patch_name1],
        ...     output_filename=DOWNLOAD_COCO_ANNOTATIONS_FOLDER
        ...     / f"{patch_name1}.json",
        ... )
        >>> patch_name2 = "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_2"
        >>> coco_annotations2_path = ProcesadorMongoDB.download_annotations_as_coco_from_mongodb(
        ...     field_name="cvat",
        ...     patches_names=[patch_name2],
        ...     output_filename=DOWNLOAD_COCO_ANNOTATIONS_FOLDER
        ...     / f"{patch_name2}.json",
        ... )
        >>> coco_annotations1 = ProcesadorCOCODataset.load_annotations_from_file(coco_annotations1_path)
        >>> coco_annotations2 = ProcesadorCOCODataset.load_annotations_from_file(coco_annotations2_path)
        >>> patch_metadata1 = ProcesadorMongoDB.get_patch_metadata_from_mongodb(patch_name1)
        >>> patch_metadata2 = ProcesadorMongoDB.get_patch_metadata_from_mongodb(patch_name2)
        >>> image_name = "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm"
        >>> image_shape = (5426, 4356)  # Altura, Ancho
        >>> metadata: Metadata = {
        ...     "pic_name": image_name,
        ...     "image_shape": image_shape,
        ...     "patches": [patch_metadata1, patch_metadata2],
        ... }
        >>> merged_annotations = ProcesadorCOCODataset.merge_coco_annotations(metadata, [coco_annotations1, coco_annotations2])
        >>> print("Merged Annotations:")
        >>> pprint(merged_annotations)
    """

    merged_annotations = {
        "info": CONFIG.coco_dataset.info,
        "licenses": CONFIG.coco_dataset.licenses,
        "categories": CONFIG.coco_dataset.categories,
        "images": [],
        "annotations": [],
    }

    image_height, image_width = metadata["image_shape"]
    pic_name = metadata["pic_name"]
    image = {
        "id": 1,
        "width": image_width,
        "height": image_height,
        "file_name": f"{pic_name}.jpg",
        "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1. Primeramente, modifico todos los bboxes de todas las anotaciones ajustandose al tamaño de la imagen objetivo, utilizando los metadatos.
    coco_annotations = _convert_patch_bboxes_to_image(metadata, coco_annotations)

    # 2. Me quedo con todas las anotaciones aplanadas.
    all_annotations = [
        {**annotation} for coco_annotation in coco_annotations for annotation in coco_annotation["annotations"]
    ]

    # 3. Simplemente modifico el id de la imagen, dado que el bbox ya está adaptado a las coordenadas de la imagen objetivo.
    annotations = [
        {"id": index + 1, "image_id": image["id"], **annotation} for index, annotation in enumerate(all_annotations)
    ]

    merged_annotations["images"] = [image]
    merged_annotations["annotations"] = annotations

    return merged_annotations


def _convert_patch_bboxes_to_image(
    metadata: Metadata,
    coco_annotations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convierte las coordenadas de los cuadros delimitadores de parches a las coordenadas de la imagen completa.

    Args:
        metadata (Metadata): Metadatos de la imagen.
        coco_annotations (List[Dict[str, Any]]): Lista de anotaciones COCO.

    Returns:
        List[Dict[str, Any]]: Lista de anotaciones COCO con cuadros delimitadores convertidos.
    """
    for coco_annotation in coco_annotations:
        for image in coco_annotation["images"]:
            patch_name = image["file_name"].split(".")[0]  # Nombre sin extensión
            x_start, y_start = next(
                (patch["x_start"], patch["y_start"])
                for patch in metadata["patches"]
                if patch["patch_name"] == patch_name
            )
            for annotation in coco_annotation["annotations"]:
                if annotation["image_id"] == image["id"]:
                    annotation["bbox"] = ConvertorCordenadas.convert_bbox_patch_to_image(
                        annotation["bbox"], x_start, y_start
                    )
    return coco_annotations


def create_coco_annotations_from_detections(
    detections: Detections,
    image_size_hw: Tuple[int, int],
    pic_name: str,
    category_map: Optional[Dict[int, str]] = None,
    should_download: bool = False,
    output_filename: Path = DOWNLOAD_COCO_ANNOTATIONS_FOLDER / "coco_annotations.json",
) -> Dict[str, Any]:
    """
    Genera anotaciones en formato COCO a partir de detecciones de objetos.

    Args:
        detections (Detections): Objeto que contiene las detecciones realizadas, incluyendo coordenadas, clases y confianza.
        image_size_hw (Tuple[int, int]): Tamaño de la imagen en formato (alto, ancho).
        pic_name (str): Nombre de la imagen (sin extensión) que será utilizada en las anotaciones.
        category_map (Optional[Dict[int, str]]): Mapa opcional que relaciona IDs de categorías con nombres de categorías.
            Si no se proporciona, se utiliza el mapa de categorías definido en la configuración del dataset COCO.
        should_download (bool): Indica si las anotaciones generadas deben ser guardadas en un archivo JSON.
        output_filename (Path): Ruta del archivo donde se guardarán las anotaciones en caso de que `should_download` sea True.

    Returns:
        Dict[str, Any]: Diccionario con las anotaciones en formato COCO, incluyendo información del dataset, licencias,
        categorías, imágenes y anotaciones.

    Notas:
        - Si no se encuentran detecciones en el objeto `detections`, se devuelve un diccionario de anotaciones vacío.
        - Las anotaciones incluyen información como el ID de la categoría, el cuadro delimitador (bbox), el área, y la confianza.
        - Si `should_download` es True, las anotaciones se guardan en el archivo especificado por `output_filename`.
        
    Advertencias:
        - Si una categoría no se encuentra en el mapa de categorías, se omite la anotación correspondiente y se registra
          una advertencia en el registro de eventos.
    """
    coco_annotations = {
        "info": CONFIG.coco_dataset.to_dict()["info"],
        "licenses": CONFIG.coco_dataset.to_dict()["licenses"],
        "categories": CONFIG.coco_dataset.to_dict()["categories"],
        "images": [],
        "annotations": [],
    }
    if category_map is None:
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

    if should_download:
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(coco_annotations, f, indent=4, ensure_ascii=False)
        LOGGER.success(f"Anotaciones COCO guardadas en {output_filename}")

    return coco_annotations
