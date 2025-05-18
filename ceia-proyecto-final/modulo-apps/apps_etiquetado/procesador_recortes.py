import copy, datetime, json, sys, os

sys.path.append(os.path.abspath("../"))

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

import cv2 as cv

from apps_config.settings import Config
from apps_utils.logging import Logging
from apps_com_db.mongodb_client import MongoDB
from apps_com_s3.minio_client import S3Client

import apps_etiquetado.utils_coco_dataset as CocoDatasetUtils
import apps_etiquetado.convertor_cordenadas as ConvertorCoordenadas

CONFIG = Config().config_data
LOGGER = Logging().logger
DB = MongoDB().db
MINIO_CLIENT = S3Client().client

MINIO_BUCKET = CONFIG["minio"]["bucket"]
download_folder = Path(CONFIG["folders"]["download_folder"])
DOWNLOAD_CUTOUTS_FOLDER = download_folder / "cutouts"
DOWNLOAD_CUTOUTS_METADATA_FOLDER = download_folder / "cutouts_metadata"


def cutout_bbox_from_image(image: np.ndarray, bbox: list) -> np.ndarray:
    """Recorta una región de interés (ROI) de una imagen utilizando un bounding box.

    Args:
        image (numpy.ndarray): Imagen de entrada.
        bbox (list): Bounding box en formato [x_min, y_min, width, height].

    Returns:
        numpy.ndarray: Imagen recortada.
    """
    x_min, y_min, width, height = bbox
    x_max = int(x_min + width)
    y_max = int(y_min + height)

    # Recortar la imagen utilizando el bounding box
    return image[int(y_min) : int(y_max), int(x_min) : int(x_max)]


def cut_palms_from_image(
    image: np.ndarray,
    coco_annotations: Dict[str, Any],
    pic_name: str,
    with_metadata: bool = True,
    cutout_folder: Optional[Path] = None,
    metadata_folder: Optional[Path] = None,
) -> Optional[Path]:
    """Recorta las palmas de una imagen utilizando las anotaciones proporcionadas en formato COCO.
    También guarda los metadatos de las imágenes recortadas en un archivo JSON.

    Este método carga una imagen y sus anotaciones asociadas en formato COCO, recorta las regiones de interés
    (ROIs) correspondientes a las palmas y guarda los recortes en una carpeta especificada.

    Args:
        image (numpy.ndarray): Imagen de entrada.
        coco_annotations (dict): Diccionario con las anotaciones en formato COCO.
        pic_name (str): Nombre de la imagen o parche para identificar los recortes.
        with_metadata (bool, optional): Si se deben guardar los metadatos de las imágenes recortadas. Defaults to True.
        cutout_folder (str, optional): Carpeta donde se guardarán los recortes. Si no se proporciona, se crea una
                                       carpeta en la ubicación predeterminada. Defaults to None.
        metadata_folder (str, optional): Carpeta donde se guardarán los metadatos de las imágenes recortadas. Si no
                                        se proporciona, se crea una carpeta en la ubicación predeterminada.

    Raises:
        FileNotFoundError: Si el archivo de imagen no existe.
        ValueError: Si no se encuentran anotaciones para la imagen o si no se puede cargar la imagen.

    Returns:
        Optional[Path]: Ruta de la carpeta donde se guardaron los recortes.

    Ejemplo de uso:

        >>> image_name = "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm"
        >>> patch_name = "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_0"
        >>> annotations_field = "cvat"
        >>> pic_name = patch_name
        >>> if pic_name == patch_name:
        >>>     image_path = download_patch_from_minio(patch_name)
        >>>     annotations_path = download_annotation_as_coco_from_mongodb(
        ...         field_name=annotations_field, patch_name=patch_name
        ...     )
        >>> else:
        >>>     image_path = download_image_from_minio(image_name)
        >>>     annotations_path = download_annotation_as_coco_from_mongodb(
        ...         field_name=annotations_field, image_name=image_name
        ...     )
        >>> image = cv.imread(image_path)
        >>> coco_annotations = load_annotations_from_file(annotations_path)
        >>> cutout_folder = cut_palms_from_image(
        ...     image=image,
        ...     coco_annotations=coco_annotations,
        ...     pic_name=pic_name,
        ... )
    """
    # Buscar el id de la imagen en las anotaciones dependiendo de si se proporciona el nombre del parche o de la imagen
    image_id = CocoDatasetUtils.get_image_id_from_annotations(pic_name, coco_annotations)

    # Filtrar las anotaciones para la imagen actual
    annotations: list = [ann for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
    if not annotations:
        LOGGER.warning(f"No se encontraron anotaciones para la imagen {pic_name}.")
        return None

    # Crear una carpeta para guardar los recortes
    if not cutout_folder:
        cutout_folder = DOWNLOAD_CUTOUTS_FOLDER / pic_name
        Path(cutout_folder).mkdir(parents=True, exist_ok=True)
        LOGGER.debug(f"Carpeta de recortes creada: {cutout_folder}")
    if not metadata_folder:
        metadata_folder = DOWNLOAD_CUTOUTS_METADATA_FOLDER / pic_name
        Path(metadata_folder).mkdir(parents=True, exist_ok=True)
        LOGGER.debug(f"Carpeta de metadatos creada: {metadata_folder}")

    # Guardar los metadatos de la imagen recortada
    cutout_coco_images_field: list = []
    cutout_coco_annotations_field: list = []

    # Recortar las regiones de interés (ROIs) y guardarlas
    for i, annotation in enumerate(annotations):
        bbox = annotation["bbox"]
        cutout_image = cutout_bbox_from_image(image, bbox)

        # Guardar la imagen recortada
        cutout_name = f"{pic_name}_cutout_{i + 1}"
        cutout_path = cutout_folder / f"{cutout_name}.jpg"
        cv.imwrite(cutout_path, cutout_image)
        LOGGER.debug(f"Recorte guardado: {cutout_path}")

        # Guardar los metadatos de la imagen recortada
        output_coco_image = {
            "id": i + 1,
            "width": cutout_image.shape[1],
            "height": cutout_image.shape[0],
            "file_name": cutout_name,
            "date_captured": datetime.date.today().strftime("%Y-%m-%d"),
        }

        output_coco_annotation = {
            "id": 1,
            "image_id": output_coco_image["id"],
            "category_id": annotation["category_id"],
        }

        cutout_coco_images_field.append(output_coco_image)
        cutout_coco_annotations_field.append(output_coco_annotation)

    # Guardar los metadatos de la imagen recortada en un archivo JSON
    cutout_coco_annotations = {
        "info": coco_annotations["info"],
        "licenses": coco_annotations["licenses"],
        "categories": coco_annotations["categories"],
        "images": cutout_coco_images_field,
        "annotations": cutout_coco_annotations_field,
    }

    if with_metadata:
        cutout_metadata_path = metadata_folder / f"{pic_name}_metadata.json"
        with open(cutout_metadata_path, "w") as f:
            json.dump(cutout_coco_annotations, f, indent=4)
            LOGGER.debug(f"Metadatos guardados en {cutout_metadata_path}.")

    return cutout_folder


def clear_minio_cutouts_folder(folder: str, is_metadata: bool = False) -> None:
    """
    Elimina el contenido de una carpeta específica en MinIO.

    Este método permite eliminar todos los objetos dentro de una carpeta en MinIO,
    ya sea para recortes o metadatos de recortes, según el prefijo proporcionado.

    Args:
        folder (str): Nombre de la carpeta cuyo contenido se desea eliminar.
        is_metadata (bool, optional): Indica si se trata de una carpeta de metadatos.
                                      Defaults to False.
    """
    prefix_key_path = (
        f"{CONFIG['minio']['cutouts_path']}/{folder}/"
        if not is_metadata
        else f"{CONFIG['minio']['cutouts_metadata_path']}/{folder}/"
    )
    objects_to_delete = MINIO_CLIENT.list_objects(Bucket=MINIO_BUCKET, Prefix=prefix_key_path)
    delete_keys = {"Objects": []}
    delete_keys["Objects"] = [{"Key": k} for k in [obj["Key"] for obj in objects_to_delete.get("Contents", [])]]
    if delete_keys["Objects"]:
        MINIO_CLIENT.delete_objects(Bucket=MINIO_BUCKET, Delete=delete_keys)
        LOGGER.debug(f"Se eliminaron {len(delete_keys['Objects'])} objetos de MinIO con el prefijo {prefix_key_path}.")
    else:
        LOGGER.debug(f"No se encontraron objetos para eliminar con el prefijo {prefix_key_path}.")


def upload_cutouts_to_mino(
    cutout_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
    cutout_metadata_folder: Optional[Path] = DOWNLOAD_CUTOUTS_METADATA_FOLDER,
) -> None:
    """
    Sube los recortes y sus metadatos a MinIO.

    Este método permite subir las imágenes recortadas y sus metadatos almacenados localmente
    a MinIO. Antes de subir los archivos, elimina el contenido existente en las carpetas
    correspondientes en MinIO.

    Args:
        cutout_folder (Path, optional): Carpeta local que contiene los recortes.
                                         Defaults to DOWNLOAD_CUTOUTS_FOLDER.
        cutout_metadata_folder (Optional[Path], optional): Carpeta local que contiene los metadatos
                                                           de los recortes. Defaults to DOWNLOAD_CUTOUTS_METADATA_FOLDER.

    Raises:
        FileNotFoundError: Si la carpeta de recortes no existe.
        FileNotFoundError: Si la carpeta de metadatos no existe (si se proporciona).
    """
    if not cutout_folder.exists():
        raise FileNotFoundError(f"La carpeta de recortes {cutout_folder} no existe.")
    if cutout_metadata_folder and not cutout_metadata_folder.exists():
        raise FileNotFoundError(f"La carpeta de metadatos {cutout_metadata_folder} no existe.")

    for folder in os.listdir(cutout_folder):
        folder_path = cutout_folder / folder

        # Eliminamos el contenido del folder en MinIO
        clear_minio_cutouts_folder(folder, is_metadata=False)

        # Subir las imágenes a MinIO
        for _, _, files in os.walk(folder_path):
            for filename in files:
                MINIO_CLIENT.upload_file(
                    Filename=os.path.join(folder_path, filename),
                    Bucket=MINIO_BUCKET,
                    Key=f"{CONFIG['minio']['cutouts_path']}/{folder}/{filename}",
                )
                LOGGER.debug(f"Imagen {filename} subida a MinIO.")

    if cutout_metadata_folder:
        for folder in os.listdir(cutout_metadata_folder):
            folder_path = cutout_metadata_folder / folder

            # Eliminamos el contenido del folder en MinIO
            clear_minio_cutouts_folder(folder, is_metadata=True)

            # Subimos los metadatos a MinIO
            for _, _, files in os.walk(folder_path):
                for filename in files:
                    # Subir el archivo JSON de metadatos a MinIO
                    MINIO_CLIENT.upload_file(
                        Filename=os.path.join(folder_path, filename),
                        Bucket=MINIO_BUCKET,
                        Key=f"{CONFIG['minio']['cutouts_metadata_path']}/{folder}/{filename}",
                    )
                    LOGGER.debug(f"Archivo JSON de metadatos {filename} subido a MinIO.")
