import datetime, json, os

from pathlib import Path
from typing import Any, Optional
import numpy as np

from loguru import logger as LOGGER
from tqdm import tqdm
from modulo_apps.config import config as CONFIG
from modulo_apps.s3_comunication.s3_client import s3client as S3_CLIENT

import modulo_apps.labeling.procesador_anotaciones_coco_dataset as CocoDatasetUtils
import modulo_apps.labeling.procesador_anotaciones_mongodb as ProcesadorCocoDataset
import modulo_apps.s3_comunication.procesador_s3 as ProcesadorS3

import typer

import cv2 as cv

MINIO_BUCKET = CONFIG.minio.bucket
DOWNLOAD_CUTOUTS_FOLDER = CONFIG.folders.download_cutouts_folder
DOWNLOAD_CUTOUTS_METADATA_FOLDER = CONFIG.folders.download_cutouts_metadata_folder
MINIO_CUTOUTS_PATH = CONFIG.minio.paths.cutouts
MINIO_CUTOUTS_METADATA_PATH = CONFIG.minio.paths.cutouts_metadata
DOWNLOAD_IMAGES_FOLDER = CONFIG.folders.download_images_folder

app = typer.Typer()


def cut_bbox_from_image(image: np.ndarray, bbox: list) -> np.ndarray:
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
    coco_annotations: dict[str, Any],
    pic_name: str,
    with_metadata: bool = True,
    output_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
    metadata_folder: Path = DOWNLOAD_CUTOUTS_METADATA_FOLDER,
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
    if not image.any():
        raise ValueError(f"La imagen {pic_name} no se pudo cargar. Verifique la ruta y el formato.")

    # Buscar el id de la imagen en las anotaciones dependiendo de si se proporciona el nombre del parche o de la imagen
    image_id = CocoDatasetUtils.get_image_id_from_annotations(pic_name, coco_annotations)

    # Filtrar las anotaciones para la imagen actual
    annotations: list = [ann for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
    if not annotations:
        LOGGER.warning(f"No se encontraron anotaciones para la imagen {pic_name}.")
        return None

    # Crear una carpeta para guardar los recortes
    output_folder /= pic_name
    output_folder.mkdir(parents=True, exist_ok=True)
    metadata_folder /= pic_name
    metadata_folder.mkdir(parents=True, exist_ok=True)

    # Guardar los metadatos de la imagen recortada
    cutout_coco_images_field: list = []
    cutout_coco_annotations_field: list = []

    # Recortar las regiones de interés (ROIs) y guardarlas
    for i, annotation in enumerate(annotations):
        bbox = annotation["bbox"]
        cutout_image = cut_bbox_from_image(image, bbox)

        # Guardar la imagen recortada
        cutout_name = f"{pic_name}_cutout_{i + 1}"
        cutout_path = output_folder / f"{cutout_name}.jpg"
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

    return output_folder


@app.command()
def cut_palms_from_image_path(
    image_path: Path,
    coco_annotations_path: Path,
    with_metadata: bool = True,
    output_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
    metadata_folder: Path = DOWNLOAD_CUTOUTS_METADATA_FOLDER,
) -> Optional[Path]:
    """Recorta las palmas de una imagen utilizando las anotaciones proporcionadas en formato COCO.

    Args:
        input_image (Path): Ruta de la imagen de entrada.
        input_coco_annotations (Path): Ruta del archivo de anotaciones en formato COCO.
        pic_name (str, optional): Nombre de la imagen o parche para identificar los recortes.
        with_metadata (bool, optional): Si se deben guardar los metadatos de las imágenes recortadas. Defaults to True.
        output_folder (Path, optional): Carpeta donde se guardarán los recortes. Defaults to DOWNLOAD_CUTOUTS_FOLDER.
        metadata_folder (Path, optional): Carpeta donde se guardarán los metadatos de las imágenes recortadas. Defaults to DOWNLOAD_CUTOUTS_METADATA_FOLDER.

    Returns:
        Optional[Path]: Ruta de la carpeta donde se guardaron los recortes.
    """
    image = cv.imread(str(image_path))
    pic_name = image_path.stem
    coco_annotations = CocoDatasetUtils.load_annotations_from_path(coco_annotations_path)

    return cut_palms_from_image(
        image=image,
        coco_annotations=coco_annotations,
        pic_name=pic_name,
        with_metadata=with_metadata,
        output_folder=output_folder,
        metadata_folder=metadata_folder,
    )


@app.command()
def cut_palms_from_images_path(
    images_paths: list[Path] = typer.Option(..., help="Lista de rutas de imágenes a recortar"),
    coco_annotations_path: Path = typer.Option(..., help="Ruta del archivo de anotaciones en formato COCO"),
    with_metadata: bool = True,
    output_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
    metadata_folder: Path = DOWNLOAD_CUTOUTS_METADATA_FOLDER,
) -> None:
    """
    Recorta las palmas de una lista de imágenes utilizando las anotaciones proporcionadas en formato COCO.

    Args:
        images_paths (list): Lista de rutas de imágenes a recortar.
        coco_annotations_path (Path): Ruta del archivo de anotaciones en formato COCO.
        with_metadata (bool, optional): Si se deben guardar los metadatos de las imágenes recortadas. Defaults to True.
        output_folder (Path, optional): Carpeta donde se guardarán los recortes. Defaults to DOWNLOAD_CUTOUTS_FOLDER.
        metadata_folder (Path, optional): Carpeta donde se guardarán los metadatos de las imágenes recortadas. Defaults to DOWNLOAD_CUTOUTS_METADATA_FOLDER.

    Ejemplo de uso:

    >>> py -m modulo_apps.labeling.procesador_recortes cut-palms-from-images-path --images-paths "img1.jpg" --images-paths "img2.jpg" --coco-annotations-path "coco_anotations.json"
    """
    coco_annotations = CocoDatasetUtils.load_annotations_from_path(coco_annotations_path)
    images = [cv.imread(str(image_path)) for image_path in images_paths]
    pic_names = [image_path.stem for image_path in images_paths]

    for image, pic_name in tqdm(zip(images, pic_names), total=len(images), desc="Recortando imágenes"):
        if image is None or not image.any():
            LOGGER.error(f"La imagen {pic_name} no se pudo cargar. Verifique la ruta y el formato.")
            continue

        cut_palms_from_image(
            image=image,
            coco_annotations=coco_annotations,
            pic_name=pic_name,
            with_metadata=with_metadata,
            output_folder=output_folder,
            metadata_folder=metadata_folder,
        )


def _clear_minio_cutouts_folder(folder: str, is_metadata: bool = False) -> None:
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
        f"{MINIO_CUTOUTS_PATH}/{folder}/" if not is_metadata else f"{MINIO_CUTOUTS_METADATA_PATH}/{folder}/"
    )
    objects_to_delete = S3_CLIENT.list_objects(Bucket=MINIO_BUCKET, Prefix=prefix_key_path)
    delete_keys = {"Objects": []}
    delete_keys["Objects"] = [{"Key": k} for k in [obj["Key"] for obj in objects_to_delete.get("Contents", [])]]
    if delete_keys["Objects"]:
        S3_CLIENT.delete_objects(Bucket=MINIO_BUCKET, Delete=delete_keys)
        LOGGER.debug(f"Se eliminaron {len(delete_keys['Objects'])} objetos de MinIO con el prefijo {prefix_key_path}.")
    else:
        LOGGER.debug(f"No se encontraron objetos para eliminar con el prefijo {prefix_key_path}.")


@app.command()
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
        _clear_minio_cutouts_folder(folder, is_metadata=False)

        # Subir las imágenes a MinIO
        for _, _, files in os.walk(folder_path):
            for filename in files:
                S3_CLIENT.upload_file(
                    Filename=os.path.join(folder_path, filename),
                    Bucket=MINIO_BUCKET,
                    Key=f"{MINIO_CUTOUTS_PATH}/{folder}/{filename}",
                )
                LOGGER.debug(f"Imagen {filename} subida a MinIO.")

    if cutout_metadata_folder:
        for folder in os.listdir(cutout_metadata_folder):
            folder_path = cutout_metadata_folder / folder

            # Eliminamos el contenido del folder en MinIO
            _clear_minio_cutouts_folder(folder, is_metadata=True)

            # Subimos los metadatos a MinIO
            for _, _, files in os.walk(folder_path):
                for filename in files:
                    # Subir el archivo JSON de metadatos a MinIO
                    S3_CLIENT.upload_file(
                        Filename=os.path.join(folder_path, filename),
                        Bucket=MINIO_BUCKET,
                        Key=f"{MINIO_CUTOUTS_METADATA_PATH}/{folder}/{filename}",
                    )
                    LOGGER.debug(f"Archivo JSON de metadatos {filename} subido a MinIO.")


@app.command()
def process_cutouts():
    """
    Procesa las imágenes recortadas y sus metadatos.

    Este comando realiza las siguientes acciones:
    1. Descarga todas las imágenes que tienen anotaciones almacenadas en MongoDB.
    2. Descarga las anotaciones en formato COCO desde MongoDB.
    3. Recorta las imágenes utilizando las anotaciones descargadas.
    4. Sube los recortes y sus metadatos generados a MinIO.

    Este flujo automatiza el procesamiento de imágenes y su almacenamiento en MinIO,
    asegurando que las imágenes recortadas y sus metadatos estén disponibles para su uso posterior.
    """
    # 1. Descargar todas las imágenes que tienen anotaciones en MongoDB.
    images_names = ProcesadorCocoDataset.list_images_w_ann_from_mongodb()
    if not images_names:
        LOGGER.warning("No se encontraron imágenes con anotaciones en MongoDB.")
        return

    # ProcesadorS3.download_images_from_minio(images_names, DOWNLOAD_IMAGES_FOLDER)
    images_paths = [DOWNLOAD_IMAGES_FOLDER / f"{image_name}.jpg" for image_name in images_names]
    # 2. Descargar las anotaciones en formato COCO desde MongoDB.
    annotations_path = ProcesadorCocoDataset.download_annotations_as_coco_from_mongodb(
        images_names=images_names, field_name="cvat"
    )
    # 3. Recortar las imágenes utilizando las anotaciones.
    cut_palms_from_images_path(images_paths=images_paths, coco_annotations_path=annotations_path)
    
    # 4. Subir los recortes y sus metadatos a MinIO.
    upload_cutouts_to_mino()

if __name__ == "__main__":
    app()
