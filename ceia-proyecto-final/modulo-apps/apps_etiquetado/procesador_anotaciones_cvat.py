import shutil, zipfile, os, sys

sys.path.append(os.path.abspath("../"))

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cvat_sdk import make_client
from cvat_sdk.core.proxies.types import Location

from apps_config.settings import Config
from apps_utils.logging import Logging
from apps_com_db.mongodb_client import MongoDB

import apps_etiquetado.procesador_anotaciones_coco_dataset as CocoDatasetUtils
import apps_etiquetado.convertor_cordenadas as ConvertorCoordenadas

CONFIG = Config().config_data
LOGGER = Logging().logger
DB = MongoDB().db

MINIO_PATCHES_FOLDER = CONFIG["minio"]["paths"]["patches"]
download_folder = Path(CONFIG["folders"]["download_folder"])
DOWNLOAD_TEMP_FOLDER = download_folder / "temp"
DOWNLOAD_JOB_FOLDER = download_folder / "jobs"
DOWNLOAD_TASK_FOLDER = download_folder / "tasks"


def test_connection():
    """Testea la conexión a CVAT y devuelve True si la conexión es exitosa, False en caso contrario."""
    try:
        with make_client(
            host=CONFIG["cvat"]["url"],
            credentials=(CONFIG["cvat"]["user"], CONFIG["cvat"]["password"]),
        ) as client:
            client.get_server_version()
            LOGGER.debug("Conexión a CVAT exitosa.")
    except Exception as e:
        raise ConnectionError(f"Error al conectar con CVAT: {e}")


def download_annotations_from_cvat(
    task_id: Optional[int] = None,
    job_id: Optional[int] = None,
    output_filename: Optional[Path] = None,
) -> Optional[Path]:
    """Descarga las anotaciones de una tarea o de un job de CVAT y las guarda en un archivo JSON.
    Si el archivo ya existe, se elimina y se vuelve a descargar.

    Se guarda el archivo JSON en la carpeta de descarga especificada en la configuración.
    Se eliminan los archivos temporales después de la descarga.

    Args:
        task_id (int): Identificador de la tarea de CVAT a descargar.
        job_id (int): Identificador del job de CVAT a descargar.
        output_filename (Path, optional): Nombre del archivo a descargar. Si no se proporciona, se utiliza el nombre por defecto.

    Returns:
        Path: Ruta del archivo JSON descargado, o None en caso de error.
    """
    if bool(task_id is None) == bool(job_id is None):  # xor
        raise ValueError("Se debe proporcionar un task_id o un job_id.")
    try:
        client = make_client(
            host=CONFIG["cvat"]["url"],
            credentials=(CONFIG["cvat"]["user"], CONFIG["cvat"]["password"]),
        )

        try:
            retrieved = client.jobs.retrieve(job_id).fetch() if job_id else client.tasks.retrieve(task_id).fetch()
            retrieved_id = job_id if job_id else task_id

            Path(DOWNLOAD_TEMP_FOLDER).mkdir(parents=True, exist_ok=True)
            if output_filename is None:
                output_filename = Path(DOWNLOAD_TEMP_FOLDER) / f"cvat_task_{retrieved_id}.zip"

            if output_filename.exists():
                output_filename.unlink()
                LOGGER.warning(
                    f"Archivo {output_filename} ya existía, se eliminó para seguir el proceso (se lo descarga otra vez)."
                )

            retrieved.export_dataset(
                format_name=CONFIG["cvat"]["export_format"],
                filename=str(output_filename),
                include_images=False,
                location=Location.LOCAL,
            )
        except Exception as e:
            msg = (
                f"Error al descargar el job {job_id}: {e}" if job_id else f"Error al descargar la tarea {task_id}: {e}"
            )
            raise ValueError(msg)
        LOGGER.debug(f"Archivo {output_filename} descargado correctamente.")
    except TimeoutError as e:
        raise ConnectionError(f"Error al conectar con CVAT: {e}")
    except Exception as e:
        raise Exception(f"Error con CVAT: {e}")

    try:
        with zipfile.ZipFile(output_filename, "r") as zip_ref:
            zip_ref.extractall(DOWNLOAD_TEMP_FOLDER)
        LOGGER.debug(f"Archivo {output_filename} descomprimido correctamente.")
        # Chequeamos si el archivo descargado fue correcto
        json_file = DOWNLOAD_TEMP_FOLDER / "annotations" / "instances_default.json"
        if not json_file.exists():
            raise FileNotFoundError(f"El archivo {json_file} no existe después de descomprimir.")
    except FileNotFoundError as e:
        raise
    except zipfile.BadZipFile as e:
        raise Exception(f"Error al descomprimir el archivo {output_filename}") from e

    if job_id:
        Path(DOWNLOAD_JOB_FOLDER).mkdir(parents=True, exist_ok=True)
        output_json_file = Path(DOWNLOAD_JOB_FOLDER) / f"cvat_job_{job_id}.json"
    else:
        Path(DOWNLOAD_TASK_FOLDER).mkdir(parents=True, exist_ok=True)
        output_json_file = Path(DOWNLOAD_TASK_FOLDER) / f"cvat_task_{task_id}.json"

    shutil.copy(json_file, output_json_file)
    LOGGER.debug(f"Archivo JSON copiado a {output_json_file}.")

    try:
        output_filename.unlink()
        LOGGER.debug(f"Archivo {output_filename} eliminado correctamente.")

        json_file.unlink()
        LOGGER.debug(f"Archivo {json_file} eliminado correctamente.")
    except OSError as e:
        LOGGER.warning(f"Error al eliminar el archivos: {e}")

    return output_json_file


def load_annotations_from_cvat(
    task_id: int = None, job_id: int = None, clean_files: bool = True
) -> Optional[Dict[str, Any]]:
    """Carga las anotaciones desde CVAT.

    Este método permite descargar y cargar anotaciones desde CVAT, ya sea para una tarea
    específica o un trabajo específico. Las anotaciones se descargan en formato JSON y
    se cargan en un diccionario.@

    Args:
        task_id (int, optional): Identificador de la tarea en CVAT. Defaults to None.
        job_id (int, optional): Identificador del trabajo en CVAT. Defaults to None.
        clean_files (bool, optional): Si se deben eliminar los archivos temporales después de la carga. Defaults to True.

    Raises:
        ValueError: Si no se proporciona ni un task_id ni un job_id.

    Returns:
        Dict[str, Any]: Diccionario con las anotaciones cargadas en formato COCO.
    """
    if bool(task_id is None) == bool(job_id is None):  # xor
        raise ValueError("Se debe proporcionar un task_id o un job_id.")
    try:
        file_path = (
            download_annotations_from_cvat(task_id=task_id)
            if task_id
            else download_annotations_from_cvat(job_id=job_id)
        )
    except Exception as e:
        raise Exception(f"Error al descargar las anotaciones: {e}")

    if file_path:
        annotations = CocoDatasetUtils.load_annotations_from_file(file_path)
        LOGGER.debug(f"Anotaciones cargadas desde {file_path}.")
        if clean_files and file_path and file_path.exists():
            try:
                file_path.unlink()
                LOGGER.debug(f"Archivo {file_path} eliminado correctamente.")
            except OSError as e:
                LOGGER.warning(f"Error al eliminar el archivo: {e}")
        return annotations
    else:
        raise FileNotFoundError(f"No se pudo encontrar el archivo de anotaciones en {file_path}.")


def convert_image_annotations_to_cvat_annotations(
    images: Dict[str, Any], annotations: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Esta función convierte las anotaciones de las imágenes a un formato igual al que se descarga de CVAT. O sea, en base a los parches de la imagen.

    Se utiliza para cargar directamente las anotaciones en que las imagenes es la imagen completa y no parches.
    Para ello, se transforman las imágenes a los parches asociados. O sea, los bbox de las imágenes se asocian a
    parches correspondientes.

    Este método modifica las anotaciones, ya que los bbox de la imagen ahora pertenece a los parches.
    Como resumen, el diccionario "images" tendría los parches y el diccionario "annotations" tendría los bbox de los parches,
    pero todo esto en concordancia con los bboxes de la imagen original.

    Args:
        images (Dict[str, Any]): Diccionario que contiene las imágenes y sus metadatos.
        annotations (Dict[str, Any]): Diccionario que contiene las anotaciones y sus metadatos.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: imágenes y anotaciones convertidas al formato de CVAT.
    """
    output_images = []
    output_annotations = []

    imagenes = DB.get_collection("imagenes")
    patches_data_list = []
    for image in images:
        image_annotations = [ann for ann in annotations if ann["image_id"] == image["id"]]
        if not image_annotations:
            LOGGER.warning(f"No se encontraron anotaciones para la imagen {image['id']}.")
            continue

        # Obtener los parches asociados a la imagen
        image_name = image["file_name"].split(".")[0]  # Obtenemos el nombre de la imagen sin la extensión
        db_image = imagenes.find_one({"id": image_name})
        if not db_image:
            raise ValueError(
                f"No se encontraron parches para la imagen {image['id']}. Toda imagen debe tener al menos un parche asociados."
            )

        for db_patch in db_image["patches"]:
            patch_data = {}
            file_name = f"{MINIO_PATCHES_FOLDER}/{image_name}/{db_patch['patch_name']}.jpg"
            patch_data["image"] = {
                "file_name": file_name,
                "height": db_patch["height"],
                "width": db_patch["width"],
                "date_captured": db_image["date_captured"].strftime("%Y-%m-%d %H:%M:%S"),
            }
            patch_data["annotations"] = []
            for annotation in image_annotations:
                # Verificar si el bbox de la imagen está dentro del parche
                patch_bbox = ConvertorCoordenadas.convert_bbox_image_to_patch(
                    annotation["bbox"],
                    db_patch["x_start"],
                    db_patch["y_start"],
                    db_patch["width"],
                    db_patch["height"],
                )
                if patch_bbox is not None:
                    # Transformar el bbox de la imagen a coordenadas locales del parche
                    patch_data["annotations"].append({**annotation, "bbox": patch_bbox})
            if not patch_data["annotations"]:
                LOGGER.warning(
                    f"No se encontraron anotaciones para el parche {db_patch['patch_name']} de la imagen {image_name}."
                )
                continue

            # Agregar el parche a la lista de parches
            patches_data_list.append(patch_data)

    # Procesar los parches
    for id, patch_data in enumerate(patches_data_list):
        image_id = id + 1
        image = {
            "id": image_id,
            **patch_data["image"],
        }

        annotations = [
            {
                "id": i + 1,
                "image_id": image_id,
                **ann,
            }
            for i, ann in enumerate(patch_data["annotations"])
        ]

        output_images.append(image)
        output_annotations.extend(annotations)

    return output_images, output_annotations


def convert_patch_annotations_to_cvat_annotations(
    images: Dict[str, Any], annotations: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Esta función convierte las anotaciones de los parches a un formato igual al que se descarga de CVAT.

    Se utiliza para cargar directamente las anotaciones en que las imágenes son parches, por lo que están
    en el formato:
    {
        "id": 0,
        "width": 4096,
        "height": 4096,
        "file_name": "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_0.jpg",
        "date_captured": "2025-05-05 21:00:34"
    },

    El objetivo es simplemente cambiar el "file_name" agregando el nombre de la imagen original (obtenido de la base de datos)
    y el prefijo "parche", para que se pueda procesar como si fuera descargado de CVAT.
    Para el diccionario de anotaciones, no se realiza ningún cambio, ya que se espera que
    ya esté en el formato correcto.

    Args:
        images (Dict[str, Any]): Diccionario con las imágenes y sus metadatos.
        annotations (Dict[str, Any]): Diccionario con las anotaciones y sus metadatos.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: imagenes y anotaciones convertidas al formato de CVAT.
    """
    imagenes = DB.get_collection("imagenes")
    for image in images:
        file_name = image["file_name"]
        patch_name = file_name.split(".")[0]  # Quitar la extensión

        # Obtener el nombre de la imagen original desde la base de datos
        image_name = imagenes.find_one({"patches.patch_name": patch_name})
        if not image_name:
            raise ValueError(f"No se encontró la imagen original para el parche {patch_name}.")

        image["file_name"] = f"{MINIO_PATCHES_FOLDER}/{image_name['id']}/{image["file_name"]}"

    return images, annotations
