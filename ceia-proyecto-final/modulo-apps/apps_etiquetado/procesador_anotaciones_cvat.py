import shutil, zipfile, os, sys

sys.path.append(os.path.abspath("../"))

from pathlib import Path
from typing import Any, Dict, Optional

from cvat_sdk import make_client
from cvat_sdk.core.proxies.types import Location

from apps_config.settings import Config
from apps_utils.logging import Logging

import apps_etiquetado.utils_coco_dataset as CocoDatasetUtils

CONFIG = Config().config_data
LOGGER = Logging().logger

download_folder = Path(CONFIG["folders"]["download_folder"])
DOWNLOAD_TEMP_FOLDER = download_folder / "temp"
DOWNLOAD_JOB_FOLDER = download_folder / "jobs"
DOWNLOAD_TASK_FOLDER = download_folder / "tasks"


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
            self.download_annotations_from_cvat(task_id=task_id)
            if task_id
            else self.download_annotations_from_cvat(job_id=job_id)
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
