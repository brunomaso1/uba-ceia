import os, sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

sys.path.append(os.path.abspath("../"))

import botocore
from apps_config.settings import Config
from apps_utils.logging import Logging
from apps_com_db.mongodb_client import MongoDB
from apps_com_s3.cliente_s3 import S3Client

CONFIG = Config().config_data
LOGGER = Logging().logger
DB = MongoDB().db

MINIO_BUCKET = CONFIG["minio"]["bucket"]

download_folder = Path(CONFIG["folders"]["download_folder"])
DOWNLOAD_IMAGES_FOLDER = download_folder / "images"
DOWNLOAD_PATCHES_FOLDER = download_folder / "patches"


class ProcesadorS3:
    def __init__(self, minio_client: botocore.client = S3Client().client) -> None:
        self.minio_client = minio_client

    def download_image_from_minio(self, image_name: str, output_filename: Optional[Path] = None) -> Optional[Path]:
        """Descarga una imagen desde MinIO y la guarda en la ruta especificada.

        Este método permite descargar una imagen desde el almacenamiento MinIO y guardarla
        en una ruta local. Si no se proporciona una ruta, la imagen se guarda en la carpeta
        de descargas configurada.

        Args:
            image_name (str): Identificador de la imagen a descargar.
            path (str, optional): Ruta local donde se guardará la imagen. Defaults to None.

        Ejemplo de uso:

            >>> download_image_from_minio("8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm", "imagen.jpg")
            >>> # Esto descargará la imagen con el ID especificado y la guardará como "imagen.jpg".
        """
        if not output_filename:
            Path(DOWNLOAD_IMAGES_FOLDER).mkdir(parents=True, exist_ok=True)
            output_filename = DOWNLOAD_IMAGES_FOLDER / f"{image_name}.jpg"

        image_key = f"{CONFIG["minio"]["paths"]["images"]}/{image_name}.jpg"

        try:
            self.minio_client.download_file(Bucket=MINIO_BUCKET, Key=image_key, Filename=output_filename)
            LOGGER.debug(f"Imagen {image_name} descargada correctamente en {output_filename}.")
        except Exception as e:
            raise Exception(f"Error al descargar la imagen {image_name}: {e}")

        return output_filename

    def download_images_from_minio(self, images_name: list[str], folder_path: Optional[Path] = None) -> bool:
        """Descarga una lista de imágenes desde MinIO.

        Este método permite descargar múltiples imágenes desde el almacenamiento MinIO y guardarlas
        en la carpeta de descargas configurada.

        Args:
            images_name (list[str]): Lista de identificadores de las imágenes a descargar.

        Ejemplo de uso:

            >>> images_name = [
            ...    "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm",
            ...    "AntelArena_20200804_dji_pc_5c"
            ... ]
            >>> download_images_from_minio(images_name)
            >>> # Esto descargará las imágenes con los IDs especificados y las guardará en la carpeta de descargas.
        """
        if not folder_path:
            Path(DOWNLOAD_IMAGES_FOLDER).mkdir(parents=True, exist_ok=True)
            folder_path = DOWNLOAD_IMAGES_FOLDER
        try:
            for image_name in tqdm(images_name, desc="Descargando imágenes", unit="MB"):
                output_filename = folder_path / f"{image_name}.jpg"
                self.download_image_from_minio(image_name, output_filename)
            LOGGER.debug(f"Se descargaron {len(images_name)} imágenes de MinIO.")
        except Exception as e:
            LOGGER.error(f"Error al descargar imágenes de MinIO: {e}")
            raise e
        return True

    def download_patch_from_minio(self, patch_name: str, output_filename: Optional[Path] = None) -> Optional[Path]:
        """Descarga un parche desde MinIO y lo guarda en la ruta especificada.

        Este método permite descargar un parche desde el almacenamiento MinIO y guardarlo
        en una ruta local. Si no se proporciona una ruta, el parche se guarda en la carpeta
        de descargas configurada.

        Args:
            patch_name (str): Nombre del parche a descargar.
            path (str, optional): Ruta local donde se guardará el parche. Defaults to None.

        Ejemplo de uso:
            >>> download_patch_from_minio("8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_0.jpg", "parche.jpg")
            >>> # Esto descargará el parche con el nombre especificado y lo guardará como "parche.jpg".
        """
        if not output_filename:
            Path(DOWNLOAD_PATCHES_FOLDER).mkdir(parents=True, exist_ok=True)
            output_filename = DOWNLOAD_PATCHES_FOLDER / f"{patch_name}.jpg"

        # Obtener el nombre de la imagen del parche desde mongodb
        imagenes = DB.get_collection("imagenes")
        image = imagenes.find_one({"patches.patch_name": patch_name})
        if not image:
            raise ValueError(f"No se encontró la imagen del parche {patch_name} en la base de datos.")

        patch = next((p for p in image.get("patches", []) if p["patch_name"] == patch_name), None)

        if patch["is_white"]:
            LOGGER.warning(f"El parche {patch_name} es blanco, no se descargará.")
            return None

        image_name = image["id"]
        patch_key = f"{CONFIG['minio']["paths"]['patches']}/{image_name}/{patch_name}.jpg"

        try:
            self.minio_client.download_file(Bucket=MINIO_BUCKET, Key=patch_key, Filename=output_filename)
            LOGGER.debug(f"Parche {patch_name} descargado correctamente en {output_filename}.")
        except Exception as e:
            raise Exception(f"Error al descargar el parche {patch_name}: {e}")

        return output_filename

    def download_patches_from_minio(self, patch_names: List[str], folder_path: Optional[Path] = None) -> None:
        """Descarga una lista de parches desde MinIO.

        Este método permite descargar múltiples parches desde el almacenamiento MinIO y guardarlos
        en la carpeta de descargas configurada.

        Args:
            patch_names (list[str]): Lista de nombres de los parches a descargar.

        Ejemplo de uso:

            >>> patch_names = [
            ...    "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_0.jpg",
            ...    "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_2.jpg"
            ... ]
            >>> download_patches_from_minio(patch_names)
            >>> # Esto descargará los parches con los nombres especificados y los guardará en la carpeta de descargas.
        """
        if not folder_path:
            Path(DOWNLOAD_PATCHES_FOLDER).mkdir(parents=True, exist_ok=True)
            folder_path = DOWNLOAD_PATCHES_FOLDER
        try:
            for patch_name in tqdm(patch_names, desc="Descargando parches", unit="MB"):
                output_filename = folder_path / f"{patch_name}.jpg"
                self.download_patch_from_minio(patch_name, output_filename)
            LOGGER.debug(f"Se descargaron {len(patch_names)} parches de MinIO.")
        except Exception as e:
            raise Exception(f"Error al descargar parches de MinIO: {e}")
        return True
