import re
from pathlib import Path
from typing import List

from tqdm import tqdm
import typer

import botocore
from loguru import logger as LOGGER
from modulo_apps.config import config as CONFIG
from modulo_apps.database_comunication.mongodb_client import mongodb as DB
from modulo_apps.s3_comunication.s3_client import s3client as S3CLIENT

MINIO_BUCKET = CONFIG.minio.bucket
DOWNLOAD_FOLDER = CONFIG.folders.download_folder
DOWNLOAD_IMAGES_FOLDER = CONFIG.folders.download_images_folder
DOWNLOAD_PATCHES_FOLDER = CONFIG.folders.download_patches_folder
DOWNLOAD_CUTOUTS_FOLDER = CONFIG.folders.download_cutouts_folder
DOWNLOAD_CUTOUTS_METADATA_FOLDER = CONFIG.folders.download_cutouts_metadata_folder

app = typer.Typer()


@app.command()
def check_connection() -> None:
    """Verifica la conexión con el servicio S3 (MinIO)."""
    try:
        S3CLIENT.head_bucket(Bucket=MINIO_BUCKET)
        LOGGER.success(f"Conexión exitosa a MinIO y acceso al bucket '{MINIO_BUCKET}' verificado.")
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "403":
            raise PermissionError(f"Acceso denegado al bucket '{MINIO_BUCKET}'. Verifica tus credenciales.")
        elif error_code == "404":
            raise FileNotFoundError(f"El bucket '{MINIO_BUCKET}' no existe. Verifica la configuración.")
        else:
            raise ConnectionError(f"Error al conectar con MinIO: {e}")
    except Exception as e:
        raise ConnectionError(f"Error inesperado al conectar con MinIO: {e}")


@app.command()
def download_image_from_minio(
    image_name: str,
    output_folder: Path = DOWNLOAD_IMAGES_FOLDER,
) -> Path:
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
    output_folder.mkdir(parents=True, exist_ok=True)
    output_filename = output_folder / f"{image_name}.jpg"

    image_key = f"{CONFIG.minio.paths.images}/{image_name}.jpg"

    try:
        S3CLIENT.download_file(Bucket=MINIO_BUCKET, Key=image_key, Filename=output_filename)
        LOGGER.debug(f"Imagen {image_name} descargada correctamente en {output_filename}.")
    except Exception as e:
        raise Exception(f"Error al descargar la imagen {image_name}: {e}")

    return output_filename


@app.command()
def download_images_from_minio(images_name: List[str], output_folder: Path = DOWNLOAD_IMAGES_FOLDER) -> bool:
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
    output_folder.mkdir(parents=True, exist_ok=True)
    LOGGER.debug(f"Descargando {len(images_name)} imágenes de MinIO a {output_folder}.")

    try:
        for image_name in tqdm(images_name, desc="Descargando imágenes", unit="MB"):
            download_image_from_minio(image_name, output_folder)
        LOGGER.debug(f"Se descargaron {len(images_name)} imágenes de MinIO.")
    except Exception as e:
        LOGGER.error(f"Error al descargar imágenes de MinIO: {e}")
        raise e
    return True


@app.command()
def download_patch_from_minio(patch_name: str, output_folder: Path = DOWNLOAD_PATCHES_FOLDER) -> Path:
    """Descarga un parche desde MinIO y lo guarda en la ruta especificada.

    Este método permite descargar un parche desde el almacenamiento MinIO y guardarlo
    en una ruta local. Si no se proporciona una ruta, el parche se guarda en la carpeta
    de descargas configurada.

    Args:
        patch_name (str): Nombre del parche a descargar.
        path (str, optional): Ruta local donde se guardará el parche. Defaults to None.

    Ejemplo de uso:
        >>> download_patch_from_minio("8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_0", "parche.jpg")
        >>> # Esto descargará el parche con el nombre especificado y lo guardará como "parche.jpg".
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    output_filename = output_folder / f"{patch_name}.jpg"

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
    patch_key = f"{CONFIG.minio.paths.patches}/{image_name}/{patch_name}.jpg"

    try:
        S3CLIENT.download_file(Bucket=MINIO_BUCKET, Key=patch_key, Filename=output_filename)
        LOGGER.debug(f"Parche {patch_name} descargado correctamente en {output_filename}.")
    except Exception as e:
        raise Exception(f"Error al descargar el parche {patch_name}: {e}")

    return output_filename


@app.command()
def download_patches_from_minio(patch_names: List[str], output_folder: Path = DOWNLOAD_PATCHES_FOLDER) -> None:
    """Descarga una lista de parches desde MinIO.

    Este método permite descargar múltiples parches desde el almacenamiento MinIO y guardarlos
    en la carpeta de descargas configurada.

    Args:
        patch_names (list[str]): Lista de nombres de los parches a descargar.

    Ejemplo de uso:

        >>> patch_names = [
        ...    "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_0",
        ...    "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_2"
        ... ]
        >>> download_patches_from_minio(patch_names)
        >>> # Esto descargará los parches con los nombres especificados y los guardará en la carpeta de descargas.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        for patch_name in tqdm(patch_names, desc="Descargando parches", unit="MB"):
            download_patch_from_minio(patch_name, output_folder)
        LOGGER.debug(f"Se descargaron {len(patch_names)} parches de MinIO.")
    except Exception as e:
        raise Exception(f"Error al descargar parches de MinIO: {e}")
    return True


@app.command()
def download_cutout_from_minio(
    cutout_name: str,
    output_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
) -> Path:
    """Descarga un recorte desde MinIO y lo guarda en la ruta especificada.

    Este método permite descargar un recorte desde el almacenamiento MinIO y guardarlo
    en una ruta local. Si no se proporciona una ruta, el recorte se guarda en la carpeta
    de descargas configurada.

    Args:
        cutout_name (str): Nombre del recorte a descargar.
        output_filename (Optional[Path], optional): Ruta local donde se guardará el recorte.
            Si no se especifica, se utiliza la carpeta de descargas configurada.

    Raises:
        Exception: Si ocurre un error durante la descarga del recorte.

    Returns:
        Path: Ruta donde se guardó el recorte descargado.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    remove_after_cutout = lambda filename: re.sub(r"_cutout.*$", "", filename)
    image_name = remove_after_cutout(cutout_name)
    output_filename = output_folder / image_name / f"{cutout_name}.jpg"
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    cutout_key = f"{CONFIG.minio.paths.cutouts}/{image_name}/{cutout_name}.jpg"

    try:
        S3CLIENT.download_file(Bucket=MINIO_BUCKET, Key=cutout_key, Filename=output_filename)
        LOGGER.debug(f"Recorte {cutout_name} descargado correctamente en {output_filename}.")
    except Exception as e:
        raise Exception(f"Error al descargar el recorte {cutout_name}: {e}")

    return output_filename


@app.command()
def download_cutouts_from_minio(
    cutout_names: List[str],
    output_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
) -> None:
    """Descarga una lista de recortes desde MinIO.

    Este método permite descargar múltiples recortes desde el almacenamiento MinIO y guardarlos
    en la carpeta de descargas configurada.

    Args:
        cutout_names (list[str]): Lista de nombres de los recortes a descargar.

    Ejemplo de uso:

        >>> cutout_names = [
        ...    "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_cutout_1",
        ...    "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_cutout_2"
        ... ]
        >>> download_cutouts_from_minio(cutout_names)
        >>> # Esto descargará los recortes con los nombres especificados y los guardará en la carpeta de descargas.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        for cutout_name in tqdm(cutout_names, desc="Descargando recortes", unit="MB"):
            download_cutout_from_minio(cutout_name, output_folder)
        LOGGER.debug(f"Se descargaron {len(cutout_names)} recortes de MinIO.")
    except Exception as e:
        raise Exception(f"Error al descargar recortes de MinIO: {e}")


# Reutilizamos la función auxiliar
def _download_objects_by_prefix(
    prefix_key_path: str,
    output_folder: Path,
    bucket_name: str,
    s3_client,
) -> None:
    """
    Descarga todos los objetos de MinIO que coinciden con un prefijo dado.

    Args:
        prefix_key_path (str): El prefijo de los objetos a descargar.
        output_folder (Path): La carpeta local donde se guardarán los objetos.
        bucket_name (str): El nombre del bucket de MinIO.
        s3_client: El cliente de MinIO/S3 (ej., boto3.client).
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    LOGGER.debug(f"Buscando objetos con el prefijo '{prefix_key_path}' en el bucket '{bucket_name}'...")

    try:
        objects_list_response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_key_path)
        if "Contents" not in objects_list_response:
            LOGGER.warning(f"No se encontraron objetos con el prefijo '{prefix_key_path}'.")
            return

        objects_to_download = objects_list_response["Contents"]

        with tqdm(
            total=len(objects_to_download), desc=f"Descargando {Path(prefix_key_path).name}", unit="archivo"
        ) as pbar:
            downloaded_count = 0
            for obj in objects_to_download:
                object_key = obj["Key"]
                file_name = Path(object_key).name
                local_file_path = output_folder / file_name

                try:
                    s3_client.download_file(bucket_name, object_key, str(local_file_path))
                    downloaded_count += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"'{file_name}'")
                except Exception as e:
                    LOGGER.error(f"Error al descargar '{object_key}': {e}")
                    pbar.write(f"Error al descargar '{object_key}': {e}")

        LOGGER.debug(
            f"Se descargaron {downloaded_count} objetos con el prefijo '{prefix_key_path}' en '{output_folder}'."
        )

    except Exception as e:
        LOGGER.error(f"Error al listar o descargar objetos de MinIO: {e}")


@app.command()
def download_image_cutouts_from_minio(
    image_name: str,
    with_metadata: bool = False,
    output_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
) -> Path:
    """
    Descarga recortes de imágenes (y opcionalmente metadatos) desde MinIO.

    Args:
        image_name (str): El nombre de la imagen para la cual se desean descargar los recortes.
        with_metadata (bool, optional): Indica si también se deben descargar los metadatos.
                                        Defaults to False.
        output_folder (Path, optional): La carpeta base donde se guardarán los recortes.
                                        Defaults to DOWNLOAD_CUTOUTS_FOLDER.
    Returns:
        Path: La ruta a la carpeta donde se descargaron los archivos.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    cutouts_prefix = f"{CONFIG.minio.paths.cutouts}/{image_name}/"
    output_folder /= image_name

    _download_objects_by_prefix(cutouts_prefix, output_folder, MINIO_BUCKET, S3CLIENT)

    if with_metadata:
        metadata_prefix = f"{CONFIG.minio.paths.cutouts_metadata}/{image_name}/"
        cutouts_metadata_folder = DOWNLOAD_CUTOUTS_METADATA_FOLDER / image_name
        _download_objects_by_prefix(metadata_prefix, cutouts_metadata_folder, MINIO_BUCKET, S3CLIENT)

    return output_folder


@app.command()
def download_images_cutouts_from_minio(
    image_names: List[str],
    with_metadata: bool = False,
    output_base_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
) -> None:
    """
    Descarga recortes (y opcionalmente metadatos) para una lista de imágenes desde MinIO.

    Args:
        image_names (List[str]): Una lista de nombres de imágenes para las cuales se desean descargar los recortes.
        with_metadata (bool, optional): Indica si también se deben descargar los metadatos. Defaults to False.
        output_base_folder (Path, optional): La carpeta base donde se guardarán todos los recortes.
                                           Si es None, se usará DOWNLOAD_CUTOUTS_FOLDER.
    """
    output_base_folder.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Iniciando la descarga de recortes para {len(image_names)} imágenes en '{output_base_folder}'.")

    with tqdm(total=len(image_names), desc="Procesando imágenes", unit="imagen") as pbar_images:
        for image_name in image_names:
            pbar_images.set_description(
                f"Procesando imagen: {image_name}"
            )  # Actualiza la descripción de la barra principal

            # Carpeta específica para los recortes de esta imagen
            image_cutouts_folder = output_base_folder / image_name
            image_cutouts_folder.mkdir(parents=True, exist_ok=True)

            cutouts_prefix = f"{CONFIG.minio.paths.cutouts}/{image_name}/"
            _download_objects_by_prefix(cutouts_prefix, image_cutouts_folder, MINIO_BUCKET, S3CLIENT)

            if with_metadata:
                metadata_prefix = f"{CONFIG.minio.paths.cutouts_metadata}/{image_name}/"
                image_metadata_folder = DOWNLOAD_CUTOUTS_METADATA_FOLDER / image_name
                image_metadata_folder.mkdir(parents=True, exist_ok=True)
                _download_objects_by_prefix(metadata_prefix, image_metadata_folder, MINIO_BUCKET, S3CLIENT)

            pbar_images.update(1)  # Actualiza la barra de progreso de imágenes

    LOGGER.info("Descarga de recortes para todas las imágenes completada.")


if __name__ == "__main__":
    app()
