from io import BufferedReader
import re
from pathlib import Path

from tqdm import tqdm
import typer

import botocore
from loguru import logger as LOGGER
from modulo_apps.config import config as CONFIG
from modulo_apps.database_comunication.mongodb_client import mongodb as DB
from modulo_apps.s3_comunication.s3_client import s3client as S3_CLIENT


DOWNLOAD_FOLDER = CONFIG.folders.download_folder
DOWNLOAD_IMAGES_FOLDER = CONFIG.folders.download_images_folder
DOWNLOAD_PATCHES_FOLDER = CONFIG.folders.download_patches_folder
DOWNLOAD_CUTOUTS_FOLDER = CONFIG.folders.download_cutouts_folder
DOWNLOAD_CUTOUTS_METADATA_FOLDER = CONFIG.folders.download_cutouts_metadata_folder

S3_BUCKET = CONFIG.minio.bucket
S3_IMAGE_PATH = CONFIG.minio.paths.images
S3_PATCHES_PATH = CONFIG.minio.paths.patches
S3_CUTOUTS_PATH = CONFIG.minio.paths.cutouts
S3_CUTOUTS_METADATA_PATH = CONFIG.minio.paths.cutouts_metadata
S3_JGW_PATH = CONFIG.minio.paths.metadata


app = typer.Typer()


@app.command()
def test_connection() -> None:
    """Verifica la conexión con el servicio S3 (S3)."""
    try:
        S3_CLIENT.head_bucket(Bucket=S3_BUCKET)
        LOGGER.success(f"Conexión exitosa a S3 y acceso al bucket '{S3_BUCKET}' verificado.")
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "403":
            raise PermissionError(f"Acceso denegado al bucket '{S3_BUCKET}'. Verifica tus credenciales.")
        elif error_code == "404":
            raise FileNotFoundError(f"El bucket '{S3_BUCKET}' no existe. Verifica la configuración.")
        else:
            raise ConnectionError(f"Error al conectar con S3: {e}")
    except Exception as e:
        raise ConnectionError(f"Error inesperado al conectar con S3: {e}")


@app.command()
def download_image_from_s3(
    image_name: str,
    group_id: str,
    output_folder: Path = DOWNLOAD_IMAGES_FOLDER,
    bucket_name: str = S3_BUCKET,
) -> Path:
    """
    Descarga una imagen JPEG desde un bucket S3/S3 y la guarda en un directorio local.
    Construye la clave del objeto como "{S3_IMAGE_PATH}/{group_id}/{image_name}.jpg" y guarda el archivo como
    "{output_folder}/{image_name}.jpg". Crea el directorio de salida si no existe y registra un mensaje de depuración
    cuando la descarga se completa.
    Args:
        image_name (str): Nombre base de la imagen sin extensión. Se asume la extensión ".jpg".
        group_id (str): Identificador del grupo utilizado para construir la clave del objeto en S3/S3.
        output_folder (Path, optional): Carpeta de destino donde se almacenará la imagen descargada. Se crea si no existe.
            Por defecto, DOWNLOAD_IMAGES_FOLDER.
        bucket_name (str, optional): Nombre del bucket de origen. Por defecto, MINIO_BUCKET.
    Returns:
        Path: Ruta del archivo de imagen descargado en el sistema local.
    Raises:
        Exception: Si ocurre un error durante la descarga desde S3/S3 o al guardar el archivo localmente.
    Notes:
        Requiere un cliente S3 configurado (S3CLIENT) y variables de configuración como S3_IMAGE_PATH.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    output_filename = output_folder / f"{image_name}.jpg"

    image_key = f"{S3_IMAGE_PATH}/{group_id}/{image_name}.jpg"

    try:
        S3_CLIENT.download_file(Bucket=bucket_name, Key=image_key, Filename=output_filename)
        LOGGER.debug(f"Imagen {image_name} descargada correctamente en {output_filename}.")
    except Exception as e:
        raise Exception(f"Error al descargar la imagen {image_name}: {e}")

    return output_filename


@app.command()
def download_images_from_s3(
    images_name: list[str], group_id: str, output_folder: Path = DOWNLOAD_IMAGES_FOLDER, bucket_name: str = S3_BUCKET
) -> bool:
    """Descarga un conjunto de imágenes desde un bucket de Amazon S3 a un directorio local.
    Crea el directorio de salida si no existe, muestra una barra de progreso durante la descarga
    y registra mensajes de depuración. Si ocurre un error en cualquiera de las descargas, se registra
    y la excepción se vuelve a propagar.
    Args:
        images_name (list[str]): Lista de nombres de objetos/archivos de imagen en S3 a descargar.
        group_id (str): Identificador del grupo o prefijo dentro del bucket que ayuda a localizar las imágenes.
        output_folder (pathlib.Path, optional): Directorio local de destino donde se guardarán las imágenes.
            Se crea si no existe. Por defecto, usa la carpeta configurada en la aplicación.
        bucket_name (str, optional): Nombre del bucket de S3 desde el cual descargar. Por defecto, el configurado en la aplicación.
    Returns:
        bool: True si todas las imágenes se descargan correctamente.
    Raises:
        Exception: Cualquier excepción ocurrida durante el proceso de descarga es registrada y vuelta a propagar.
    Example:
        >>> from pathlib import Path
        >>> download_images_from_s3(
        ...     images_name=["img1.jpg", "img2.png"],
        ...     group_id="grupo-123",
        ...     output_folder=Path("/ruta/de/salida"),
        ...     bucket_name="mi-bucket"
        ... )
        True
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    LOGGER.debug(f"Descargando {len(images_name)} imágenes de S3 a {output_folder}.")
    try:
        for image_name in tqdm(images_name, desc="Descargando imágenes", unit="MB"):
            download_image_from_s3(image_name, group_id, output_folder, bucket_name)
        LOGGER.debug(f"Se descargaron {len(images_name)} imágenes de S3.")
    except Exception as e:
        LOGGER.error(f"Error al descargar imágenes de S3: {e}")
        raise e
    return True


@app.command()
def download_patch_from_s3(
    patch_name: str, group_id, output_folder: Path = DOWNLOAD_PATCHES_FOLDER, bucket_name: str = S3_BUCKET
) -> Path:
    """
    Descarga un parche desde s3 y lo guarda en una carpeta local.
    Args:
        patch_name (str): Nombre del parche a descargar.
        group_id: Identificador del grupo al que pertenece el parche.
        output_folder (Path, optional): Carpeta de salida donde se guardará el parche.
            Por defecto es `DOWNLOAD_PATCHES_FOLDER`.
        bucket_name (str, optional): Nombre del bucket en MinIO. Por defecto es `S3_BUCKET`.
    Returns:
        Path: Ruta completa del archivo descargado.
    Raises:
        ValueError: Si no se encuentra la imagen asociada al parche en la base de datos.
        Exception: Si ocurre un error durante la descarga del archivo desde MinIO.
    Warnings:
        Si el parche es blanco (`is_white`), se registra una advertencia y no se descarga el archivo.
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
    patch_key = f"{CONFIG.minio.paths.patches}/{group_id}/{image_name}/{patch_name}.jpg"

    try:
        S3_CLIENT.download_file(Bucket=bucket_name, Key=patch_key, Filename=output_filename)
        LOGGER.debug(f"Parche {patch_name} descargado correctamente en {output_filename}.")
    except Exception as e:
        raise Exception(f"Error al descargar el parche {patch_name}: {e}")

    return output_filename


@app.command()
def download_patches_from_s3(
    patch_names: list[str], group_id: str, output_folder: Path = DOWNLOAD_PATCHES_FOLDER, bucket_name: str = S3_BUCKET
) -> None:
    """
    Descarga una lista de parches desde un bucket de S3 y los guarda en una carpeta local.
    Args:
        patch_names (list[str]): Lista de nombres de los parches a descargar.
        group_id (str): Identificador del grupo al que pertenecen los parches.
        output_folder (Path, optional): Carpeta de destino donde se guardarán los parches descargados.
            Por defecto es `DOWNLOAD_PATCHES_FOLDER`.
        bucket_name (str, optional): Nombre del bucket de S3 desde donde se descargarán los parches.
            Por defecto es `S3_BUCKET`.
    Returns:
        None
    Raises:
        Exception: Si ocurre un error durante la descarga de los parches.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    try:
        for patch_name in tqdm(patch_names, desc="Descargando parches", unit="MB"):
            download_patch_from_s3(patch_name, group_id, output_folder, bucket_name)
        LOGGER.debug(f"Se descargaron {len(patch_names)} parches de S3.")
    except Exception as e:
        raise Exception(f"Error al descargar parches de S3: {e}")
    return True


@app.command()
def download_cutout_from_s3(
    cutout_name: str,
    group_id: str,
    output_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
    bucket_name: str = S3_BUCKET,
) -> Path:
    """
    Descarga un recorte (cutout) desde un bucket de S3 y lo guarda en una carpeta local.
    Args:
        cutout_name (str): Nombre del recorte a descargar.
        group_id (str): Identificador del grupo al que pertenece el recorte.
        output_folder (Path, optional): Carpeta local donde se guardará el recorte.
            Por defecto, se utiliza la constante `DOWNLOAD_CUTOUTS_FOLDER`.
        bucket_name (str, optional): Nombre del bucket de S3 desde donde se descargará el recorte.
            Por defecto, se utiliza la constante `S3_BUCKET`.
    Returns:
        Path: Ruta completa del archivo descargado en el sistema de archivos local.
    Raises:
        Exception: Si ocurre un error durante la descarga del recorte desde S3.
    """
    remove_after_cutout = lambda filename: re.sub(r"_cutout.*$", "", filename)
    image_name = remove_after_cutout(cutout_name)
    output_folder /= image_name
    output_folder.mkdir(parents=True, exist_ok=True)
    output_filename /= f"{cutout_name}.jpg"

    cutout_key = f"{S3_CUTOUTS_PATH}/{group_id}/{image_name}/{cutout_name}.jpg"

    try:
        S3_CLIENT.download_file(Bucket=bucket_name, Key=cutout_key, Filename=output_filename)
        LOGGER.debug(f"Recorte {cutout_name} descargado correctamente en {output_filename}.")
    except Exception as e:
        raise Exception(f"Error al descargar el recorte {cutout_name}: {e}")

    return output_filename


@app.command()
def download_cutouts_from_s3(
    cutout_names: list[str],
    group_id: str,
    output_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
    bucket_name: str = S3_BUCKET,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        for cutout_name in tqdm(cutout_names, desc="Descargando recortes", unit="MB"):
            download_cutout_from_s3(cutout_name, group_id, output_folder, bucket_name)
        LOGGER.debug(f"Se descargaron {len(cutout_names)} recortes de S3.")
    except Exception as e:
        raise Exception(f"Error al descargar recortes de S3: {e}")


def _download_objects_by_prefix(
    prefix_key_path: str,
    output_folder: Path,
    bucket_name: str,
) -> None:
    """
    Descarga todos los objetos de S3 que coinciden con un prefijo dado.

    Args:
        prefix_key_path (str): El prefijo de los objetos a descargar.
        output_folder (Path): La carpeta local donde se guardarán los objetos.
        bucket_name (str): El nombre del bucket de S3.
        s3_client: El cliente de S3/S3 (ej., boto3.client).
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    LOGGER.debug(f"Buscando objetos con el prefijo '{prefix_key_path}' en el bucket '{bucket_name}'...")

    try:
        objects_list_response = S3_CLIENT.list_objects_v2(Bucket=bucket_name, Prefix=prefix_key_path)
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
                    S3_CLIENT.download_file(bucket_name, object_key, str(local_file_path))
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
        LOGGER.error(f"Error al listar o descargar objetos de S3: {e}")


@app.command()
def download_image_cutouts_from_s3(
    image_name: str,
    group_id: str,
    with_metadata: bool = False,
    output_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
    bucket_name: str = S3_BUCKET,
) -> Path:
    """
    Descarga recortes de una imagen (y opcionalmente metadatos) desde S3.

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
    cutouts_prefix = f"{S3_CUTOUTS_PATH}/{group_id}/{image_name}/"
    output_folder /= image_name

    _download_objects_by_prefix(cutouts_prefix, output_folder, bucket_name)

    if with_metadata:
        metadata_prefix = f"{S3_CUTOUTS_METADATA_PATH}/{group_id}/{image_name}/"
        cutouts_metadata_folder = DOWNLOAD_CUTOUTS_METADATA_FOLDER / image_name
        _download_objects_by_prefix(metadata_prefix, cutouts_metadata_folder, bucket_name)

    return output_folder


@app.command()
def download_images_cutouts_from_minio(
    image_names: list[str],
    group_id: str,
    with_metadata: bool = False,
    output_base_folder: Path = DOWNLOAD_CUTOUTS_FOLDER,
    bucket_name: str = S3_BUCKET,
) -> None:
    """
    Descarga recortes (y opcionalmente metadatos) para una lista de imágenes desde S3.

    Args:
        image_names (list[str]): Una lista de nombres de imágenes para las cuales se desean descargar los recortes.
        with_metadata (bool, optional): Indica si también se deben descargar los metadatos. Defaults to False.
        output_base_folder (Path, optional): La carpeta base donde se guardarán todos los recortes.
                                           Si es None, se usará DOWNLOAD_CUTOUTS_FOLDER.
    """
    output_base_folder.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Iniciando la descarga de recortes para {len(image_names)} imágenes en '{output_base_folder}'.")

    with tqdm(total=len(image_names), desc="Procesando imágenes", unit="imagen") as pbar_images:
        for image_name in image_names:
            pbar_images.set_description(f"Procesando imagen: {image_name}")

            # Carpeta específica para los recortes de esta imagen
            image_cutouts_folder = output_base_folder / image_name
            image_cutouts_folder.mkdir(parents=True, exist_ok=True)

            cutouts_prefix = f"{S3_CUTOUTS_PATH}/{group_id}/{image_name}/"
            _download_objects_by_prefix(cutouts_prefix, image_cutouts_folder, bucket_name)

            if with_metadata:
                metadata_prefix = f"{S3_CUTOUTS_METADATA_PATH}/{group_id}/{image_name}/"
                image_metadata_folder = DOWNLOAD_CUTOUTS_METADATA_FOLDER / image_name
                image_metadata_folder.mkdir(parents=True, exist_ok=True)
                _download_objects_by_prefix(metadata_prefix, image_metadata_folder, bucket_name)

            pbar_images.update(1)  # Actualiza la barra de progreso de imágenes

    LOGGER.info("Descarga de recortes para todas las imágenes completada.")


@app.command()
def upload_image_to_s3(
    image_file: BufferedReader,
    image_filename: str,
    group_id: str,
    bucket_name: str = S3_BUCKET,
) -> None:
    """
    Subes una imagen a un bucket de S3.

    Args:
        image_file (BufferedReader): Archivo de imagen que se desea subir.
        image_filename (str): Nombre del archivo de imagen que se subirá. Con extensión.
        group_id (str): Identificador del grupo al que pertenece la imagen.
        bucket_name (str, optional): Nombre del bucket de S3 donde se subirá la imagen.
            Por defecto, se utiliza el valor de la constante S3_BUCKET.
    Raises:
        Exception: Si ocurre un error al subir la imagen al bucket de S3.
    """
    key = f"{S3_IMAGE_PATH}/{group_id}/{image_filename}"
    try:
        S3_CLIENT.upload_fileobj(image_file, bucket_name, key)
        LOGGER.debug(f"Imagen {image_filename} subida correctamente a {bucket_name}.")
    except Exception as e:
        raise Exception(f"Error al subir la imagen {image_filename} a S3: {e}")


@app.command()
def upload_jgw_to_s3(
    jgw_file: BufferedReader,
    jgw_filename: str,
    group_id: str,
    bucket_name: str = S3_BUCKET,
) -> None:
    """
    Subes un archivo JGW a un bucket de S3.

    Args:
        jgw_file (BufferedReader): Archivo JGW que se desea subir.
        jgw_filename (str): Nombre del archivo JGW que se subirá. Con extensión.
        group_id (str): Identificador del grupo al que pertenece el archivo JGW.
        bucket_name (str, optional): Nombre del bucket de S3 donde se subirá el archivo JGW.
            Por defecto, se utiliza el valor de la constante S3_BUCKET.
    Raises:
        Exception: Si ocurre un error al subir el archivo JGW al bucket de S3.
    """
    key = f"{S3_JGW_PATH}/{group_id}/{jgw_filename}"
    try:
        S3_CLIENT.upload_fileobj(jgw_file, bucket_name, key)
        LOGGER.debug(f"Archivo JGW {jgw_filename} subido correctamente a {bucket_name}.")
    except Exception as e:
        raise Exception(f"Error al subir el archivo JGW {jgw_filename} a S3: {e}")


@app.command()
def upload_patch_to_s3(
    patch_file: BufferedReader,
    download_id: str,
    patch_filename: str,
    group_id: str,
    bucket_name: str = S3_BUCKET,
) -> None:

    key = f"{S3_PATCHES_PATH}/{group_id}/{download_id}/{patch_filename}"
    try:
        S3_CLIENT.upload_fileobj(patch_file, bucket_name, key)
        LOGGER.debug(f"Parche {patch_filename} subido correctamente a {bucket_name}.")
    except Exception as e:
        raise Exception(f"Error al subir el parche {patch_filename} a S3: {e}")


if __name__ == "__main__":
    app()
