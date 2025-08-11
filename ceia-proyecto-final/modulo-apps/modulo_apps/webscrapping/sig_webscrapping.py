from datetime import datetime
from pathlib import Path
from typing import Optional
import os, re, zipfile, requests, shutil
from bs4 import BeautifulSoup
from tqdm import tqdm
from modulo_apps.config import config as CONFIG
from loguru import logger as LOGGER

from modulo_apps.database_comunication.mongodb_client import mongodb as DB

import modulo_apps.s3_comunication.procesador_s3 as ProcesadorS3
import modulo_apps.utils.helpers as Helpers

import cv2

import typer

from modulo_apps.utils.types import DownloadFileMetadata, JGWData, Patch

# Configuracion
DOWNLOAD_FOLDER = CONFIG.folders.download_folder
PATCHES_FOLDER = CONFIG.folders.download_patches_folder
ZIP_FOLDER = CONFIG.folders.download_zip_folder
EXCTRACT_FOLDER = CONFIG.folders.download_extract_folder


TILE_SIZE = (4096, 4096)
OVER_LAP = 400
PURGE_WHITE_IMAGES = True
DEFAULT_THRESHOLD_PERCENT = 60
DEFAULT_WHITE_THRESHOLD = 200
S3_BUCKET_IMAGES_PATH = CONFIG.minio.paths.images
S3_BUCKET_PATCHES_PATH = CONFIG.minio.paths.patches
S3_BUCKET_METADATA_PATH = CONFIG.minio.paths.metadata
S3_BUCKET = CONFIG.minio.bucket

# Constantes
URL_MAIN_PAGE = "https://gis.montevideo.gub.uy/pmapper/map.phtml?&config=default&me=548000,6130000,596000,6162000"
URL_TOC = "https://intgis.montevideo.gub.uy/pmapper/incphp/xajax/x_toc.php?"
URL_GENERATE_DRON_ZIP = "https://intgis.montevideo.gub.uy/sit/php/common/datos/generar_zip2.php?nom_jpg=/inetpub/wwwroot/sit/mapserv/data/fotos_dron/{id}&tipo=jpg"
URL_GENERATE_FOTOS2024_ZIP = "https://intgis.montevideo.gub.uy/sit/php/common/datos/generar_zip2.php?nom_jpg=/inetpub/wwwroot/sit/mapserv/data/fotos_2024/{id}&tipo=jpg"
URL_DOWNLOAD_ZIP = "https://intgis.montevideo.gub.uy/sit/tmp/{id}.zip"
URL_JS = "https://intgis.montevideo.gub.uy/pmapper/config/default/custom.js"

HEADERS_COMMON = {
    "User-Agent": "Mozilla/5.0",
}

HEADERS_TOC = {
    "User-Agent": "Mozilla/5.0",
    "Referer": URL_MAIN_PAGE,
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/x-www-form-urlencoded",
}

BODY_TOC = {"dummy": "dummy"}

app = typer.Typer()


def _get_url(url: str, headers: dict) -> str:
    """
    Realiza una solicitud HTTP GET a la URL especificada y devuelve el contenido de la respuesta como texto.

    Args:
        url (str): La URL a la que se realizará la solicitud.
        headers (dict): Un diccionario con los encabezados HTTP que se incluirán en la solicitud.
    Returns:
        str: El contenido de la respuesta HTTP en formato de texto.
    Raises:
        Exception: Si ocurre un error HTTP al realizar la solicitud, se lanza una excepción con un mensaje descriptivo.
    """

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as err:
        raise Exception(f"Error al acceder a {url}. Razón: {err}")


def _scrap_toc_as_html() -> str:
    """
    Scrapea la página principal de la aplicación web para obtener el HTML de la Table of Contents (TOC).

    Inicia una sesión, realiza una petición GET a la página principal y posteriormente
    una petición POST para obtener el HTML que contiene la Table of Contents (TOC).

    La table of contents tiene toda la información de las capas. Es la tabla que
    aparece en el panel izquierdo de la aplicación web.

    Devuelve este HTML (tabla TOC) a la fecha de 2025-08-01:

    <div id='buscarCapa' style='padding-left:3px;'>
    ...
    </div>
    <br />
    <ul>
        <li id="licat_Actualización Cartográfica" class="toccat">
            ...
        </li>
        ...
        <li id="licat_Fotos Aéreas y MDS" class="toccat">
            <span class="vis cat-label" id="spxg_Fotos Aéreas y MDS">Fotos Aéreas y MDS</span>
            <ul>
                <li id="ligrp_grillaFotosDron" class="tocgrp" name="Grilla de Fotos DRON">
                    ...
                </li>
                ...
            </ul>
        </li>
        <li id="licat_zonas" class="toccat">
            ...
        </li>
        ...
    </ul>

    Returns:
        str: El contenido HTML que representa la estructura TOC de la página.

    Raises:
        Exception: Si se produce un error HTTP (4xx o 5xx) al acceder a la URL de TOC,
                   se lanza una excepción con un mensaje indicando la razón del fallo.
    """
    session = requests.Session()

    try:
        session.get(URL_MAIN_PAGE, headers=HEADERS_COMMON)
        response = session.post(URL_TOC, headers=HEADERS_TOC, data=BODY_TOC)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as err:
        raise Exception(f"Error al acceder a {URL_TOC}. Razón: {err}")
    finally:
        session.close()


def _scrap_images_as_html(toc_html: str) -> str:
    """
    Scrapea el HTML de la sección de imágenes de drones a partir del HTML de la TOC.

    A partir del toc HTML recibido, localiza el elemento `<li>` con id 'ligrp_grillaFotosDron'
    que es donde se encuentran las imágenes de drones, y retorna el contenido HTML del `<ul>`
    padre que contiene todos los elementos (<`li>`) con las imágenes.

    Devuelve este HTML a la fecha de 2025-08-01:
    <ul>
        <li class="tocgrp" id="ligrp_grillaFotosDron" name="Grilla de Fotos DRON">
            ...
        </li>
        <li class="tocgrp" id="ligrp_fotosDron_8deOctubreyCentenario-EspLibreLarranaga_20190828">
            ...
        </li>
        <li class="tocgrp" id="ligrp_fotosDron_AntelArena_20200804" name="Antel Arena - 04/08/2020">
            <input id="ginput_fotosDron_AntelArena_20200804" name="groupscbx" type="checkbox"
                value="fotosDron_AntelArena_20200804" />
            <span class="vis" id="spxg_fotosDron_AntelArena_20200804">
                <span class="grp-title" title="Antel Arena - 04/08/2020">Antel Arena - 04/08/2020</span>
            </span>
        </li>
        ...
        <li class="tocgrp" id="ligrp_fotos2024" name="Fotos aereas 2024">
            ...
        </li>
        ...
        <li class="tocgrp" id="ligrp_Descarga_fotos_2024" name="Descarga Fotos 2024">
            ...
        </li>
        ...
    </ul>

    Args:
        html (str): Cadena que representa el contenido HTML de la página.

    Returns:
        str: El contenido HTML del elemento `<ul>` que contiene las imágenes de drones.

    Raises:
        ValueError: Si no se encuentra el `<li>` con el id especificado o si no existe
                    un `<ul>` padre que lo contenga.
    """

    li_grilla_fotos_dron = "ligrp_grillaFotosDron"
    soup = BeautifulSoup(toc_html, "html.parser")

    # 1) Buscamos el <li> deseado
    li_drones_pictures = soup.find("li", {"id": li_grilla_fotos_dron})
    if li_drones_pictures is None:
        raise ValueError("No se encontró el <li> con id='{li_grilla_fotos_dron}' en el HTML.")

    # 2) Obtenemos el <ul> que lo contiene
    ul_drones_pictures = li_drones_pictures.find_parent("ul")
    if ul_drones_pictures is None:
        raise ValueError("No se encontró un elemento <ul> padre para el <li> con id='{li_grilla_fotos_dron}'.")

    # Devolvemos el HTML del <ul> que contiene las imágenes
    return str(ul_drones_pictures)


def _build_image_info_dict(images_html: str) -> list[DownloadFileMetadata]:
    """
    Construye un diccionario con información de las imágenes a partir del HTML proporcionado.

    Procesa el images HTML para obtener información sobre cada elemento que
    contiene imágenes de drones. Busca spans con IDs que coincidan con la expresión
    regular `.*fotosDron.*` y extrae un identificador, así como el atributo `title`.

    Procesa elementos <li> con el siguiente HTML:
    <li class="tocgrp" id="ligrp_fotosDron_AntelArena_20200804" name="Antel Arena - 04/08/2020">
        <input id="ginput_fotosDron_AntelArena_20200804" name="groupscbx" type="checkbox"
            value="fotosDron_AntelArena_20200804" />
        <span class="vis" id="spxg_fotosDron_AntelArena_20200804">
            <span class="grp-title" title="Antel Arena - 04/08/2020">Antel Arena - 04/08/2020</span>
        </span>
    </li>

    Args:
        html (str): Contenido HTML que representa la lista de imágenes (un `<ul>`).

    Returns:
        dict: Un diccionario donde las claves son los identificadores de las imágenes
              y los valores son los atributos `title` correspondientes. Esto se utiliza para luego
              descargar las imágenes que están en un Javascript.

    Raises:
        (No lanza excepciones explícitas propias, pero podría propagar excepciones de BeautifulSoup.)
    """

    soup = BeautifulSoup(images_html, "html.parser")

    spans_fotos_dron = soup.find_all("span", id=re.compile(r".*fotosDron.*"))

    resultados = []

    for idx, sp in enumerate(spans_fotos_dron, start=1):
        # 1) Obtener el nombre a partir del id del span
        span_id = sp.get("id", "")
        value_attr = span_id.replace("spxg_", "")  # Obtengo ej: fotosDron_AntelArena_20200804

        # 2) Obtener el atributo title del span hijo
        span_child_element = sp.find("span")
        if not span_child_element:
            # Si no lo encuentra, pasa al siguiente
            LOGGER.warning(f"No se encontró <span class='grp-title'> dentro del span con id='{span_id}'.")
            continue
        title_attr = span_child_element.get("title")

        # 3) Obtengo la fecha mediante expresión regular del título
        date_format = "%d/%m/%Y"
        match = re.search(r"\d{2}/\d{2}/\d{4}", title_attr)
        date_str = match.group(0) if match else None
        if not date_str:
            LOGGER.warning(f"No se encontró la fecha en el atributo title='{title_attr}'.")
            continue
        date_captured = datetime.strptime(date_str, date_format)

        resultados.append(
            DownloadFileMetadata(
                js_name=value_attr,
                title=title_attr,
                index=idx,
                date_captured=date_captured,
            )
        )

    return resultados


def _download_zip_file(download_id: str, url_generate_zip: str) -> Path:
    """
    Dada una cadena `id`, realiza un get para generar un archivo ZIP en el servidor y luego
    lo descarga si existe. El ZIP contiene las imágenes correspondientes al identificador.

    Args:
        id (str): Identificador que se usará para generar y descargar el ZIP.

    Returns:
        str | None: La ruta absoluta del archivo ZIP descargado. Retorna None si no se
                    puede generar o descargar el ZIP.

    Raises:
        (Las excepciones de requests se manejan internamente, produciendo logs de error
         o warning en caso de fallos. No se relanza la excepción.)
    """
    # Ruta para descargar el archivo ZIP
    url_download_zip = URL_DOWNLOAD_ZIP.format(id=download_id)

    try:
        response = requests.get(url_generate_zip, headers=HEADERS_COMMON)
        response.raise_for_status()
        LOGGER.debug(f"Archivo ZIP generado correctamente en el servidor. URL: {url_generate_zip}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"RequestException, no se pudo generar el ZIP. URL: {url_generate_zip} - {e}")
    finally:
        try:
            response = requests.get(url_download_zip, headers=HEADERS_COMMON)
            response.raise_for_status()
            LOGGER.debug(f"Archivo ZIP descargado correctamente. URL: {url_download_zip}")

            file_name = f"{download_id}.zip"
            ZIP_FOLDER.mkdir(parents=True, exist_ok=True)

            file_path = ZIP_FOLDER / file_name

            with open(file_path, "wb") as f:
                f.write(response.content)
            LOGGER.debug(f"Archivo '{file_name}' descargado correctamente en {file_path}")

            return file_path
        except requests.exceptions.RequestException as e:
            raise Exception(f"No se pudo descargar el ZIP. URL: {url_download_zip} - {e}")


def _scrap_javascript_code_as_text() -> str:
    """
    Scrapea el contenido de un archivo JavaScript que contiene información sobre las imágenes.

    Returns:
        str: El contenido del archivo JavaScript en formato de texto.
    """
    return _get_url(URL_JS, HEADERS_COMMON)


def _build_image_mapping_dict(js_code_text: str) -> dict:
    """
    Construye un diccionario a partir del código JavaScript que contiene información sobre las imágenes.

    Parsea el código JavaScript para encontrar las líneas que contengan casos y sus file_descarga.

    Toma el contenido del archivo JavaScript y busca patrones específicos para ubicar un case
    con un valor y una asignación a `file_descarga`. Por ejemplo:
    case 'fotosDron_AntelArena_20200804':
        file_descarga = 'fotos_dron/AntelArena_20200804';

    Devuelve un diccionario donde las claves son los valores del case:
    {
        ...,
        "grillaFotosDron": "fotosDron_2017-2024",
        "fotosDron_8deOctubreyCentenario-EspLibreLarranaga_20190828": "fotos_dron/8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm",
        "fotosDron_Rotonda_Arocena_20240927": "fotos_dron/Rotonda_Arocena_20240927_dji_rtk_5cm",
        "fotosDron_AcostayLara_20240927": "fotos_dron/AcostayLara_20240927_dji_rtk_5cm",
        ...
    }

    Args:
        js_code (str): El contenido del archivo JavaScript.
    Returns:
        dict: Diccionario con los casos y sus rutas de archivo correspondientes.
    """
    pattern = r"""case\s+'([^']+)':\s*
                  (?:[^\n]*\n)?          # Captura opcional cualquier cosa hasta el fin de línea
                  \s*file_descarga\s*=\s*'([^']+)'\s*;"""

    # re.VERBOSE permite escribir la regex más legible con comentarios
    # re.MULTILINE permite que ^ y $ coincidan con principio y fin de línea
    # re.DOTALL hace que . coincida también con saltos de línea
    matches = re.findall(pattern, js_code_text, re.VERBOSE | re.MULTILINE | re.DOTALL)

    return {case_val: file_descarga for case_val, file_descarga in matches}


def _add_file_download_id_to_results(
    pending_download_files_metadata: list[DownloadFileMetadata], image_id_to_download_id_map: dict
) -> list[DownloadFileMetadata]:
    """
    Agrega el identificador de descarga de archivo (`file_download_id`) a las entradas de metadatos de archivos pendientes.
    Este método toma una lista de metadatos de archivos pendientes y un diccionario que mapea identificadores de imágenes
    a identificadores de descarga. Si el identificador de imagen (`js_name`) de una entrada de metadatos está presente
    en el diccionario, se asigna el identificador de descarga correspondiente a la entrada de metadatos. Si no se encuentra
    un valor en el mapeo, se registra una advertencia.
    Args:
        pending_download_files_metadata (list[DownloadFileMetadata]):
            Lista de objetos `DownloadFileMetadata` que representan los archivos pendientes de descarga.
        image_id_to_download_id_map (dict):
            Diccionario que mapea identificadores de imágenes (`js_name`) a identificadores de descarga.
    Returns:
        list[DownloadFileMetadata]:
            La lista actualizada de metadatos de archivos pendientes con los identificadores de descarga asignados.
    """
    for file_metadata_entry in pending_download_files_metadata:
        value_attr = file_metadata_entry.js_name
        if value_attr in image_id_to_download_id_map:
            desc = image_id_to_download_id_map[value_attr]
            file_download_id = desc.removeprefix("fotos_dron/")
            file_metadata_entry.file_download_id = file_download_id
            file_metadata_entry.url_generate_zip = URL_GENERATE_DRON_ZIP.format(id=file_download_id)
        else:
            LOGGER.warning(f"No se encontró un valor para '{value_attr}' en el mapping. No se descargará.")
    return pending_download_files_metadata


def _extract_files(zip_path: Path, download_id: str = None) -> tuple[Path, Path]:
    """
    Extrae los archivos JPG y JGW de un archivo ZIP descargado.

    Args:
        zip_path (str): Ruta absoluta del archivo ZIP descargado.

    Returns:
        tuple[str, str]: Tupla con las rutas absolutas de los archivos JPG y JGW extraídos.
                         Si no se pudo extraer alguno de los archivos, se devuelve None.

    Raises:
        Exception: Si el ZIP no contiene exactamente dos archivos, se lanza una excepción.
    """
    EXCTRACT_FOLDER.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(EXCTRACT_FOLDER)

        # Encontrar los archivos JPG y JGW extraídos
        jpg_file = None
        jgw_file = None

        # Obtengo los archivos extraídos
        jpg_file = EXCTRACT_FOLDER / f"{download_id}.jpg"
        jgw_file = EXCTRACT_FOLDER / f"{download_id}.jgw"

        # Chequeo que existan los archivos
        if not (jpg_file.exists() and jgw_file.exists()):
            raise Exception(f"El ZIP no contiene los archivos esperados: {zip_path}")
        # Chequeo el tamaño de los archivos
        if jpg_file.stat().st_size == 0 or jgw_file.stat().st_size == 0:
            raise Exception(f"Los archivos extraídos están vacíos: {jpg_file}, {jgw_file}")

        LOGGER.debug(f"Archivo ZIP extraído de {download_id} correctamente. JPG: {jpg_file}, JGW: {jgw_file}")
        return jpg_file, jgw_file

    except Exception as e:
        raise Exception(f"Error al extraer archivos del ZIP {zip_path}: {str(e)}")


def _upload_img_to_s3(jpg_path: Path, jgw_path: Path, group_id: str, download_id: str) -> None:
    try:
        # Upload JPG file
        with open(jpg_path, "rb") as jpg_file:
            ProcesadorS3.upload_image_to_s3(jpg_file, f"{download_id}.jpg", group_id)

        # Upload JGW file
        with open(jgw_path, "rb") as jgw_file:
            ProcesadorS3.upload_jgw_to_s3(jgw_file, f"{download_id}.jgw", group_id)
        LOGGER.debug(f"Files uploaded to S3: {download_id}")
    except Exception as e:
        raise Exception(f"Error uploading files to S3 for {download_id}: {str(e)}")


def _read_jgw_data(jgw_path: Path) -> JGWData:
    """
    Lee un archivo JGW (World File) y extrae los parámetros de georreferenciación.
    Un archivo JGW contiene 6 líneas con los parámetros de transformación geoespacial
    que permiten convertir coordenadas de píxeles a coordenadas del mundo real.
    Args:
        jgw_path (Path): Ruta al archivo JGW que se va a leer.
    Returns:
        JGWData: Objeto que contiene los parámetros de georreferenciación extraídos
                del archivo JGW, incluyendo tamaño de píxel, rotaciones y origen.
    Raises:
        Exception: Si ocurre un error al leer el archivo o al procesar los datos,
                  incluyendo problemas de formato o archivo no encontrado.
    Note:
        El archivo JGW debe contener exactamente 6 líneas con valores numéricos
        en el siguiente orden:
        1. Tamaño de píxel en X
        2. Rotación en Y
        3. Rotación en X
        4. Tamaño de píxel en Y (generalmente negativo)
        5. Coordenada X del origen
        6. Coordenada Y del origen
    """
    try:
        with open(jgw_path, "r") as jgw_file:
            lines = jgw_file.readlines()
            data = {
                "x_pixel_size": float(lines[0].strip()),
                "y_rotation": float(lines[1].strip()),
                "x_rotation": float(lines[2].strip()),
                "y_pixel_size": float(lines[3].strip()),
                "x_origin": float(lines[4].strip()),
                "y_origin": float(lines[5].strip()),
            }
            LOGGER.debug(f"JGW file read successfully: {jgw_path}")
            return JGWData(**data)
    except Exception as e:
        raise Exception(f"Error reading JGW file {jgw_path}: {str(e)}")


def remove_directorys() -> None:
    """
    Elimina directorios específicos utilizados en el proceso de web scrapping.
    Elimina los directorios PATCHES_FOLDER, EXCTRACT_FOLDER y ZIP_FOLDER,
    ignorando errores si no existen. También elimina el directorio DOWNLOAD_FOLDER
    si está vacío (no contiene archivos ni subdirectorios).
    Returns:
        None: Esta función no retorna ningún valor.
    Note:
        - Los directorios se eliminan usando shutil.rmtree con ignore_errors=True
        - El directorio de descarga solo se elimina si está completamente vacío
        - Se registra un mensaje de debug cuando se elimina el directorio de descarga vacío
    """

    shutil.rmtree(PATCHES_FOLDER, ignore_errors=True)
    shutil.rmtree(EXCTRACT_FOLDER, ignore_errors=True)
    shutil.rmtree(ZIP_FOLDER, ignore_errors=True)

    # Delete download folder if it has no files or directories
    if not any(os.scandir(DOWNLOAD_FOLDER)):
        os.rmdir(DOWNLOAD_FOLDER)
        LOGGER.debug(f"Removed empty download folder: {DOWNLOAD_FOLDER}")


def _split_image_with_overlap(
    download_id: str,
    jpg_path: str,
    tile_size: tuple[int, int],
    overlap: int,
    output_dir: Path,
    purge_white_images: bool,
    threshold_percent: int,
    white_threshold: int,
) -> list[Patch]:
    """
    Divide una imagen en parches rectangulares con superposición y opcionalmente filtra parches blancos.
    Esta función toma una imagen y la divide en múltiples parches más pequeños con un tamaño específico
    y una superposición definida entre parches adyacentes. Opcionalmente puede filtrar parches que
    contengan principalmente píxeles blancos.
    Args:
        download_id (str): Identificador único para el conjunto de parches generados.
        jpg_path (str): Ruta al archivo de imagen JPG que se va a dividir.
        tile_size (tuple[int, int]): Tupla que especifica el ancho y alto de cada parche en píxeles.
        overlap (int): Número de píxeles de superposición entre parches adyacentes.
        output_dir (Path): Directorio donde se guardarán los parches generados.
        purge_white_images (bool): Si es True, no guarda parches que sean predominantemente blancos.
        threshold_percent (int): Porcentaje mínimo de píxeles blancos para considerar un parche como "blanco".
        white_threshold (int): Valor de umbral para determinar si un píxel se considera blanco (0-255).
    Returns:
        list[Patch]: Lista de objetos Patch con metadatos de cada parche generado, incluyendo
                    coordenadas, dimensiones, nombre del archivo y estado de filtrado.
    Raises:
        Exception: Si no se puede cargar la imagen del archivo especificado o si el formato es inválido.
    """
    # Cargar la imagen
    try:
        image = cv2.imread(jpg_path)
    except Exception as e:
        raise Exception(f"Error loading image {jpg_path}: {str(e)}")
    if image is None:
        raise Exception(f"Error loading image {jpg_path}: Image not found or invalid format.")
    height, width = image.shape[:2]

    # Crear el directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    patches_metadata = []
    patch_id = 1

    # Recorrer la imagen en bloques con superposición
    for y in range(0, height, tile_size[1] - overlap):
        for x in range(0, width, tile_size[0] - overlap):
            # Definir las coordenadas del recorte
            x_end = min(x + tile_size[0], width)
            y_end = min(y + tile_size[1], height)

            # Recortar la región
            tile = image[y:y_end, x:x_end]

            tile_name = f"{download_id}_patch_{patch_id}"

            # Chequear si la mayoría de la imagen es blanca.
            img_is_white = Helpers.is_white_image(tile, threshold_percent, white_threshold)[0]

            # Si la mayoría es blanca, y está activada la opción, no guardar la imagen.
            patch_filename = output_dir / f"{tile_name}.jpg"
            if purge_white_images:
                if not img_is_white:
                    cv2.imwrite(patch_filename, tile)
            else:
                cv2.imwrite(patch_filename, tile)

            patches_metadata.append(
                Patch(
                    patch_id=patch_id,
                    patch_name=tile_name,
                    x_start=x,
                    y_start=y,
                    x_end=x_end,
                    y_end=y_end,
                    width=x_end - x,
                    height=y_end - y,
                    is_white=img_is_white,
                    white_threeshold_percent=threshold_percent,
                    white_threeshold_value=white_threshold,
                )
            )

            patch_id += 1

    return patches_metadata


def _upload_patches_to_s3(download_id: str, group_id: str, patches_dir: Path) -> None:
    """
    Sube todos los patches de un directorio específico a S3.
    Esta función itera a través de todos los archivos en el directorio de patches
    especificado y los sube a S3 utilizando el ProcesadorS3. Cada archivo se
    sube individualmente con los metadatos correspondientes.
    Args:
        download_id (str): Identificador único de la descarga asociada a los patches.
        group_id (str): Identificador del grupo al que pertenecen los patches.
        patches_dir (Path): Ruta del directorio que contiene los archivos de patches
                           a subir.
    Returns:
        None
    Raises:
        Exception: Si ocurre algún error durante el proceso de subida de los patches
                   a S3, se lanza una excepción con detalles del error.
    Note:
        La función registra un mensaje de debug cuando todos los patches se han
        subido exitosamente.
    """
    try:
        # Listar los archivos en el directorio
        for patch_filename in os.listdir(patches_dir):
            patch_path = patches_dir / patch_filename
            with open(patch_path, "rb") as patch_file:
                ProcesadorS3.upload_patch_to_s3(patch_file, download_id, patch_filename, group_id)
        LOGGER.debug(f"Patches uploaded to S3: {download_id}")
    except Exception as e:
        raise Exception(f"Error uploading patches to S3 for {download_id}: {str(e)}")


def _update_DB_patches(download_file_metadata: DownloadFileMetadata) -> None:
    """
    Agrega metadatos de los parches a DB.
    """
    try:
        image_collection = DB.get_collection("imagenes")
        image_collection.update_one(
            {"file_download_id": download_file_metadata.file_download_id},
            {"$set": {"patches": [patch.__dict__ for patch in download_file_metadata.patches], "has_patches": True}},
        )
        LOGGER.debug(f"DB updated with patches for {download_file_metadata.file_download_id}")
    except Exception as e:
        raise Exception(f"Error updating DB with patches: {str(e)}")


def _is_image_downloaded(download_id):
    """
    Check if the file with the given download_id has already been downloaded.

    Args:
        download_id (str): The ID of the file to check.

    Returns:
        bool: True if the file is already downloaded, False otherwise.
    """
    image_collection = DB.get_collection("imagenes")
    is_downloaded = image_collection.find_one({"file_download_id": download_id})
    return is_downloaded is not None


def _update_jgw_data(download_file_metadata: DownloadFileMetadata, jgw_path: Path) -> DownloadFileMetadata:
    """
    Actualiza los metadatos de descarga con los datos del archivo JGW.
    Lee los datos de georreferenciación desde un archivo JGW y los asigna a los metadatos
    del archivo de descarga proporcionado.
    Args:
        download_file_metadata (DownloadFileMetadata): Los metadatos del archivo de descarga
            que serán actualizados con los datos JGW.
        jgw_path (Path): La ruta al archivo JGW que contiene los datos de georreferenciación.
    Returns:
        DownloadFileMetadata: Los metadatos del archivo de descarga actualizados con los
            datos JGW leídos desde el archivo especificado.
    """

    jgw_data = _read_jgw_data(jgw_path)
    download_file_metadata.jgw_data = jgw_data
    return download_file_metadata


def _update_width_height(download_file_metadata: DownloadFileMetadata, jpg_path: Path) -> DownloadFileMetadata:
    """
    Actualiza las dimensiones de ancho y alto en los metadatos de un archivo descargado.
    Lee una imagen JPEG desde la ruta especificada utilizando OpenCV y extrae sus
    dimensiones para actualizar los campos width y height del objeto de metadatos.
    Args:
        download_file_metadata (DownloadFileMetadata): Objeto de metadatos del archivo
            descargado que será actualizado con las nuevas dimensiones.
        jpg_path (Path): Ruta al archivo de imagen JPEG del cual se extraerán
            las dimensiones.
    Returns:
        DownloadFileMetadata: El objeto de metadatos actualizado con los valores
            de ancho y alto de la imagen.
    Raises:
        cv2.error: Si la imagen no puede ser leída o el archivo no existe.
    """
    try:
        img = cv2.imread(jpg_path)
    except cv2.error as e:
        raise cv2.error(f"Error reading image at {jpg_path}: {str(e)}")
    height, width = img.shape[:2]
    download_file_metadata.width = width
    download_file_metadata.height = height
    return download_file_metadata


def _insert_into_DB(download_file_metadata: DownloadFileMetadata) -> None:
    """
    Inserta los metadatos de un archivo descargado en la base de datos.
    Esta función toma un objeto DownloadFileMetadata y lo convierte en un
    diccionario para luego insertarlo en la colección 'imagenes' de la base de datos.
    Args:
        download_file_metadata (DownloadFileMetadata): Objeto que contiene los
            metadatos del archivo descargado a ser insertado en la base de datos.
    Returns:
        None
    Raises:
        Exception: Si ocurre un error durante la inserción en la base de datos.
    """
    try:
        image_collection = DB.get_collection("imagenes")
        doc = download_file_metadata.model_dump()
        doc["image_name"] = download_file_metadata.image_name
        image_collection.insert_one(doc)
        LOGGER.debug(f"Inserted into DB: {download_file_metadata.file_download_id}")
    except Exception as e:
        raise Exception(f"Error inserting into DB: {str(e)}")


def _process_and_upload_image(
    download_file_metadata: DownloadFileMetadata,
    group_id: str,
    apply_image_splitting: bool = True,
) -> None:
    """
    Procesa y sube una imagen satelital desde un archivo ZIP descargado, con opciones para división de imágenes.
    Esta función ejecuta un pipeline completo de procesamiento de imágenes satelitales que incluye:
    1. Verificación de descarga previa
    2. Descarga del archivo ZIP desde el servidor
    3. Extracción de archivos JPG y JGW
    4. Actualización de metadatos (dimensiones, datos geoespaciales)
    5. Subida a S3
    6. Inserción en base de datos
    7. División opcional de imágenes en patches
    Args:
        download_file_metadata (DownloadFileMetadata): Metadatos del archivo a procesar,
            incluyendo información de descarga y propiedades de la imagen.
        group_id (str): Identificador del grupo al cual pertenece la imagen para
            organización en S3.
        apply_image_splitting (bool, optional): Si True, divide la imagen en patches
            más pequeños con solapamiento. Por defecto True.
    Returns:
        None: La función no retorna valores, pero modifica el estado del sistema
        mediante subidas a S3 e inserciones en base de datos.
    Raises:
        Exception: Puede lanzar excepciones relacionadas con:
            - Errores de descarga del archivo ZIP
            - Problemas de extracción de archivos
            - Fallos en la subida a S3
            - Errores de conexión a base de datos
    Note:
        - Si la imagen ya fue descargada previamente, la función termina tempranamente
        - Los patches solo se generan si apply_image_splitting=True y se crean patches válidos
        - La función utiliza variables globales para configuración (TILE_SIZE, OVER_LAP, etc.)
        - Los logs se generan en cada paso del proceso para facilitar el debugging
    """
    LOGGER.debug(f"Starting processing for download_id: {download_file_metadata.file_download_id}")
    download_id = download_file_metadata.file_download_id
    if _is_image_downloaded(download_id):
        LOGGER.debug(f"File {download_id} already downloaded. Skipping processing.")
        return

    LOGGER.debug(f"Processing download_id: {download_id}")

    # Download the ZIP file
    zip_path = _download_zip_file(download_id, download_file_metadata.url_generate_zip)
    LOGGER.debug(f"ZIP downloaded for {download_id}")

    # Update download metadata
    download_file_metadata.downloaded_date = datetime.now()
    download_file_metadata.group_id = group_id

    # Extract files
    paths = _extract_files(zip_path, download_id)
    jpg_path, jgw_path = paths
    LOGGER.debug(f"Files extracted for {download_id}")

    download_file_metadata = _update_width_height(download_file_metadata, jpg_path)
    download_file_metadata = _update_jgw_data(download_file_metadata, jgw_path)

    # Upload to S3
    _upload_img_to_s3(jpg_path, jgw_path, group_id, download_id)
    LOGGER.debug(f"Files uploaded to S3 for {download_id}")

    _insert_into_DB(download_file_metadata)
    LOGGER.debug(f"DB inserted for {download_id}")
    if apply_image_splitting:
        LOGGER.debug(f"Image split for {download_id}")
        image_patches_output_dir = PATCHES_FOLDER / download_id
        patches = _split_image_with_overlap(
            download_id,
            jpg_path,
            TILE_SIZE,
            OVER_LAP,
            image_patches_output_dir,
            PURGE_WHITE_IMAGES,
            DEFAULT_THRESHOLD_PERCENT,
            DEFAULT_WHITE_THRESHOLD,
        )

        if not patches:
            LOGGER.warning(f"No patches created for {download_id}. Skipping patch upload.")
        else:
            _upload_patches_to_s3(download_id, group_id, image_patches_output_dir)
            LOGGER.debug(f"Patches uploaded to S3 for {download_id}")

            download_file_metadata.patches = patches

            _update_DB_patches(download_file_metadata)
            LOGGER.debug(f"DB updated with patches for {download_id}")


def scrap_drones_images() -> list[DownloadFileMetadata]:
    """
    Extrae metadatos de imágenes de drones desde la página web del SIG.
    Esta función realiza web scraping para obtener información sobre imágenes de drones
    disponibles para descarga. Procesa el contenido HTML de la tabla de contenidos,
    extrae información de las imágenes, obtiene los IDs de descarga desde el código
    JavaScript y combina toda la información para generar metadatos completos de los
    archivos listos para descarga.
    Returns:
        list[DownloadFileMetadata]: Lista de metadatos de archivos que contiene la
            información necesaria para descargar las imágenes de drones, incluyendo
            IDs de descarga, URLs y otros metadatos relevantes.
    Raises:
        Exception: Si ocurre un error durante el proceso de scraping o procesamiento
            de los datos obtenidos.
    """

    pending_download_files_metadata = _build_image_info_dict(_scrap_images_as_html(_scrap_toc_as_html()))
    image_id_to_download_id_map = _build_image_mapping_dict(_scrap_javascript_code_as_text())

    files_metadata_to_download = _add_file_download_id_to_results(
        pending_download_files_metadata, image_id_to_download_id_map
    )
    return files_metadata_to_download


def scrap_fotos2024_images(images_names: list) -> list[DownloadFileMetadata]:
    """
    Genera metadatos de descarga para imágenes de fotos aéreas del año 2024.
    Esta función toma una lista de nombres de imágenes y crea objetos DownloadFileMetadata
    correspondientes, añadiendo el prefijo "RGB_MVD_2024_" a cada ID de descarga y
    utilizando la fecha actual como fecha de captura.
    Args:
        images_names (list): Lista de nombres de imágenes para procesar.
    Returns:
        list[DownloadFileMetadata]: Lista de objetos DownloadFileMetadata con los
            metadatos generados para cada imagen, incluyendo ID de descarga con prefijo,
            título original y fecha de captura actual.
    """
    download_files_metadata = []
    for image_name in images_names:
        file_download_id = image_name.prefix("RGB_MVD_2024_")
        download_files_metadata.append(
            DownloadFileMetadata(
                file_download_id=file_download_id,
                title=image_name,
                date_captured=datetime.now(),
                url_generate_zip=URL_GENERATE_FOTOS2024_ZIP.format(id=file_download_id),
            )
        )
    return download_files_metadata


def processs_files(files_metadata_to_download: list[DownloadFileMetadata], group_id: str) -> None:
    """
    Procesa una lista de metadatos de archivos para descargar y subir imágenes.
    Esta función itera sobre una lista de metadatos de archivos, verificando si cada archivo
    tiene un ID de descarga válido. Para los archivos con ID válido, procesa y sube la imagen
    correspondiente. Los archivos sin ID de descarga son omitidos con un mensaje de advertencia.
    Args:
        files_metadata_to_download (list[DownloadFileMetadata]): Lista de metadatos de archivos
            que contienen la información necesaria para descargar y procesar cada archivo.
        group_id (str): Identificador del grupo al cual pertenecen los archivos a procesar.
    Returns:
        None: Esta función no retorna ningún valor.
    Notes:
        - La función registra información de progreso usando logging
        - Muestra una barra de progreso usando tqdm durante el procesamiento
        - Los archivos sin file_download_id son omitidos y se registra una advertencia
    """

    LOGGER.info(f"Processing {len(files_metadata_to_download)} files for group {group_id}")
    for file_to_download_metadata in tqdm(files_metadata_to_download, desc="Processing images"):
        if file_to_download_metadata.file_download_id:
            try:
                LOGGER.info(f"Processing download ID {file_to_download_metadata.file_download_id}")
                _process_and_upload_image(file_to_download_metadata, group_id)
            except Exception as e:
                LOGGER.error(f"Error processing {file_to_download_metadata.file_download_id}: {str(e)}")
                continue
            LOGGER.success(f"Processed {file_to_download_metadata.file_download_id} successfully.")
        else:
            LOGGER.warning(f"No download ID found for {file_to_download_metadata.js_name}. Skipping.")
