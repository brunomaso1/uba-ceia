from concurrent.futures import ProcessPoolExecutor
import io
import datetime, json, os, zipfile

from tqdm import tqdm

import kml2geojson
import requests

import pandas as pd
import geopandas as gpd
from fastkml import kml

from pathlib import Path
from typing import Any, Optional, TextIO

from shapely.geometry import Point, Polygon

import typer

from loguru import logger as LOGGER
from modulo_apps.config import config as CONFIG
from modulo_apps.database_comunication.mongodb_client import mongodb as DB

import modulo_apps.labeling.procesador_anotaciones_coco_dataset as CocoDatasetUtils
import modulo_apps.labeling.convertor_cordenadas as ConvertorCoordenadas
import modulo_apps.labeling.procesador_anotaciones_mongodb as ProcesadorAnotacionesMongoDB

DOWNLOAD_TEMP_FOLDER = CONFIG.folders.download_temp_folder
DOWNLOAD_GOOGLE_MAPS_FOLDER = CONFIG.folders.download_google_maps_folder
DOWNLOAD_KMLS_FOLDER = CONFIG.folders.download_kmls_folder
DOWNLOAD_GEOJSON_FOLDER = CONFIG.folders.download_geojson_folder

COCO_DATASET_DATA = CONFIG.coco_dataset.to_dict()
COCO_DATASET_CATEGORIES = CONFIG.coco_dataset.categories

app = typer.Typer()


@app.command()
def download_kmz_from_gmaps(
    base_url: str = CONFIG.google_maps.base_url,
    mid: str = CONFIG.google_maps.mid,
    output_filename: Optional[Path] = None,
) -> Path:
    """Descarga un archivo KMZ desde Google Maps y lo descomprime.

    Este método utiliza la configuración proporcionada para construir la URL de descarga
    del archivo KMZ desde Google Maps. Una vez descargado, el archivo se descomprime
    y se guarda en la carpeta especificada.

    Args:
        filename (str, optional): Ruta donde se guardará el archivo KMZ descargado.
                                  Si no se proporciona, se utiliza una ruta temporal.

    Returns:
        str: Ruta del archivo KML descargado y descomprimido.

    Raises:
        Exception: Si ocurre un error al acceder a la URL de descarga.
    """
    url = f"{base_url}?mid={mid}"
    if not output_filename:
        DOWNLOAD_TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
        output_filename = DOWNLOAD_TEMP_FOLDER / "google_maps.kmz"
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_filename, "wb") as f:
            f.write(response.content)
        LOGGER.debug(f"Archivo KMZ descargado y guardado en {output_filename}.")

        extract_path = (
            DOWNLOAD_GOOGLE_MAPS_FOLDER / f"google_maps_{mid}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.kmz"
        )

        # Descomprimir el archivo KMZ
        with zipfile.ZipFile(output_filename, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        LOGGER.debug(f"Archivo KMZ descomprimido en {extract_path}.")

        # Clear the temporary file
        os.remove(output_filename)

        return f"{extract_path}/doc.kml"

    except requests.exceptions.HTTPError as err:
        raise Exception(f"Error al acceder a {url}. Razón: {err}")


def convert_kml_to_geojson(
    kml_data: str,
    should_download: bool = False,
    output_filename: Path = DOWNLOAD_GEOJSON_FOLDER / "converted.geojson",
) -> gpd.GeoDataFrame:
    """
    Convierte datos en formato KML a GeoJSON y los retorna como un GeoDataFrame.

    Args:
        kml_data (str): Cadena de texto que contiene los datos en formato KML.
        should_download (bool, opcional): Indica si el archivo GeoJSON generado debe ser guardado en disco.
            Por defecto es False.
        output_filename (Path, opcional): Ruta del archivo donde se guardará el GeoJSON si `should_download` es True.
            Por defecto se guarda en `DOWNLOAD_GEOJSON_FOLDER/converted.geojson`.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame generado a partir de los datos convertidos de KML a GeoJSON.

    Raises:
        Exception: Si ocurre un error durante la conversión de KML a GeoJSON.

    Notas:
        - Utiliza la biblioteca `kml2geojson` para realizar la conversión.
        - Si `should_download` es True, el archivo GeoJSON se guarda en la ubicación especificada.
        - El GeoDataFrame se genera a partir de las características del GeoJSON convertido.
    """
    try:
        buffer = io.StringIO(kml_data)
        geojson = kml2geojson.main.convert(buffer)[0]  # https://mrcagney.github.io/kml2geojson_docs/
        LOGGER.success(f"Conversión exitosa.")
    except Exception as e:
        LOGGER.error(f"Error al convertir KML a GeoJSON: {e}")

    if should_download:
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        LOGGER.success(f"GeoJSON guardado en {output_filename}")

    return gpd.GeoDataFrame.from_features(geojson["features"])


@app.command()
def convert_kml_to_geojson_from_path(
    kml_filepath: Path, should_download: bool = None, output_filename: Path = None
) -> gpd.GeoDataFrame:
    """
    Convierte un archivo KML a GeoJSON desde una ruta de archivo.

    Args:
        kml_filepath (Path): Ruta al archivo KML que se desea convertir.
                             Debe ser un objeto de tipo `Path`.
        should_download (bool, opcional): Indica si el archivo GeoJSON resultante
                                          debe ser descargado. Por defecto es `None`.
        output_filename (Path, opcional): Ruta y nombre del archivo GeoJSON de salida.
                                          Si no se especifica, se utiliza un nombre predeterminado.

    Returns:
        gpd.GeoDataFrame: Un GeoDataFrame que contiene los datos convertidos del archivo KML.

    Raises:
        FileNotFoundError: Si el archivo KML especificado no existe.
    """
    # Read kml file
    if not kml_filepath.exists():
        raise FileNotFoundError(f"El archivo KML {kml_filepath} no existe.")
    with open(kml_filepath, "r", encoding="utf-8") as f:
        kml_text = f.read()

    kwargs = {}
    if output_filename is not None:
        kwargs["output_filename"] = output_filename
    if should_download is not None:
        kwargs["should_download"] = should_download

    return convert_kml_to_geojson(kml_text, **kwargs)


def create_geojson_from_annotations(
    pic_name: str,
    coco_annotations: dict[str, Any],
    jgw_data: dict[str, Any],
    should_download: bool = False,
    output_filename: Path = DOWNLOAD_GEOJSON_FOLDER / "annotations.geojson",
    upload_to_drive: bool = False,
    epsg_code: str = CONFIG.georeferenciacion.codigo_epsg,
) -> gpd.GeoDataFrame:
    """
    Crea un GeoDataFrame a partir de las anotaciones COCO y los datos de georreferenciación (JGW),
    y opcionalmente guarda el resultado como un archivo GeoJSON.

    Args:
        pic_name (str): Nombre de la imagen para la cual se generarán las anotaciones geográficas.
        coco_annotations (dict[str, Any]): Diccionario con las anotaciones en formato COCO, incluyendo
            categorías, imágenes y anotaciones.
        jgw_data (dict[str, Any]): Diccionario con los datos de georreferenciación provenientes del archivo JGW.
        should_download (bool, opcional): Indica si el GeoDataFrame generado debe guardarse como un archivo GeoJSON.
            Por defecto es False.
        output_filename (Path, opcional): Ruta del archivo donde se guardará el GeoJSON si `should_download` es True.
            Por defecto es "annotations.geojson" en la carpeta `DOWNLOAD_GEOJSON_FOLDER`.
        upload_to_drive (bool, opcional): Indica si el archivo GeoJSON generado debe subirse a Google Drive.
            Por defecto es False. Actualmente no implementado.
        geo_sistema_referencia (str, opcional): Código EPSG del sistema de referencia geográfico que se asignará
            al GeoDataFrame. Por defecto se toma de la configuración global `CONFIG.georeferenciacion.codigo_epsg`.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame que contiene las anotaciones geográficas con sus propiedades y geometrías.
        Si no se encuentran anotaciones para la imagen especificada, se devuelve un GeoDataFrame vacío.

    Raises:
        NotImplementedError: Si `upload_to_drive` es True, ya que la funcionalidad de subida a Google Drive
        no está implementada.

    Notas:
        - Las coordenadas globales de los bounding boxes se calculan utilizando los datos de georreferenciación
          proporcionados en el archivo JGW.
        - Las geometrías generadas son puntos (centroides) basados en los bounding boxes de las anotaciones.
        - El archivo GeoJSON se guarda en el sistema de archivos si `should_download` es True.
    """
    # 1 - Configuraciones generales
    category_map = {cat["id"]: cat["name"] for cat in coco_annotations["categories"]}

    # 2 - Obtener el id de la imagen en las anotaciones
    image_id = CocoDatasetUtils.get_image_id_from_annotations(pic_name, coco_annotations)

    # 3 - Obtener las anotaciones de la imagen
    annotations = [ann for ann in coco_annotations["annotations"] if ann["image_id"] == image_id]
    if not annotations:
        LOGGER.warning(f"No se encontraron anotaciones para la imagen {pic_name}.")
        return gpd.GeoDataFrame()

    # 4 - Preparar listas para almacenar los datos
    geometries = []
    properties = []

    # 5 - Para cada anotación, obtener el bbox y la categoría
    for annotation in annotations:
        bbox = annotation["bbox"]
        category_name = category_map.get(annotation["category_id"], "Sin categoría")

        # 5.1 - Convertir el bbox a coordenadas geográficas utilizando los datos del archivo JGW
        global_coordinates = ConvertorCoordenadas.convert_bbox_image_to_world(bbox, jgw_data)

        # 5.2 - Obtener el centroide del bbox
        x_coords = [
            global_coordinates["tl"][0],
            global_coordinates["tr"][0],
            global_coordinates["br"][0],
            global_coordinates["bl"][0],
        ]
        y_coords = [
            global_coordinates["tl"][1],
            global_coordinates["tr"][1],
            global_coordinates["br"][1],
            global_coordinates["bl"][1],
        ]
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        centroid = (centroid_x, centroid_y)

        # 5.3 - Crear un objeto Point de Shapely con las coordenadas del centroide
        point = Point(centroid)

        # 5.4 - Guardar la geometría y propiedades
        geometries.append(point)
        properties.append(
            {
                "name": category_name,
                "annotation_id": annotation.get("id", None),
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_width": bbox[2],
                "bbox_height": bbox[3],
                "global_tl_x": global_coordinates["tl"][0],
                "global_tl_y": global_coordinates["tl"][1],
                "global_br_x": global_coordinates["br"][0],
                "global_br_y": global_coordinates["br"][1],
            }
        )

    # 6 - Crear un DataFrame con las propiedades
    properties_df = pd.DataFrame(properties)

    # 7 - Crear un GeoDataFrame con las geometrías y propiedades
    gdf = gpd.GeoDataFrame(properties_df, geometry=geometries)

    # 8 - Configurar el sistema de coordenadas (CRS)
    gdf.crs = epsg_code

    # 9 - Reproyectar a EPSG:4326 si es necesario
    # Reproyectar a WGS84 (EPSG:4326) si no está ya en ese CRS
    if gdf.crs is not None and gdf.crs != "EPSG:4326":
        try:
            gdf = gdf.to_crs(epsg=4326)
            LOGGER.debug("GeoDataFrame reproyectado a EPSG:4326.")
        except Exception as e:
            raise ValueError(
                f"Error al reproyectar el GeoDataFrame a EPSG:4326: {e}. Asegúrate de que el CRS original sea válido."
            )

    if should_download:
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_filename, driver="GeoJSON")
        LOGGER.info(f"GeoJSON guardado en {output_filename}")
        # 10 - Opcional: Subir a Google Drive si se solicita
        if upload_to_drive:
            # Aquí iría tu código para subir a Drive
            raise NotImplementedError("Subida a Google Drive no implementada.")
    return gdf


@app.command()
def generate_geojson_from_annotations_from_path(
    pic_name: str,
    coco_annotation_path: Path,
    jgw_data_path: Path,
    should_download: bool = None,
    output_filename: Path = None,
    upload_to_drive: bool = None,
    geo_sistema_referencia: str = None,
) -> gpd.GeoDataFrame:
    """
    Genera un GeoJSON a partir de anotaciones COCO y datos JGW desde rutas especificadas.

    Args:
        pic_name (str): Nombre de la imagen asociada a las anotaciones.
        coco_annotation_path (Path): Ruta al archivo de anotaciones en formato COCO.
        jgw_data_path (Path): Ruta al archivo JGW que contiene datos de georreferenciación.
        should_download (bool, opcional): Indica si el archivo generado debe descargarse. Por defecto es None.
        output_filename (Path, opcional): Ruta y nombre del archivo de salida GeoJSON. Por defecto es None.
        upload_to_drive (bool, opcional): Indica si el archivo generado debe subirse a Google Drive. Por defecto es None.
        geo_sistema_referencia (str, opcional): Sistema de referencia geográfico para el GeoJSON. Por defecto es None.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame generado a partir de las anotaciones y datos de georreferenciación.

    Raises:
        FileNotFoundError: Si el archivo de anotaciones COCO o el archivo JGW no existen en las rutas especificadas.
    """
    if not coco_annotation_path.exists():
        raise FileNotFoundError(f"El archivo de anotaciones {coco_annotation_path} no existe.")
    if not jgw_data_path.exists():
        raise FileNotFoundError(f"El archivo JGW {jgw_data_path} no existe.")

    coco_annotations = CocoDatasetUtils.load_annotations_from_path(coco_annotation_path)
    with open(jgw_data_path, "r") as f:
        jgw_data = json.load(f)

    kwargs = {}
    if output_filename is not None:
        kwargs["output_filename"] = output_filename
    if should_download is not None:
        kwargs["should_download"] = should_download
    if upload_to_drive is not None:
        kwargs["upload_to_drive"] = upload_to_drive
    if geo_sistema_referencia is not None:
        kwargs["geo_sistema_referencia"] = geo_sistema_referencia

    return create_geojson_from_annotations(pic_name, coco_annotations, jgw_data, **kwargs)


def generate_kml_from_geojson(
    gdf: gpd.GeoDataFrame,
    category_column: str = "name",
    target_category: Optional[str] = None,
    should_download: bool = False,
    output_filename: Path = DOWNLOAD_KMLS_FOLDER / "palmeras.kml",
) -> Optional[kml.KML]:
    """
    Genera un archivo KML a partir de un GeoDataFrame de GeoPandas.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame que contiene las geometrías y atributos.
        category_column (str): Nombre de la columna que contiene las categorías. Por defecto es "category".
        target_category (Optional[str]): Categoría específica que se desea filtrar. Si es None, se incluyen todas las categorías.
        should_download (bool): Indica si el archivo KML generado debe guardarse en disco. Por defecto es False.
        output_filename (Path): Ruta y nombre del archivo KML a guardar. Por defecto es "palmeras.kml" en la carpeta definida por DOWNLOAD_KMLS_FOLDER.

    Returns:
        Optional[kml.KML]: Objeto KML generado. Retorna None si el GeoDataFrame está vacío o si no se encuentran elementos con la categoría especificada.

    Raises:
        ValueError: Si ocurre un error al reproyectar el GeoDataFrame a EPSG:4326.

    Notas:
        - Solo se procesan geometrías de tipo "Point". Las demás geometrías se ignoran.
        - Si `should_download` es True, el archivo KML se guarda en la ubicación especificada por `output_filename`.
        - El CRS del GeoDataFrame debe ser válido para realizar la reproyección.
    """
    if gdf.empty:
        LOGGER.warning("El GeoDataFrame está vacío. No se creará el archivo KML.")
        return None

    k = kml.KML()
    ns = "{http://www.opengis.net/kml/2.2}"

    # Crear un documento KML
    palm_document = kml.Document(ns, id="docid", description="PalmTrees")
    k.append(palm_document)

    # Crear una carpeta para las palmeras
    palm_folder = kml.Folder(ns, id="palmeras_folder", name="Palmeras")
    palm_document.append(palm_folder)

    if target_category:
        palm_gdf = gdf[gdf[category_column] == target_category].copy()
    else:
        palm_gdf = gdf.copy()

    if palm_gdf.empty:
        LOGGER.warning(
            f"No se encontraron elementos con la categoría '{target_category}'. No se creará el archivo KML."
        )
        return None

    # Reproyectar a WGS84 (EPSG:4326) si no está ya en ese CRS
    if palm_gdf.crs is not None and palm_gdf.crs != "EPSG:4326":
        try:
            palm_gdf = palm_gdf.to_crs(epsg=4326)
            LOGGER.debug("GeoDataFrame reproyectado a EPSG:4326 para el KML.")
        except Exception as e:
            raise ValueError(
                f"Error al reproyectar el GeoDataFrame a EPSG:4326: {e}. Asegúrate de que el CRS original sea válido."
            )

    for index, row in palm_gdf.iterrows():
        if row.geometry.geom_type == "Point":
            coords = (row.geometry.x, row.geometry.y)
            point = Point(coords)
            p = kml.Placemark(ns, id=f"palmera_{index}", name=f"{row[category_column]}", geometry=point)
            palm_folder.append(p)
        else:
            LOGGER.warning(f"La geometría del elemento con índice {index} no es un Point. No se agregará al KML.")

    if should_download:
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            k.write(output_filename)
            LOGGER.info(f"Archivo KML guardado en {output_filename}")
        except Exception as e:
            LOGGER.error(f"Error al guardar el archivo KML: {e}")
    return k


@app.command()
def generate_kml_from_geojson_from_path(
    gdf_path: Path,
    category_column: str = None,
    target_category: Optional[str] = None,
    reproject: bool = None,
    should_download: bool = None,
    output_filename: Path = None,
) -> Optional[kml.KML]:
    """
    Genera un archivo KML a partir de un archivo GeoJSON ubicado en una ruta específica.

    Args:
        gdf_path (Path): Ruta al archivo GeoJSON que se utilizará como entrada.
        category_column (str, opcional): Nombre de la columna que contiene las categorías en el GeoDataFrame.
        target_category (Optional[str], opcional): Categoría específica que se desea filtrar en el GeoDataFrame.
        reproject (bool, opcional): Indica si se debe reproyectar el GeoDataFrame a un sistema de coordenadas específico.
        should_download (bool, opcional): Indica si el archivo KML generado debe ser descargado automáticamente.
        output_filename (Path, opcional): Ruta y nombre del archivo KML de salida.

    Returns:
        Optional[kml.KML]: Objeto KML generado, o None si el archivo GeoJSON está vacío o no se puede procesar.
    """
    gdf = _load_gdf_from_path(gdf_path)
    if gdf.empty:
        LOGGER.warning(f"El archivo {gdf_path} está vacío. No se creará el archivo KML.")
        return None

    kwargs = {}
    if category_column is not None:
        kwargs["category_column"] = category_column
    if target_category is not None:
        kwargs["target_category"] = target_category
    if reproject is not None:
        kwargs["reproject"] = reproject
    if should_download is not None:
        kwargs["should_download"] = should_download
    if output_filename is not None:
        kwargs["output_filename"] = output_filename

    return generate_kml_from_geojson(gdf, **kwargs)


def _load_gdf_from_path(
    file_path: Path,
    crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Carga un archivo GeoJSON y lo convierte a un GeoDataFrame.

    Args:
        file_path (str): Ruta al archivo GeoJSON.
        driver (str, optional): Controla el formato del archivo. Defaults to "GeoJSON".
        crs (str, optional): Sistema de referencia de coordenadas. Defaults to None.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame que contiene los datos del archivo.
    """
    gdf = gpd.read_file(file_path)
    if crs:
        gdf.crs = crs
    return gdf


def _process_single_patch(
    imagen: dict[str, Any],
    patch: dict[str, Any],
    gdf: gpd.GeoDataFrame,
    bbox_size: tuple[float, float] = (CONFIG.bbox_size.width, CONFIG.bbox_size.height),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Procesa un solo parche de imagen y genera las anotaciones COCO correspondientes para los puntos del GeoDataFrame que caen dentro del parche.

    Args:
        imagen (dict[str, Any]): Diccionario con información de la imagen original, incluyendo metadatos y datos de georreferenciación (jgw_data).
        patch (dict[str, Any]): Diccionario con información del parche, incluyendo nombre, dimensiones y posición dentro de la imagen.
        gdf (gpd.GeoDataFrame): GeoDataFrame con puntos georreferenciados (por ejemplo, palmeras) a ser anotados.
        bbox_size (tuple[float, float], optional): Tamaño del bounding box (ancho, alto) en píxeles para cada anotación. Por defecto, se toma de la configuración.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]: Una tupla con el diccionario de la imagen (formato COCO) y una lista de anotaciones COCO generadas para el parche.
    """
    image = {
        "width": patch["width"],
        "height": patch["height"],
        "file_name": f"{patch["patch_name"]}.jpg",
        "date_captured": imagen["date_captured"].strftime("%Y-%m-%d %H:%M:%S"),
    }
    annotations = []
    jgw_data = imagen.get("jgw_data")
    if not jgw_data:
        LOGGER.warning(f"No se encontró el archivo JGW para la imagen {imagen['name']}.")
        return image, annotations

    # Obtener las coordenadas del parche
    x_start, y_start, patch_width, patch_height = (
        patch["x_start"],
        patch["y_start"],
        patch["width"],
        patch["height"],
    )
    # Esquinas del parche dentro de la imagen
    esquinas_imagen = [
        (x_start, y_start),  # esquina superior izquierda
        (x_start + patch_width, y_start),  # esquina superior derecha
        (x_start + patch_width, y_start + patch_height),  # esquina inferior derecha
        (x_start, y_start + patch_height),  # esquina inferior izquierda
    ]

    # Convertir las coordenadas del parche a coordenadas globales
    esquinas_mundo = [
        ConvertorCoordenadas.convert_point_image_to_world(punto, jgw_data=jgw_data) for punto in esquinas_imagen
    ]

    poligono_parche = Polygon(esquinas_mundo)

    # Filtrar los puntos del GeoDataFrame que están dentro del polígono del parche
    puntos_en_parche = gdf[gdf.geometry.within(poligono_parche)]

    if puntos_en_parche.empty:
        LOGGER.debug(f"No se encontraron puntos dentro del parche {patch['patch_name']}.")
        return image, annotations

    # Procesar cada punto encontrado del parche
    puntos_en_parche.reset_index(inplace=True, drop=True)
    for index, row in puntos_en_parche.iterrows():
        punto_mundo = (row.geometry.x, row.geometry.y)

        # Convertir a coordenadas de imagen
        punto_imagen = ConvertorCoordenadas.convert_point_world_to_image(punto_mundo, jgw_data)

        # Convertir a coordenadas locales del parche
        punto_parche = ConvertorCoordenadas.convert_point_image_to_patch(
            punto_imagen, x_start, y_start, patch_width, patch_height
        )

        # Crear el bounding box
        bbox_ancho, bbox_alto = bbox_size
        x_centro, y_centro = punto_parche

        # Asegurarse que el bbox no exceda los límites del parche
        x_min = max(0, x_centro - bbox_ancho / 2)
        y_min = max(0, y_centro - bbox_alto / 2)
        x_max = min(patch_width, x_centro + bbox_ancho / 2)
        y_max = min(patch_height, y_centro + bbox_alto / 2)

        # Calcular dimensiones finales del bbox
        ancho = x_max - x_min
        alto = y_max - y_min
        area = ancho * alto

        category_name = "palmera-google-maps"
        annotation = {
            "id": index + 1,
            "segmentation": [],
            "iscrowd": 0,
            "attributes": {
                "occluded": False,
                "rotation": 0.0,
            },
            "category_name": category_name,
            "area": area,
            "bbox": [x_min, y_min, ancho, alto],
        }

        annotations.append(annotation)

    return image, annotations


def generate_coco_annotations_from_geojson(
    gdf: gpd.GeoDataFrame,
    output_filename: Optional[Path] = None,
    use_parallel: bool = True,
    max_workers: int = 10,
) -> dict[str, Any]:
    """Crea anotaciones en formato COCO a partir de un GeoDataFrame.

    Este método procesa un GeoDataFrame que contiene puntos georreferenciados y genera
    anotaciones en formato COCO para imágenes y parches asociados. Las anotaciones incluyen
    información sobre bounding boxes y categorías.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame que contiene los puntos georreferenciados.
        output_filename (Optional[Path], optional): Ruta donde se guardará el archivo de anotaciones COCO.
                                                    Si no se proporciona, las anotaciones no se guardarán en un archivo.
                                                    Defaults to None.
        use_parallel (bool, optional): Indica si se debe usar procesamiento en paralelo para acelerar la generación
                                       de anotaciones. Defaults to True.
        max_workers (int, optional): Número máximo de procesos paralelos a utilizar. Defaults to 10.

    Returns:
        dict[str, Any]: Diccionario con las anotaciones en formato COCO, incluyendo las imágenes, categorías y bounding boxes.
    """
    coco_annotations = {
        "info": COCO_DATASET_DATA["info"],
        "licenses": COCO_DATASET_DATA["licenses"],
        "categories": COCO_DATASET_DATA["categories"],
        "images": [],
        "annotations": [],
    }

    category_map = {cat["name"]: cat["id"] for cat in coco_annotations["categories"]}

    coco_images = []
    image_annotations = []
    imagenes = DB.get_collection("imagenes")

    # Consulta con agregación para filtrar imágenes y sus patches
    pipeline = [
        # Filtrar imágenes donde downloaded = true
        {"$match": {"downloaded": True}},
        # Crear un nuevo campo 'patches_filtrados' que contenga solo los patches donde is_white = false
        {
            "$addFields": {
                "patches_filtrados": {
                    "$filter": {"input": "$patches", "as": "patch", "cond": {"$eq": ["$$patch.is_white", False]}}
                }
            }
        },
        # Filtrar para incluir solo imágenes que tienen al menos un patch válido
        {"$match": {"patches_filtrados.0": {"$exists": True}}},
        # Opcionalmente: proyectar solo campos necesarios con $project (mejorar optimización, pero queda hardcodeado)
    ]

    filtered_images = list(imagenes.aggregate(pipeline))
    LOGGER.debug(f"Se encontraron {len(filtered_images)} imágenes con parches no blancos.")

    # Creamos tareas asíncronas para cada imagen y parche
    tareas = [(imagen, patch) for imagen in filtered_images for patch in imagen["patches_filtrados"]]
    LOGGER.debug(f"Se encontraron {len(tareas)} tareas para procesar.")

    if tareas:
        annotation_images = []
        if use_parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [(executor.submit(_process_single_patch, imagen, patch, gdf)) for imagen, patch in tareas]
                for future in tqdm(futures, desc="Procesando parches"):
                    try:
                        image, annotations = future.result()
                        if image and annotations:
                            annotation_images.append((image, annotations))
                    except Exception as e:
                        LOGGER.error(f"Error procesando el parche: {e}")
        else:
            for imagen, patch in tqdm(tareas, desc="Procesando parches"):
                try:
                    image, annotations = _process_single_patch(imagen, patch, gdf)
                    if image and annotations:
                        annotation_images.append((image, annotations))
                except Exception as e:
                    LOGGER.error(f"Error procesando el parche: {e}")

        for id, image, annotations in enumerate(annotation_images):
            image_id = id + 1
            image = {"id": image_id, **image}

            annotations = [
                {
                    **annotation,
                    "image_id": image_id,
                    "category_id": category_map[annotation["category_name"]],
                }
                for annotation in annotations
            ]

            coco_images.append(image)
            image_annotations.append(annotations)

        coco_annotations["images"] = coco_images
        coco_annotations["annotations"] = image_annotations

        if output_filename:
            with open(output_filename, "w") as f:
                json.dump(coco_annotations, f, indent=4)
                LOGGER.debug(f"Anotaciones guardadas en {output_filename}")

        return coco_annotations
    else:
        LOGGER.warning("No se encontraron tareas para procesar.")


@app.command()
def generate_coco_annotations_from_geojson_from_path(
    gdf_path: Path,
    output_filename: Optional[Path] = None,
    use_parallel: bool = None,
    max_workers: int = None,
) -> Optional[dict[str, Any]]:
    """
    Genera anotaciones en formato COCO a partir de un archivo GeoJSON especificado por su ruta.

    Args:
        gdf_path (Path): Ruta al archivo GeoJSON que contiene los datos geoespaciales.
        output_filename (Optional[Path], opcional): Ruta del archivo donde se guardarán las anotaciones COCO generadas.
            Si no se especifica, las anotaciones no se guardarán en un archivo.
        use_parallel (bool, opcional): Indica si se debe utilizar procesamiento paralelo para generar las anotaciones.
            Por defecto es None.
        max_workers (int, opcional): Número máximo de trabajadores para el procesamiento paralelo.
            Solo se utiliza si `use_parallel` es True. Por defecto es None.

    Returns:
        Optional[dict[str, Any]]: Diccionario con las anotaciones en formato COCO generadas.
        Si el archivo GeoJSON está vacío o no existe, se devuelve None.

    Raises:
        FileNotFoundError: Si el archivo GeoJSON especificado por `gdf_path` no existe.

    Advertencias:
        - Si el archivo GeoJSON está vacío, se genera una advertencia en el registro y no se generan anotaciones COCO.
    """
    if not gdf_path.exists():
        raise FileNotFoundError(f"El archivo GeoJSON {gdf_path} no existe.")

    gdf = _load_gdf_from_path(gdf_path)
    if gdf.empty:
        LOGGER.warning(f"El archivo {gdf_path} está vacío. No se generarán anotaciones COCO.")
        return None

    kwargs = {}
    if output_filename is not None:
        kwargs["output_filename"] = output_filename
    if use_parallel is not None:
        kwargs["use_parallel"] = use_parallel
    if max_workers is not None:
        kwargs["max_workers"] = max_workers

    return generate_coco_annotations_from_geojson(gdf, **kwargs)


def merge_annotations():
    raise NotImplementedError("Función merge_annotations no implementada.")


if __name__ == "__main__":
    app()
