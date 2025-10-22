# Dependencias del sistema
from concurrent.futures import ProcessPoolExecutor
import datetime, json, os, zipfile, io

# Dependencias propias
from modulo_utilidades.config import COCOCategory, settings as CONFIG
from modulo_utilidades.database_comunication.mongodb_client import mongodb as DB
from ..core.labeling.procesador_geojson_kml_core import create_geojson_from_annotations, generate_kml_from_geojson
from ..core.labeling.convertor_cordenadas_core import (
    convert_point_image_to_patch,
    convert_point_image_to_world,
    convert_point_world_to_image,
)
from .procesador_anotaciones_coco_dataset import load_annotations_from_path

# Dependencias de terceros
from tqdm import tqdm
import kml2geojson
import requests
import geopandas as gpd
from fastkml import kml
from pathlib import Path
from typing import Any, Optional
from shapely.geometry import Polygon
import typer
from loguru import logger as LOGGER

# Configuraciones
DOWNLOAD_TEMP_FOLDER: Path = CONFIG.folders.download_temp_folder
DOWNLOAD_GOOGLE_MAPS_FOLDER: Path = CONFIG.folders.download_google_maps_folder
DOWNLOAD_KMLS_FOLDER: Path = CONFIG.folders.download_kmls_folder
DOWNLOAD_GEOJSON_FOLDER: Path = CONFIG.folders.download_geojson_folder
COCO_DATASET_DATA: dict[str, Any] = CONFIG.coco_dataset.model_dump()
COCO_DATASET_CATEGORIES: COCOCategory = CONFIG.coco_dataset.categories
CODIGO_EPSG_DEFAULT: str = CONFIG.georeferenciacion.codigo_epsg
GOOGLE_MAPS_BASE_URL: str = CONFIG.google_maps.base_url
GOOGLE_MAPS_MID: str = CONFIG.google_maps.mid
BBOX_SIZE_DEFAULT_WH: tuple[int, int] = (CONFIG.bbox_size.width, CONFIG.bbox_size.height)

app = typer.Typer()


@app.command()
def download_kmz_from_gmaps(
    base_url: str = GOOGLE_MAPS_BASE_URL,
    mid: str = GOOGLE_MAPS_MID,
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


def create_geojson_from_annotations_wrapper(
    pic_name: str,
    coco_annotations: dict[str, Any],
    jgw_data: dict[str, Any],
    output_file_path: Optional[Path] = None,
    upload_to_drive: bool = False,
    epsg_code: str = CODIGO_EPSG_DEFAULT,
) -> gpd.GeoDataFrame:
    return create_geojson_from_annotations(
        pic_name,
        coco_annotations,
        jgw_data,
        output_file_path=output_file_path,
        upload_to_drive=upload_to_drive,
        epsg_code=epsg_code,
    )


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

    coco_annotations = load_annotations_from_path(coco_annotation_path)
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

    return create_geojson_from_annotations_wrapper(pic_name, coco_annotations, jgw_data, **kwargs)


def generate_kml_from_geojson_wrapper(
    gdf: gpd.GeoDataFrame,
    category_column: str = "name",
    target_category: Optional[str] = None,
    output_file_path: Optional[Path] = None,
) -> Optional[kml.KML]:
    return generate_kml_from_geojson(
        gdf,
        category_column=category_column,
        target_category=target_category,
        output_file_path=output_file_path,
    )


@app.command()
def generate_kml_from_geojson_from_path(
    gdf_path: Path,
    category_column: str = None,
    target_category: Optional[str] = None,
    reproject: bool = None,
    output_file_path: Optional[Path] = None,
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
    if output_file_path is not None:
        kwargs["output_file_path"] = output_file_path

    return generate_kml_from_geojson_wrapper(gdf, **kwargs)


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
    bbox_size: tuple[float, float] = BBOX_SIZE_DEFAULT_WH,
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
    esquinas_mundo = [convert_point_image_to_world(punto, jgw_data=jgw_data) for punto in esquinas_imagen]

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
        punto_imagen = convert_point_world_to_image(punto_mundo, jgw_data)

        # Convertir a coordenadas locales del parche
        punto_parche = convert_point_image_to_patch(punto_imagen, x_start, y_start, patch_width, patch_height)

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
