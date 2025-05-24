from concurrent.futures import ProcessPoolExecutor
from pprint import pprint
import datetime, sys, json, os
import zipfile

from tqdm import tqdm

sys.path.append(os.path.abspath("../"))

import kml2geojson
import requests

import pandas as pd
import geopandas as gpd
from fastkml import kml

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Point, Polygon, box

from apps_config.settings import Config
from apps_utils.logging import Logging
from apps_com_db.mongodb_client import MongoDB

import apps_etiquetado.procesador_anotaciones_coco_dataset as CocoDatasetUtils
import apps_etiquetado.convertor_cordenadas as ConvertorCoordenadas

CONFIG = Config().config_data
LOGGER = Logging().logger
DB = MongoDB().db

download_folder = Path(CONFIG["folders"]["download_folder"])
DOWNLOAD_TEMP_FOLDER = download_folder / "temp"
DOWNLOAD_GOOGLE_MAPS_FOLDER = download_folder / "google_maps"
KMLS_FOLDER = download_folder / "kmls"


def download_kmz_from_gmaps(
    base_url: str = CONFIG["google_maps"]["base_url"],
    mid: str = CONFIG["google_maps"]["mid"],
    output_filename: Optional[Path] = None,
) -> Optional[Path]:
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
        Path(DOWNLOAD_TEMP_FOLDER).mkdir(parents=True, exist_ok=True)
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


def convert_kml_to_geojson(kml_file_path: Path, geojson_file_path: Optional[Path]) -> Optional[Path]:
    """
    Convierte un archivo KML a formato GeoJSON.

    Args:
        kml_file_path (str): Ruta al archivo KML de entrada.
        geojson_file_path (str): Ruta donde se guardará el archivo GeoJSON convertido.
    """
    try:
        geojson = kml2geojson.main.convert(kml_file_path)[0]
        LOGGER.info(f"Conversión exitosa de '{kml_file_path}' a '{geojson_file_path}'.")

        if not geojson_file_path:
            Path(DOWNLOAD_GOOGLE_MAPS_FOLDER).mkdir(parents=True, exist_ok=True)
            geojson_file_path = DOWNLOAD_GOOGLE_MAPS_FOLDER / f"{kml_file_path.stem}.geojson"
        # Guardar el archivo GeoJSON
        with open(geojson_file_path, "w") as f:
            json.dump(geojson, f, indent=4)
            return geojson_file_path
        LOGGER.info(f"Archivo GeoJSON guardado en '{geojson_file_path}'.")

    except FileNotFoundError:
        LOGGER.error(f"Error: Archivo KML no encontrado en '{kml_file_path}'.")
    except ValueError as e:
        LOGGER.error(f"Error de valor: {e}")
    except Exception as e:
        LOGGER.error(f"Ocurrió un error inesperado: {e}")
    return None


def create_geojson_from_annotation(
    pic_name: str,
    coco_annotation: Dict[str, Any],
    jgw_data: Dict[str, Any],
    output_filename: Optional[Path] = None,
    upload_to_drive: bool = False,
    geo_sistema_referencia: str = CONFIG["georeferenciacion"]["codigo_epsg"],
) -> gpd.GeoDataFrame:
    """
    Crea un GeoDataFrame a partir de las anotaciones COCO y datos de georreferenciación.

    Args:
        pic_name (str): Nombre de la imagen para la cual se generará el GeoJSON.
        coco_annotation (Dict[str, Any]): Anotaciones en formato COCO que incluyen categorías y bounding boxes.
        jgw_data (Dict[str, Any]): Datos de georreferenciación provenientes de un archivo JGW.
        output_filename (Optional[Path], optional): Ruta donde se guardará el archivo GeoJSON. Defaults to None.
        upload_to_drive (bool, optional): Indica si el archivo GeoJSON debe subirse a Google Drive. Defaults to False.
        geo_sistema_referencia (str, optional): Código EPSG del sistema de referencia geográfico. Defaults to CONFIG["georeferenciacion"]["codigo_epsg"].

    Raises:
        NotImplementedError: Si se solicita subir el archivo a Google Drive, pero la funcionalidad no está implementada.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame que contiene las geometrías y propiedades de las anotaciones.

    Ejemplo de uso:

        >>> image_name = "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm"
        >>> patch_name = "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_0"
        >>> annotations_field = "cvat"
        >>> pic_name = image_name
        >>> if pic_name == image_name:
        >>>     coco_annotations = load_coco_annotation_from_mongodb(
        ...         field_name=annotations_field, image_name=image_name
        ...     )
        >>>     jgw_data = load_jgw_file_from_mongodb(image_name=image_name)
        >>> else:
        >>>     coco_annotations = load_coco_annotation_from_mongodb(
        ...         field_name=annotations_field, patch_name=patch_name
        ...     )
        >>>     jgw_data = load_jgw_file_from_mongodb(patch_name=patch_name)
        >>> gdf = create_geojson_from_annotation(
        ...     pic_name=pic_name,
        ...     coco_annotation=coco_annotations,
        ...     jgw_data=jgw_data,
        ...     output_filename=DOWNLOAD_TEMP_FOLDER / f"{pic_name}.geojson",
        ... )

        >>> # Cambiar proyectar en otro sistema de coordenadas
        >>> # gdf_crs = gdf.to_crs("EPSG:4326")
        >>> # gdf_crs.to_file(
        ... #     DOWNLOAD_TEMP_FOLDER / f"{pic_name}_4326.geojson",
        ... #     driver="GeoJSON",
        ... # )
    """
    # 1 - Configuraciones generales
    category_map = {cat["id"]: cat["name"] for cat in coco_annotation["categories"]}

    # 2 - Obtener el id de la imagen en las anotaciones
    image_id = CocoDatasetUtils.get_image_id_from_annotations(pic_name, coco_annotation)

    # 3 - Obtener las anotaciones de la imagen
    annotations = [ann for ann in coco_annotation["annotations"] if ann["image_id"] == image_id]
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
                "category": category_name,
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
    gdf.crs = geo_sistema_referencia

    # 9 - Guardar como GeoJSON si se proporciona un nombre de archivo
    if output_filename:
        output_path = output_filename
        Path(output_path.parent).mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")
        LOGGER.info(f"GeoJSON guardado en {output_path}")

        # 10 - Opcional: Subir a Google Drive si se solicita
        if upload_to_drive:
            # Aquí iría tu código para subir a Drive
            raise NotImplementedError("Subida a Google Drive no implementada.")
    return gdf


def create_kml_from_geojson(
    gdf: gpd.GeoDataFrame,
    kml_filename: Optional[Path] = None,
    category_column: str = "category",
    target_category: Optional[str] = "palmera",
    reproject: bool = True,
):
    """
    Crea un archivo KML a partir de un GeoDataFrame.

    Este método toma un GeoDataFrame con geometrías y propiedades, filtra las categorías
    deseadas, y genera un archivo KML con las geometrías y propiedades correspondientes.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame que contiene las geometrías y propiedades.
        kml_filename (Optional[Path], optional): Ruta donde se guardará el archivo KML.
                                                 Si no se proporciona, se utiliza una ruta predeterminada.
                                                 Defaults to None.
        category_column (str, optional): Nombre de la columna que contiene las categorías.
                                         Defaults to "category".
        target_category (Optional[str], optional): Categoría objetivo que se desea incluir en el KML.
                                                   Si no se proporciona, se incluyen todas las categorías.
                                                   Defaults to "palmera".
        reproject (bool, optional): Si se debe reproyectar el GeoDataFrame a EPSG:4326 (WGS84).
                                    Defaults to True.

    Raises:
        ValueError: Si ocurre un error al reproyectar el GeoDataFrame.

    Returns:
        None: El archivo KML se guarda en la ubicación especificada o predeterminada.

    Ejemplo de uso:

        >>> image_name = "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm"
        >>> patch_name = "8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm_patch_0"
        >>> annotations_field = "cvat"
        >>> pic_name = image_name
        >>> if pic_name == image_name:
        >>>     coco_annotations = load_coco_annotation_from_mongodb(
        ...         field_name=annotations_field, image_name=image_name
        ...     )
        >>>     jgw_data = load_jgw_file_from_mongodb(image_name=image_name)
        >>> else:
        >>>     coco_annotations = load_coco_annotation_from_mongodb(
        ...         field_name=annotations_field, patch_name=patch_name
        ...     )
        >>>     jgw_data = load_jgw_file_from_mongodb(patch_name=patch_name)
        >>> gdf = create_geojson_from_annotation(
        ...     pic_name=pic_name,
        ...     coco_annotation=coco_annotations,
        ...     jgw_data=jgw_data,
        ...     output_filename=DOWNLOAD_TEMP_FOLDER / f"{pic_name}.geojson",
        ... )
        >>> create_kml_from_geojson(
        ...     gdf=gdf,
        ...     kml_filename=KMLS_FOLDER / f"{pic_name}.kml",
        ...     category_column="category",
        ...     target_category=None
        ... )
    """
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
        return

    # Reproyectar a WGS84 (EPSG:4326) si no está ya en ese CRS
    if reproject and palm_gdf.crs is not None and palm_gdf.crs != "EPSG:4326":
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
            p = kml.Placemark(ns, id=f"palmera_{index}", name=f"{row.category}", geometry=point)
            palm_folder.append(p)
        else:
            LOGGER.warning(f"La geometría del elemento con índice {index} no es un Point. No se agregará al KML.")

    if kml_filename:
        Path(kml_filename).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(KMLS_FOLDER).mkdir(parents=True, exist_ok=True)
        kml_filename = KMLS_FOLDER / f"kml_{target_category}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.kml"
    try:
        k.write(kml_filename)
        LOGGER.info(f"Archivo KML guardado en {kml_filename}")
    except Exception as e:
        LOGGER.error(f"Error al guardar el archivo KML: {e}")


def load_gdf_from_file(
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
    imagen: Dict[str, Any],
    patch: Dict[str, Any],
    gdf: gpd.GeoDataFrame,
    bbox_size: Tuple[float, float] = (CONFIG["bbox_size"]["width"], CONFIG["bbox_size"]["height"]),
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Procesa un solo parche de imagen y genera las anotaciones COCO correspondientes para los puntos del GeoDataFrame que caen dentro del parche.

    Args:
        imagen (Dict[str, Any]): Diccionario con información de la imagen original, incluyendo metadatos y datos de georreferenciación (jgw_data).
        patch (Dict[str, Any]): Diccionario con información del parche, incluyendo nombre, dimensiones y posición dentro de la imagen.
        gdf (gpd.GeoDataFrame): GeoDataFrame con puntos georreferenciados (por ejemplo, palmeras) a ser anotados.
        bbox_size (Tuple[float, float], optional): Tamaño del bounding box (ancho, alto) en píxeles para cada anotación. Por defecto, se toma de la configuración.

    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]: Una tupla con el diccionario de la imagen (formato COCO) y una lista de anotaciones COCO generadas para el parche.
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


# TODO: Pendiente
def create_annotations_from_geojson(
    gdf: gpd.GeoDataFrame,
    output_filename: Optional[Path] = None,
    use_parallel: bool = True,
    max_workers: int = 10,
) -> Dict[str, Any]:
    coco_annotations = {
        "info": CONFIG["coco_dataset"]["info"],
        "licenses": CONFIG["coco_dataset"]["licenses"],
        "categories": CONFIG["google_maps"]["categories"],
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

        pprint(coco_annotations)
    else:
        LOGGER.warning("No se encontraron tareas para procesar.")


def merge_annotations():
    raise NotImplementedError("Función merge_annotations no implementada.")
