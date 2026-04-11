# Dependencias del sistema
from pathlib import Path
from typing import Any, Optional

# Dependencias propias.
from modulo_utilidades.config import settings as CONFIG
from .convertor_cordenadas_core import convert_bbox_image_to_world
from .procesador_anotaciones_coco_dataset_core import get_image_id_from_annotations

# Dependencias de terceros.
import pandas as pd
import geopandas as gpd
from fastkml import kml
from shapely.geometry import Point
from loguru import logger as LOGGER

# Configuraciones
DOWNLOAD_KMLS_FOLDER: Path = CONFIG.folders.download_kmls_folder
DOWNLOAD_GEOJSON_FOLDER: Path = CONFIG.folders.download_geojson_folder
CODIGO_EPSG_DEFAULT: str = CONFIG.georeferenciacion.codigo_epsg


def create_geojson_from_annotations(
    pic_name: str,
    coco_annotations: dict[str, Any],
    jgw_data: dict[str, Any],
    output_file_path: Path = DOWNLOAD_GEOJSON_FOLDER / "annotations.geojson",
    upload_to_drive: bool = False,
    epsg_code: str = CODIGO_EPSG_DEFAULT,
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
    if not coco_annotations:
        LOGGER.warning("No hay anotaciones en el archivo COCO.")
        return gpd.GeoDataFrame()

    # 1 - Configuraciones generales
    category_map = {cat["id"]: cat["name"] for cat in coco_annotations["categories"]}

    # 2 - Obtener el id de la imagen en las anotaciones
    image_id = get_image_id_from_annotations(pic_name, coco_annotations)

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
        confidence_raw = annotation.get("confidence", "Sin datos")
        confidence = str(round(float(confidence_raw), 2)) if confidence_raw != "Sin datos" else None

        # 5.1 - Convertir el bbox a coordenadas geográficas utilizando los datos del archivo JGW
        global_coordinates = convert_bbox_image_to_world(bbox, jgw_data)

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
                "confidence": confidence,
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

    if output_file_path:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_file_path, driver="GeoJSON")
        LOGGER.info(f"GeoJSON guardado en {output_file_path}")
        # 10 - Opcional: Subir a Google Drive si se solicita
        if upload_to_drive:
            # Aquí iría tu código para subir a Drive
            raise NotImplementedError("Subida a Google Drive no implementada.")
    return gdf


def generate_kml_from_geojson(
    gdf: gpd.GeoDataFrame,
    category_column: str = "name",
    target_category: Optional[str] = None,
    output_file_path: Optional[Path] = None,
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

    if output_file_path:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            k.write(output_file_path)
            LOGGER.info(f"Archivo KML guardado en {output_file_path}")
        except Exception as e:
            LOGGER.error(f"Error al guardar el archivo KML: {e}")
    return k
