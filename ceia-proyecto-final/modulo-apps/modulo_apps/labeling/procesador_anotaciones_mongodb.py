import copy, datetime, json

from pathlib import Path
from typing import Any, Optional

from pymongo import UpdateOne

import typer

from loguru import logger as LOGGER
from modulo_apps.config import config as CONFIG
from modulo_apps.database_comunication.mongodb_client import mongodb as DB
from modulo_apps.utils.types import AnnotationType

import modulo_apps.labeling.procesador_anotaciones_coco_dataset as CocoDatasetUtils
import modulo_apps.labeling.convertor_cordenadas as ConvertorCoordenadas
import modulo_apps.labeling.procesador_anotaciones_cvat as ProcesadorAnotacionesCVAT

MINIO_PATCHES_PATH = CONFIG.minio.paths.patches
DOWNLOAD_COCO_ANNOTATIONS_FOLDER = CONFIG.folders.download_coco_annotations_folder
DOWNLOAD_JGW_FOLDER = CONFIG.folders.download_jgw_folder

COCO_DATASET_DATA = CONFIG.coco_dataset.to_dict()
COCO_DATASET_CATEGORIES = CONFIG.coco_dataset.categories

app = typer.Typer()


@app.command()
def test_connection():
    """
    Test the connection to the MongoDB database.
    """
    try:
        DB.command("ping")
        LOGGER.debug("Conexión a la base de datos MongoDB exitosa.")
    except Exception as e:
        LOGGER.error(f"Error al conectar a la base de datos MongoDB: {e}")
        raise e


def save_coco_annotations(
    coco_annotations: dict[str, Any],
    field_name: str,
    annotation_type: AnnotationType = "cvat",
) -> bool:
    """Guarda las anotaciones en la base de datos MongoDB. Las anotaciones se guardan
    en los parches correspondientes, por más que se pase una anotaciones de una imagen completa.
    Se eliminan las anotaciones existentes para cada parche antes de guardar las nuevas.

    Este método procesa las anotaciones en formato COCO y las guarda en la base de datos MongoDB
    asociándolas a sus parches correspondientes. Se eliminan las anotaciones existentes
    para cada parche antes de guardar las nuevas. Las anotaciones se guardan en un campo específico
    de la base de datos, cuyo nombre se pasa como argumento.

    Se espera que las anotaciones sean todos parches o todas imágenes completas. No una combinación de ambos.
    Si se pasan anotaciones de imágenes completas

    Args:
        annotations (dict[str, any]): Diccionario con las anotaciones en formato COCO.
        field_name (str): Nombre del campo donde se guardarán las anotaciones en la base de datos.

    Returns:
        bool: True si las anotaciones se guardaron correctamente, False en caso contrario.
    """
    images = coco_annotations["images"]
    annotations = coco_annotations["annotations"]
    categories = coco_annotations["categories"]

    category_map = {cat["id"]: cat["name"] for cat in categories}

    image_patch_pairs = []
    upsert_operations = []

    if annotation_type == "images":
        images, annotations = ProcesadorAnotacionesCVAT.convert_image_annotations_to_cvat_annotations(images, annotations)
    elif annotation_type == "patches":
        images, annotations = ProcesadorAnotacionesCVAT.convert_patch_annotations_to_cvat_annotations(images, annotations)

    for image in images:
        file_name = image["file_name"]
        image_id = image["id"]

        # Obtener el nombre de la imagen y el parche
        # Ejemplo de file_name: "patches/Barrio17metros_20231212_dji_rtk_pc_5cm/Barrio17metros_20231212_dji_rtk_pc_5cm_patch_4.jpg"
        # imagen: quedarme con la subcarpeta de la imagen
        # parche: quedarme con el ultimo elemento del path y quitar la extension
        image_name = file_name.split("/")[-2]
        patch_name = file_name.split("/")[-1].split(".")[0]

        # Preparar los datos filtrados sin image_id
        patch_annotations = [
            {k: v for k, v in copy.deepcopy(ann).items() if k != "image_id"}
            for ann in annotations
            if ann["image_id"] == image_id
        ]

        if not patch_annotations:
            LOGGER.debug(f"No se encontraron anotaciones para la imagen {file_name}.")
        else:
            LOGGER.debug(f"Se encontraron {len(patch_annotations)} anotaciones para {file_name}.")

            # Mapear las categorías a los nombres correspondientes
            patch_annotations = [
                {**ann, "category_name": category_map[ann["category_id"]]} for ann in patch_annotations
            ]

            # Eliminar la clave category_id de las anotaciones
            patch_annotations = [{k: v for k, v in ann.items() if k != "category_id"} for ann in patch_annotations]

            # Crear operación de actualización para MongoDB
            upsert_operations.append(
                UpdateOne(
                    {"id": image_name, "patches.patch_name": patch_name},
                    {
                        "$set": {
                            # Actualizar anotaciones en el patch específico
                            f"patches.$.{field_name}_annotations": patch_annotations,
                            # Actualizar la fecha de modificación a hoy
                            f"patches.$.last_modified": datetime.datetime.now(),
                        }
                    },
                    upsert=True,
                )
            )

            image_patch_pairs.append((image_name, patch_name))

    # Proceder con las operaciones en la base de datos
    if image_patch_pairs:
        imagenes = DB.get_collection("imagenes")

        # Primero, para cada par de imagen/parche, eliminar las anotaciones existentes
        for image_name, patch_name in image_patch_pairs:
            # Utilizar arrayFilters para actualizar sólo el elemento específico del array
            imagenes.update_one(
                {"id": image_name, "patches.patch_name": patch_name},
                {"$set": {f"patches.$.{field_name}_annotations": []}},
            )

        LOGGER.info(f"Se eliminaron las anotaciones de {len(image_patch_pairs)} imágenes/parches.")

        # Luego realizar las operaciones de actualización/inserción
        if upsert_operations:
            LOGGER.info(f"Se van a ejecutar {len(upsert_operations)} operaciones de actualización.")
            result = imagenes.bulk_write(upsert_operations, ordered=False)
            LOGGER.info(f"Documentos modificados: {result.modified_count}")
        return True
    else:
        LOGGER.warning("No se encontraron pares de imagen/parche para actualizar.")
        return False


def _create_patch_fields(
    db_image: dict[str, Any],
    field_name: str,
    category_map: dict[str, Any],
    patch_name: str,
) -> Optional[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """
    Crea anotaciones en formato COCO para un parche específico.

    Este método toma las anotaciones de un parche almacenadas en la base de datos,
    las procesa y las convierte al formato COCO, incluyendo las categorías y la
    información de la imagen asociada al parche.

    Args:
        db_image (dict[str, Any]): Documento de la imagen en la base de datos que contiene
                                información sobre los parches y sus anotaciones.
        field_name (str): Nombre del campo en la base de datos que contiene las anotaciones.
        category_map (dict[str, Any]): Mapeo de nombres de categorías a sus IDs en formato COCO.
        patch_name (str): Nombre del parche cuyas anotaciones se desean procesar.

    Returns:
        Optional[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
            Una tupla que contiene:
            - Una lista de anotaciones en formato COCO.
            - Una lista con la información de la imagen asociada al parche.
            Devuelve None si no se encuentran anotaciones para el parche.
    """
    patch = next((p for p in db_image.get("patches", []) if p["patch_name"] == patch_name), None)

    annotations = patch.get(f"{field_name}_annotations", [])
    if not annotations:
        LOGGER.warning(f"No se encontraron anotaciones para el parche {patch_name}.")

    image = {
        "id": 1,
        "width": patch["width"],
        "height": patch["height"],
        "file_name": f"{patch_name}.jpg",
        "date_captured": patch.get("last_modified", db_image["date_captured"]).strftime("%Y-%m-%d %H:%M:%S"),
    }

    images = [image]

    # Mapeamos las categorías a los nombres correspondientes
    annotations = [
        {
            **ann,
            "category_id": category_map[ann["category_name"]],
            "image_id": image["id"],
        }
        for ann in annotations
    ]

    # Eliminamos la clave category_name de las anotaciones
    annotations = [{k: v for k, v in ann.items() if k != "category_name"} for ann in annotations]

    return annotations, images


def _create_images_fields(
    db_image: dict[str, Any],
    field_name: str,
    category_map: dict[str, Any],
    image_name: str,
) -> Optional[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """
    Crea anotaciones en formato COCO para una imagen específica.

    Este método toma las anotaciones de los parches asociados a una imagen almacenada
    en la base de datos, las procesa y las convierte al formato COCO, incluyendo las
    categorías y la información de la imagen.

    Args:
        db_image (dict[str, Any]): Documento de la imagen en la base de datos que contiene
                                información sobre los parches y sus anotaciones.
        field_name (str): Nombre del campo en la base de datos que contiene las anotaciones.
        category_map (dict[str, Any]): Mapeo de nombres de categorías a sus IDs en formato COCO.
        image_name (str): Nombre de la imagen cuyas anotaciones se desean procesar.

    Returns:
        Optional[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
            Una tupla que contiene:
            - Una lista de anotaciones en formato COCO.
            - Una lista con la información de la imagen asociada.
            Devuelve None si no se encuentran anotaciones para la imagen.
    """
    image = {
        "id": 1,
        "width": db_image["width"],
        "height": db_image["height"],
        "file_name": f"{db_image["id"]}.jpg",
        "date_captured": db_image["date_captured"].strftime("%Y-%m-%d %H:%M:%S"),
    }

    images = [image]

    # 2 - Obtener los nombres de los parches asociados a la imagen
    db_patches = db_image.get("patches", [])
    annotations = []

    # 3 - Para cada annotación, mapeamos los bboxes a bboxes de la imagen
    for patch in db_patches:
        patch_annotations = patch.get(f"{field_name}_annotations", [])
        if not patch_annotations:
            LOGGER.debug(f"No se encontraron anotaciones para el parche {patch['patch_name']}.")
            continue

        # Mapeamos las categorías a los nombres correspondientes
        # Agregamos el campo image_id y modificamos el id de la anotación
        # Convertimos los bboxes del parche a la imagen
        patch_annotations = [
            {
                **ann,
                "category_id": category_map[ann["category_name"]],
                "image_id": image["id"],
                "bbox": ConvertorCoordenadas.convert_bbox_patch_to_image(
                    ann["bbox"], patch["x_start"], patch["y_start"]
                ),
            }
            for i, ann in enumerate(patch_annotations)
        ]

        # Eliminamos la clave category_name de las anotaciones
        patch_annotations = [{k: v for k, v in ann.items() if k != "category_name"} for ann in patch_annotations]

        # Agregamos las anotaciones a la lista de anotaciones
        annotations.extend(patch_annotations)

    # Agregamos el id del parche a cada anotación
    # Se podría mejorar esto agregándolo en el for, utilizando un generador para mejorar
    # la performance.
    annotations = [{**ann, "id": i + 1} for i, ann in enumerate(annotations)]

    return annotations, images


def _download_annotation_as_coco_from_mongodb(
    field_name: str,
    patch_name: Optional[str] = None,
    image_name: Optional[str] = None,
    output_filename: Optional[Path] = None,
) -> Optional[Path]:
    """
    Descarga anotaciones en formato COCO desde MongoDB.

    Este método permite descargar anotaciones almacenadas en MongoDB en formato COCO,
    ya sea para un parche específico o para una imagen completa. Las anotaciones se
    guardan en un archivo JSON en la ubicación especificada o en una carpeta predeterminada.

    Args:
        field_name (str): Nombre del campo en la base de datos que contiene las anotaciones.
        patch_name (Optional[str], optional): Nombre del parche cuyas anotaciones se desean descargar.
                                            Si se proporciona, se descargan las anotaciones del parche.
                                            Defaults to None.
        image_name (Optional[str], optional): Nombre de la imagen cuyas anotaciones se desean descargar.
                                            Si se proporciona, se descargan las anotaciones de la imagen.
                                            Defaults to None.
        output_filename (Optional[Path], optional): Ruta del archivo donde se guardarán las anotaciones.
                                                    Si no se proporciona, se utiliza una ruta predeterminada.
                                                    Defaults to None.

    Raises:
        ValueError: Si no se proporciona ni `patch_name` ni `image_name`.
        ValueError: Si no se encuentra la imagen o el parche en la base de datos.

    Returns:
        Optional[Path]: Ruta del archivo JSON generado con las anotaciones en formato COCO.
                        Devuelve None si no se encuentran anotaciones.
    """
    if bool(patch_name is None) == bool(image_name is None):  # xor
        raise ValueError("Se debe proporcionar un patch_name o un image_name.")

    category_map = {cat.name: cat.id for cat in COCO_DATASET_CATEGORIES}

    db_images = DB.get_collection("imagenes")
    db_image = (
        db_images.find_one({"patches.patch_name": patch_name}) if patch_name else db_images.find_one({"id": image_name})
    )
    if not db_image:
        msg = (
            f"No se encontró la imagen {image_name} en la base de datos."
            if image_name
            else f"No se encontró la imagen con el parche {patch_name} en la base de datos."
        )
        raise ValueError(msg)

    annotations, images = (
        _create_patch_fields(db_image, field_name, category_map, patch_name)
        if patch_name
        else _create_images_fields(db_image, field_name, category_map, image_name)
    )
    if not annotations:
        LOGGER.debug(f"No se encontraron anotaciones para el parche {patch_name}.")

    coco_annotations = {
        "info": COCO_DATASET_DATA["info"],
        "licenses": COCO_DATASET_DATA["licenses"],
        "categories": COCO_DATASET_DATA["categories"],
        "images": images,
        "annotations": annotations,
    }

    if not output_filename:
        DOWNLOAD_COCO_ANNOTATIONS_FOLDER.mkdir(parents=True, exist_ok=True)
        output_filename = DOWNLOAD_COCO_ANNOTATIONS_FOLDER / (
            f"{field_name}_{patch_name}_annotations.json"
            if patch_name
            else f"{field_name}_{image_name}_annotations.json"
        )

    with open(output_filename, "w") as f:
        json.dump(coco_annotations, f, indent=4)
        LOGGER.debug(f"Archivo JSON guardado en {output_filename}.")

    return output_filename


def _load_coco_annotation_from_mongodb(
    field_name: str,
    patch_name: Optional[str] = None,
    image_name: Optional[str] = None,
    clean_files: bool = True,
) -> Optional[dict[str, Any]]:
    """
    Carga anotaciones desde MongoDB en formato COCO.

    Este método permite descargar y cargar anotaciones almacenadas en MongoDB,
    ya sea para un parche específico o para una imagen completa. Las anotaciones
    se descargan en formato JSON y se cargan en un diccionario.

    Args:
        field_name (str): Nombre del campo en la base de datos que contiene las anotaciones.
        patch_name (Optional[str], optional): Nombre del parche cuyas anotaciones se desean cargar.
                                            Si se proporciona, se cargan las anotaciones del parche.
                                            Defaults to None.
        image_name (Optional[str], optional): Nombre de la imagen cuyas anotaciones se desean cargar.
                                            Si se proporciona, se cargan las anotaciones de la imagen.
                                            Defaults to None.
        clean_files (bool, optional): Si se deben eliminar los archivos temporales después de la carga.
                                    Defaults to True.

    Raises:
        ValueError: Si no se proporciona ni `patch_name` ni `image_name`.
        Exception: Si ocurre un error al descargar las anotaciones.
        FileNotFoundError: Si no se encuentra el archivo de anotaciones descargado.

    Returns:
        Optional[dict[str, Any]]: Diccionario con las anotaciones cargadas en formato COCO.
                                Devuelve None si no se encuentran anotaciones.
    """
    if bool(patch_name is None) == bool(image_name is None):
        raise ValueError("Se debe proporcionar un patch_name o un image_name.")
    try:
        file_path = (
            _download_annotation_as_coco_from_mongodb(field_name=field_name, patch_name=patch_name)
            if patch_name
            else _download_annotation_as_coco_from_mongodb(field_name=field_name, image_name=image_name)
        )
    except Exception as e:
        raise Exception(f"Error al descargar las anotaciones: {e}")

    if file_path:
        annotations = CocoDatasetUtils.load_annotations_from_path(file_path)
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


def _create_patches_coco_annotations(
    field_name: str, patches_names: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Crea anotaciones en formato COCO para una lista de parches.

    Este método toma las anotaciones de una lista de parches almacenadas en MongoDB,
    las procesa y las convierte al formato COCO, incluyendo las categorías y la
    información de las imágenes asociadas a los parches.

    Args:
        field_name (str): Nombre del campo en la base de datos que contiene las anotaciones.
        patches_names (list[str]): Lista de nombres de los parches cuyas anotaciones se desean procesar.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            Una tupla que contiene:
            - Una lista de anotaciones en formato COCO.
            - Una lista con la información de las imágenes asociadas a los parches.
    """
    images = []
    annotations = []
    for id, patch_name in enumerate(patches_names):
        image_id = id + 1
        patch_annotations = _load_coco_annotation_from_mongodb(field_name=field_name, patch_name=patch_name)

        if not patch_annotations:
            LOGGER.debug(f"No se encontraron anotaciones para el parche {patch_name}.")

        patch_annotations["images"][0]["id"] = image_id
        patch_annotations["annotations"] = [{**ann, "image_id": image_id} for ann in patch_annotations["annotations"]]

        # Agregar las imágenes y anotaciones al diccionario principal
        images.extend(patch_annotations["images"])
        annotations.extend(patch_annotations["annotations"])

        LOGGER.debug(f"Anotaciones del parche {patch_name} agregadas correctamente.")

    return annotations, images


def _create_images_coco_annotations(
    field_name: str, images_names: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Crea anotaciones en formato COCO para una lista de imágenes.

    Este método toma las anotaciones de una lista de imágenes almacenadas en MongoDB,
    las procesa y las convierte al formato COCO, incluyendo las categorías y la
    información de las imágenes asociadas.

    Args:
        field_name (str): Nombre del campo en la base de datos que contiene las anotaciones.
        images_names (list[str]): Lista de nombres de las imágenes cuyas anotaciones se desean procesar.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            Una tupla que contiene:
            - Una lista de anotaciones en formato COCO.
            - Una lista con la información de las imágenes asociadas.
    """
    images = []
    annotations = []
    for id, image_name in enumerate(images_names):
        image_id = id + 1
        image_annotations = _load_coco_annotation_from_mongodb(field_name=field_name, image_name=image_name)

        if not image_annotations["annotations"]:
            LOGGER.debug(f"No se encontraron anotaciones para la imagen {image_name}.")

        image_annotations["images"][0]["id"] = image_id
        image_annotations["annotations"] = [{**ann, "image_id": image_id} for ann in image_annotations["annotations"]]

        # Agregar las imágenes y anotaciones al diccionario principal
        images.extend(image_annotations["images"])
        annotations.extend(image_annotations["annotations"])

        LOGGER.debug(f"Anotaciones de la imagen {image_name} agregadas correctamente.")

    return annotations, images


@app.command()
def download_annotations_as_coco_from_mongodb(
    field_name: str = "cvat",
    patches_names: list[str] = None,
    images_names: list[str] = None,
    output_filename: Optional[Path] = None,
) -> Path:
    """
    Descarga anotaciones en formato COCO desde MongoDB.

    Este método permite descargar anotaciones almacenadas en MongoDB en formato COCO,
    ya sea para una lista de parches o para una lista de imágenes. Las anotaciones se
    guardan en un archivo JSON en la ubicación especificada o en una carpeta predeterminada.

    Args:
        field_name (str): Nombre del campo en la base de datos que contiene las anotaciones.
        patches_names (list[str], optional): Lista de nombres de los parches cuyas anotaciones se desean descargar.
                                            Si se proporciona, se descargan las anotaciones de los parches.
                                            Defaults to None.
        images_names (list[str], optional): Lista de nombres de las imágenes cuyas anotaciones se desean descargar.
                                            Si se proporciona, se descargan las anotaciones de las imágenes.
                                            Defaults to None.
        output_filename (Optional[Path], optional): Ruta del archivo donde se guardarán las anotaciones.
                                                    Si no se proporciona, se utiliza una ruta predeterminada.
                                                    Defaults to None.

    Raises:
        ValueError: Si no se proporciona ni `patches_names` ni `images_names`.

    Returns:
        Path: Ruta del archivo JSON generado con las anotaciones en formato COCO.
    """
    if bool(patches_names is None) == bool(images_names is None):  # xor
        raise ValueError("Se debe proporcionar una lista de nombres de parches o imágenes.")

    annotations, images = (
        _create_patches_coco_annotations(field_name, patches_names)
        if patches_names
        else _create_images_coco_annotations(field_name, images_names)
    )

    coco_annotations = {
        "info": COCO_DATASET_DATA["info"],
        "licenses": COCO_DATASET_DATA["licenses"],
        "categories": COCO_DATASET_DATA["categories"],
        "images": images,
        "annotations": annotations,
    }

    if not output_filename:
        DOWNLOAD_COCO_ANNOTATIONS_FOLDER.mkdir(parents=True, exist_ok=True)
        output_filename = DOWNLOAD_COCO_ANNOTATIONS_FOLDER / (
            f"{field_name}_patches_annotations.json" if patches_names else f"{field_name}_images_annotations.json"
        )

    with open(output_filename, "w") as f:
        json.dump(coco_annotations, f, indent=4)
        LOGGER.debug(f"Archivo JSON guardado en {output_filename}.")

    return output_filename


def load_coco_annotations_from_mongodb(
    field_name: str = "cvat",
    patches_names: Optional[list[str]] = None,
    images_names: Optional[list[str]] = None,
    clean_files: bool = True,
) -> dict[str, Any]:
    if bool(patches_names is None) == bool(images_names is None):  # xor
        raise ValueError("Se debe proporcionar una lista de nombres de parches o imágenes.")

    try:
        file_path = (
            download_annotations_as_coco_from_mongodb(field_name=field_name, patches_names=patches_names)
            if patches_names
            else download_annotations_as_coco_from_mongodb(field_name=field_name, images_names=images_names)
        )
    except Exception as e:
        raise Exception(f"Error al descargar las anotaciones: {e}")

    if file_path:
        annotations = CocoDatasetUtils.load_annotations_from_path(file_path)
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


def load_jgw_file_from_mongodb(
    image_name: Optional[str] = None,
    patch_name: Optional[str] = None,
    should_download: bool = False,
    output_filename: Path = DOWNLOAD_JGW_FOLDER / "jgw_data.json",
) -> Optional[dict[str, Any]]:
    """
    Carga un archivo JGW desde MongoDB.

    Este método busca y carga un archivo JGW asociado a una imagen o parche específico
    almacenado en la base de datos MongoDB. Si se proporciona el nombre de un parche,
    se busca el archivo jgw de la imagen y se hace la conversión al parche. Si se proporciona el nombre
    de una imagen, se busca el archivo JGW asociado a esa imagen.

    Args:
        image_name (Optional[str], optional): Nombre de la imagen cuya información JGW se desea cargar.
                                            Defaults to None.
        patch_name (Optional[str], optional): Nombre del parche cuya información JGW se desea cargar.
                                            Defaults to None.

    Raises:
        ValueError: Si no se proporciona ni `image_name` ni `patch_name`.

    Returns:
        Optional[dict[str, Any]]: Diccionario con la información del archivo JGW.
                                Devuelve None si no se encuentra el archivo.
    """
    if bool(image_name is None) == bool(patch_name is None):  # xor
        raise ValueError("Se debe proporcionar un image_name o un patch_name.")

    db_images = DB.get_collection("imagenes")
    db_image = (
        db_images.find_one({"patches.patch_name": patch_name}) if patch_name else db_images.find_one({"id": image_name})
    )
    if not db_image:
        msg = (
            f"No se encontró la imagen {image_name} en la base de datos."
            if image_name
            else f"No se encontró la imagen con el parche {patch_name} en la base de datos."
        )
        raise ValueError(msg)

    image_jgw_data = jgw_data = db_image.get("jgw_data", None)
    if not image_jgw_data:
        LOGGER.warning(f"No se encontró el archivo JGW para la imagen {image_name}.")

    if image_name:
        jgw_data = image_jgw_data
    else:
        # Si se proporciona un nombre de parche, debemos convertir el jgw_data a las coordenadas del parche
        patch = next((p for p in db_image.get("patches", []) if p["patch_name"] == patch_name), None)
        if not patch:
            raise ValueError(f"No se encontró el parche {patch_name} en la imagen {image_name}.")

        # Convertir el jgw_data a las coordenadas del parche
        x_origin_patch = image_jgw_data["x_origin"] + (patch["x_start"] * image_jgw_data["x_pixel_size"])
        y_origin_patch = image_jgw_data["y_origin"] + (patch["y_start"] * image_jgw_data["y_pixel_size"])

        jgw_data = {
            "x_pixel_size": image_jgw_data["x_pixel_size"],
            "y_rotation": image_jgw_data["y_rotation"],
            "x_rotation": image_jgw_data["x_rotation"],
            "y_pixel_size": image_jgw_data["y_pixel_size"],
            "x_origin": x_origin_patch,
            "y_origin": y_origin_patch,
        }

    if should_download:
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filename, "w") as f:
            json.dump(jgw_data, f, indent=4)
            LOGGER.success(f"Archivo JGW guardado en {output_filename}.")

    return jgw_data


def list_patches_w_ann_from_mongodb(field_name: str = "cvat") -> list[str]:
    """
    Lista los nombres de los patches que tienen anotaciones en el campo especificado.

    Args:
        field_name (str, optional): Nombre base del campo de anotaciones a buscar en cada patch.
            Por defecto es "cvat", lo que buscará el campo "cvat_annotations".

    Returns:
        list: Lista de nombres de patches que contienen anotaciones en el campo especificado.
    """
    imagenes = DB.get_collection("imagenes")
    field_name += "_annotations"
    pipeline = [
        # Descomponer el array de patches en documentos individuales
        {"$unwind": "$patches"},
        # Filtrar solo los patches que tienen el campo cvat_annotations
        # y que este campo tiene al menos un elemento
        {"$match": {f"patches.{field_name}": {"$exists": True, "$ne": []}}},
        # Proyectar solo el nombre del patch
        {"$project": {"patch_name": "$patches.patch_name", "_id": 0}},
    ]

    # Ejecutar el pipeline de agregación
    resultados = list(imagenes.aggregate(pipeline))

    # Extraer solo los nombres de los patches
    patch_names = [doc["patch_name"] for doc in resultados]

    return patch_names


def list_images_w_ann_from_mongodb(field_name: str = "cvat") -> list[str]:
    """
    Lista los nombres de las imágenes que tienen anotaciones en el campo especificado.

    Args:
        field_name (str, optional): Nombre base del campo de anotaciones a buscar en cada imagen.
            Por defecto es "cvat", lo que buscará el campo "cvat_annotations".

    Returns:
        list: Lista de nombres de imágenes que contienen anotaciones en el campo especificado.
    """
    imagenes = DB.get_collection("imagenes")
    annotation_field = f"patches.{field_name}_annotations"
    pipeline = [
        # Desenrollar el array de patches para poder filtrar por cada uno
        {"$unwind": "$patches"},
        # Filtrar los parches que tienen el campo de anotaciones y que no está vacío
        {"$match": {annotation_field: {"$exists": True, "$ne": []}}},
        # Agrupar por el _id de la imagen original para obtener imágenes únicas
        {"$group": {"_id": "$id"}},
        # Proyectar solo el nombre de la imagen
        {"$project": {"image_name": "$_id", "_id": 0}},
    ]

    # Ejecutar el pipeline de agregación
    resultados = list(imagenes.aggregate(pipeline))

    # Extraer solo los nombres de las imágenes
    image_names = [doc["image_name"] for doc in resultados]

    return image_names

def list_images_from_mongodb(field_name: str = "cvat") -> list[str]:
    """
    Lista los nombres de las imágenes que tienen el campo especificado.

    Args:
        field_name (str, optional): Nombre base del campo a buscar en cada imagen.
            Por defecto es "cvat", lo que buscará el campo "cvat_annotations".

    Returns:
        list: Lista de nombres de imágenes que contienen el campo especificado.
    """
    raise NotImplementedError("TODO: Función no implementada aún.")

    return image_names


def get_patch_metadata_from_mongodb(patch_name: str) -> Optional[dict[str, Any]]:
    """
    Obtiene la metadata de un parche específico desde MongoDB.

    Args:
        patch_name (str): Nombre del parche cuya metadata se desea obtener.

    Returns:
        Optional[dict[str, Any]]: Diccionario con la metadata del parche.
                                Devuelve None si no se encuentra el parche.
    """
    imagenes = DB.get_collection("imagenes")
    db_image = imagenes.find_one({"patches.patch_name": patch_name})
    if not db_image:
        LOGGER.warning(f"No se encontró el parche {patch_name} en la base de datos.")
        return None

    # Obtener la metadata del parche
    patch_metadata = next((p for p in db_image.get("patches", []) if p["patch_name"] == patch_name), None)
    if not patch_metadata:
        LOGGER.warning(f"No se encontró la metadata para el parche {patch_name}.")
        return None

    return patch_metadata

if __name__ == "__main__":
    app()
