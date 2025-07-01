import datetime
import numpy as np
import os, sys, yaml, shutil

sys.path.append(os.path.abspath("../../modulo-apps"))  # Se agrega modulo-mini-apps

from pathlib import Path
from typing import Any, Dict, List, Optional
from sklearn.model_selection import train_test_split

from utils.logging import Logging
from apps_config.settings import Config
from database_comunication.mongodb_client import MongoDB

from s3_comunication.procesador_s3 import ProcesadorS3
from tqdm import tqdm
import labeling.procesador_anotaciones_mongodb as ProcesadorCocoDataset

from ultralytics.engine.results import Results

CONFIG = Config().config_data
LOGGER = Logging().logger
DB = MongoDB().db


def split_yolo_dataset(dataset_path: Path, ratios=(0.7, 0.2, 0.1)) -> None:
    """
    Divide un dataset en formato YOLO en conjuntos de entrenamiento, validación y prueba,
    moviendo los archivos y eliminando la carpeta 'full'.

    Args:
        dataset_path (Path): Ruta al directorio principal del dataset YOLO.
        ratios (tuple): Tupla con los ratios para (entrenamiento, validación, prueba).
                       Por defecto es (0.7, 0.2, 0.1).
    """
    train_split, val_split, test_split = ratios
    if not (0 < train_split < 1 and 0 <= val_split < 1 and 0 <= test_split < 1):
        raise ValueError("Los ratios deben estar entre 0 y 1 y sumar 1.")
    ruta_imagenes_full = dataset_path / "images" / "full"
    ruta_etiquetas_full = dataset_path / "labels" / "full"

    if not ruta_imagenes_full.is_dir() or not ruta_etiquetas_full.is_dir():
        raise FileNotFoundError(
            f"No se encontraron las carpetas 'full' en {dataset_path / 'images'} o {dataset_path / 'labels'}. "
            "Asegúrate de que el dataset esté en el formato correcto."
        )

    nombres_imagenes = [f.name for f in ruta_imagenes_full.iterdir() if f.is_file()]
    nombres_etiquetas = [f.name for f in ruta_etiquetas_full.iterdir() if f.is_file()]

    # Asegurarse de que haya correspondencia entre imágenes y etiquetas (mismo nombre base)
    nombres_imagenes_base = {f.stem for f in ruta_imagenes_full.iterdir() if f.is_file()}
    nombres_etiquetas_base = {f.stem for f in ruta_etiquetas_full.iterdir() if f.is_file()}
    nombres_comunes = list(nombres_imagenes_base.intersection(nombres_etiquetas_base))

    if not nombres_comunes:
        LOGGER.warning(
            "No se encontraron pares de imágenes y etiquetas con nombres coincidentes en las carpetas 'full'."
        )
        return None

    # Dividir los nombres de los archivos en conjuntos de entrenamiento, validación y prueba
    train_nombres, temp_nombres = train_test_split(
        nombres_comunes, test_size=(val_split + test_split), random_state=CONFIG["seed"]
    )
    if test_split > 0:
        val_nombres, test_nombres = train_test_split(
            temp_nombres, test_size=test_split / (val_split + test_split), random_state=CONFIG["seed"]
        )
    else:
        val_nombres = temp_nombres
        test_nombres = []

    conjuntos = (
        {"train": train_nombres, "val": val_nombres, "test": test_nombres}
        if test_split > 0
        else {"train": train_nombres, "val": val_nombres}
    )
    subcarpetas = ["images", "labels"]

    # Crear las carpetas para los conjuntos divididos si no existen
    for subcarpeta in subcarpetas:
        for conjunto in conjuntos:
            (dataset_path / subcarpeta / conjunto).mkdir(parents=True, exist_ok=True)

    # Mover los archivos a sus respectivas carpetas
    total_files = sum(len(nombres) for nombres in conjuntos.values())
    with tqdm(total=total_files, desc="Moviendo archivos", unit="archivo") as pbar:
        for conjunto, nombres in conjuntos.items():
            for nombre_base in nombres:
                # Obtener el nombre de la imagen con su extensión
                resultados_glob_img = list(ruta_imagenes_full.glob(f"{nombre_base}.*"))
                nombre_imagen = ""
                if resultados_glob_img:
                    nombre_imagen = resultados_glob_img[0].name

                nombre_etiqueta = nombre_base + ".txt"

                ruta_imagen_origen = ruta_imagenes_full / nombre_imagen
                ruta_etiqueta_origen = ruta_etiquetas_full / nombre_etiqueta

                ruta_imagen_destino = dataset_path / "images" / conjunto / nombre_imagen
                ruta_etiqueta_destino = dataset_path / "labels" / conjunto / nombre_etiqueta

                if ruta_imagen_origen.exists():
                    shutil.move(str(ruta_imagen_origen), str(ruta_imagen_destino))
                if ruta_etiqueta_origen.exists():
                    shutil.move(str(ruta_etiqueta_origen), str(ruta_etiqueta_destino))
                pbar.update(1)

    # Eliminar las carpetas "full"
    try:
        shutil.rmtree(ruta_imagenes_full)
        LOGGER.debug(f"Carpeta eliminada: {ruta_imagenes_full}")
    except OSError as e:
        LOGGER.warning(f"Error al eliminar {ruta_imagenes_full}: {e}")

    try:
        shutil.rmtree(ruta_etiquetas_full)
        LOGGER.debug(f"Carpeta eliminada: {ruta_etiquetas_full}")
    except OSError as e:
        LOGGER.warning(f"Error al eliminar {ruta_etiquetas_full}: {e}")

    # Actualizar el archivo dataset.yaml
    ruta_yaml = dataset_path / "dataset.yaml"
    if ruta_yaml.exists():
        with open(ruta_yaml, "r") as f:
            data = yaml.safe_load(f)
        data["train"] = str(Path("images") / "train")
        data["val"] = str(Path("images") / "val")
        if "test" in data and test_split > 0:
            data["test"] = str(Path("images") / "test")
        if "full" in data:
            del data["full"]
        with open(ruta_yaml, "w") as f:
            yaml.dump(data, f)
        LOGGER.debug(
            f"Archivo {ruta_yaml} actualizado con las rutas de train, val y test, y se eliminó la clave 'path' si era necesario."
        )
    else:
        LOGGER.warning(f"Advertencia: No se encontró el archivo {ruta_yaml}. Deberás actualizar las rutas manualmente.")

    LOGGER.info("División del dataset YOLO completada (archivos movidos y carpetas 'full' eliminadas).")
    return None


def filter_results_by_confidence(
    results: List[Dict[str, Any]],
    min_confidence: float = 0.5,
) -> Results:
    """
    Filtra los resultados de detección de objetos para eliminar aquellas detecciones
    con una confianza inferior al umbral especificado.

    Args:
        results (List[Dict[str, Any]]): Lista de resultados de detección, donde cada resultado
                                        es un diccionario que contiene información sobre las
                                        cajas delimitadoras, categorías y confianza.
        min_confidence (float): Umbral mínimo de confianza para filtrar las detecciones.
                                Por defecto es 0.5.

    Returns:
        Results: Resultados filtrados que cumplen con el umbral de confianza.
    """
    if len(results) > 1:
        raise ValueError(
            "Se detectó un resultado para varias imágenes. Asegúrate de que solo haya el resultado de una imagen."
        )
    result = results[0]

    # Filtrar los índices según el umbral
    filtered_indices = [i for i, conf in enumerate(result.boxes.conf) if conf.item() >= min_confidence]

    # Filtrar las cajas y otros atributos
    filtered_boxes = result.boxes[filtered_indices]

    # Crear un nuevo objeto Results con las mismas propiedades pero solo con las detecciones filtradas
    results_filtered = Results(
        orig_img=result.orig_img,
        path=result.path,
        names=result.names,
    )
    results_filtered.boxes = filtered_boxes
    results_filtered.orig_shape = result.orig_shape
    results_filtered.speed = result.speed
    results_filtered.save_dir = result.save_dir

    return [results_filtered]


def create_coco_annotations_from_yolo_result(
    results: List[Dict[str, Any]],
    pic_name: str,
) -> Dict[str, Any]:
    """
    Convierte los resultados de detección de objetos en formato YOLO a anotaciones en formato COCO.

    Esta función toma una imagen y sus resultados de detección, y genera un diccionario
    que sigue la estructura del formato COCO. Se espera que los resultados contengan
    información sobre las cajas delimitadoras, categorías y confianza de las detecciones.
    Solamente se procesa una imagen a la vez.

    Args:
        image (np.ndarray): Imagen en formato numpy array sobre la que se realizó la detección.
        results (Dict[str, Any]): Resultados de la detección de objetos, típicamente una lista de predicciones YOLO.
        pic_name (str): Nombre base del archivo de la imagen (sin extensión).

    Raises:
        ValueError: Si los resultados contienen detecciones para más de una imagen.

    Returns:
        Dict[str, Any]: Diccionario con las anotaciones en formato COCO, incluyendo info, licenses, categories, images y annotations.
    """
    bboxes_result = results.boxes.cpu()
    names = results.names
    coco_annotations = {
        "info": CONFIG["coco_dataset"]["info"],
        "licenses": CONFIG["coco_dataset"]["licenses"],
        "categories": CONFIG["coco_dataset"]["categories"],
        "images": [],
        "annotations": [],
    }

    category_map = {cat["name"]: cat["id"] for cat in coco_annotations["categories"]}
    image_height, image_width = results.orig_shape

    image = {
        "id": 1,
        "width": image_width,
        "height": image_height,
        "file_name": f"{pic_name}.jpg",
        "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    coco_annotations["images"] = [image]

    if not results.boxes:
        LOGGER.warning("No se encontraron resultados de detección de objetos.")
        return coco_annotations

    annotations = []
    for index, bbox_result in enumerate(bboxes_result):
        id = index + 1
        category_name = names[bbox_result.cls.int().item()]
        category_id = category_map.get(category_name, None)
        if category_id is None:
            LOGGER.warning(f"Categoría '{category_name}' no encontrada en el mapa de categorías.")
            continue

        x_min, y_min, x_max, y_max = bbox_result.xyxy[0].numpy()
        ancho = x_max - x_min
        alto = y_max - y_min
        area = ancho * alto

        conf = bbox_result.conf[0].numpy()

        # Casteamos a float para evitar problemas de serialización
        conf = float(conf)
        x_min, y_min, ancho, alto = map(float, [x_min, y_min, ancho, alto])
        area = float(area)

        annotation = {
            "id": id,
            "image_id": image["id"],
            "category_id": category_id,
            "bbox": [x_min, y_min, ancho, alto],
            "area": area,
            "iscrowd": 0,
            "attributes": {
                "occluded": False,
                "rotation": 0.0,
            },
            "confidence": conf,
        }
        annotations.append(annotation)

    coco_annotations["annotations"] = annotations

    return coco_annotations
