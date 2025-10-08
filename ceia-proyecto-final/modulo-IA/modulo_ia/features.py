from collections import Counter, defaultdict
from pathlib import Path
import shutil
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
from pydash import sample
import random, yaml

from loguru import logger as LOGGER
from modulo_ia.config import config as CONFIG

import cv2  # Debe venir después de la importación del config por configuraciones de variables de entorno.

import typer

from tqdm import tqdm
import typer

from modulo_ia.utils.types import DatasetFormat
import modulo_utilidades.utils.helpers as Helpers

import fiftyone as fo
from fiftyone import ViewField as F

from deprecated import deprecated

RAW_DATA_FOLDER = CONFIG.folders.raw_data_folder
EXTERNAL_DATA_FOLDER = CONFIG.folders.external_data_folder
INTERIM_DATA_FOLDER = CONFIG.folders.interim_data_folder
PROCESSED_DATA_FOLDER = CONFIG.folders.processed_data_folder
TEMP_DATA_FOLDER = CONFIG.folders.temp_data_folder

app = typer.Typer()


@app.command()
def crop_dataset(
    dataset_path: Path,
    dataset_format: DatasetFormat = DatasetFormat.YOLO,
    crop_size: int = 640,
    overlap: int = 250,
    threshold: float = 0.2,
) -> None:
    """
    Recorta imágenes de un dataset y ajusta las anotaciones correspondientes.
    Esta función toma un dataset en formato YOLO y genera recortes (crops) de las imágenes
    originales junto con sus anotaciones ajustadas. Los recortes se generan con un tamaño
    específico y pueden tener solapamiento entre ellos.
    Args:
        dataset_path (Path): Ruta al directorio del dataset que contiene las carpetas
            'images' y 'labels'.
        dataset_format (DatasetFormat, optional): Formato del dataset. Por defecto
            DatasetFormat.YOLO. Actualmente solo soporta formato YOLO.
        crop_size (int, optional): Tamaño en píxeles de cada recorte (cuadrado).
            Por defecto 640.
        overlap (int, optional): Solapamiento en píxeles entre recortes adyacentes.
            Por defecto 250.
        threshold (float, optional): Umbral mínimo de área de una anotación que debe
            estar presente en el recorte para ser incluida. Valor entre 0.0 y 1.0.
            Por defecto 0.2.
    Returns:
        None: La función no retorna valores, pero genera los recortes y anotaciones
        en el sistema de archivos.
    Raises:
        NotImplementedError: Si se especifica un formato de dataset diferente a YOLO.
    Examples:
        >>> from pathlib import Path
        >>> dataset_path = Path("/ruta/al/dataset")
        >>> crop_dataset(dataset_path, crop_size=512, overlap=100, threshold=0.3)
        >>> # Usar parámetros por defecto
        >>> crop_dataset(Path("./mi_dataset"))
    Notes:
        - La función busca imágenes en formato JPG y PNG en la carpeta 'images/full'.
        - Las anotaciones deben estar en formato YOLO en la carpeta 'labels/full'.
        - Los recortes generados mantienen la estructura de directorios original.
        - Si una anotación no cumple con el threshold de área mínima, se excluye
          del recorte correspondiente.
    """
    if not dataset_path.exists():
        LOGGER.error(f"El dataset {dataset_path} no existe.")
        return

    if dataset_format != DatasetFormat.YOLO:
        raise NotImplementedError(
            f"El formato de dataset {dataset_format} no está implementado para el recorte de imágenes."
        )

    images_dir = dataset_path / "images" / "full"
    labels_dir = dataset_path / "labels" / "full"

    if not images_dir.exists() or not labels_dir.exists():
        LOGGER.error(f"Las carpetas 'images' o 'labels' no se encontraron en {dataset_path}.")
        return

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    LOGGER.info(f"Procesando {len(image_files)} imágenes en {images_dir}")

    cant_crops = 0
    for image_file in tqdm(image_files, desc="Recortando imágenes y ajustando anotaciones"):
        cant_crops += _crop_and_adjust_annotations(crop_size, overlap, threshold, images_dir, labels_dir, image_file)

    LOGGER.success(f"Recorte de dataset completado en {dataset_path}")


def _crop_and_adjust_annotations(
    crop_size: int, overlap: int, threshold: float, images_dir: Path, labels_dir: Path, image_file: Path
) -> int:
    """
    Recorta una imagen en múltiples fragmentos de tamaño fijo con solapamiento y ajusta las anotaciones correspondientes.
    Esta función toma una imagen y la divide en recortes cuadrados de tamaño especificado con un solapamiento
    configurable. Para cada recorte, ajusta las anotaciones YOLO correspondientes y elimina los recortes
    que sean completamente blancos. La imagen original y su archivo de etiquetas son eliminados después
    del procesamiento.
    Args:
        crop_size (int): Tamaño en píxeles de cada lado del recorte cuadrado.
        overlap (int): Número de píxeles de solapamiento entre recortes adyacentes.
        threshold (float): Umbral mínimo para considerar válida una anotación en el recorte.
        images_dir (Path): Directorio donde se guardarán los recortes de imagen.
        labels_dir (Path): Directorio donde se guardarán los archivos de etiquetas ajustados.
        image_file (Path): Ruta al archivo de imagen original a procesar.
    Returns:
        int: Número total de recortes válidos (no blancos) generados.
    Raises:
        ValueError: Si no se puede leer la imagen (archivo no existe o formato inválido).
    Examples:
        >>> from pathlib import Path
        >>> images_dir = Path("./crops/images")
        >>> labels_dir = Path("./crops/labels")
        >>> image_file = Path("./original/image.jpg")
        >>> num_crops = _crop_and_adjust_annotations(512, 50, 0.3, images_dir, labels_dir, image_file)
        >>> print(f"Se generaron {num_crops} recortes válidos")
    Notes:
        - Los recortes se nombran con el patrón "{nombre_base}_crop_{numero}.jpg"
        - Los archivos de etiquetas siguen el patrón "{nombre_base}_crop_{numero}.txt"
        - Se asume formato de anotaciones YOLO (class_id x_center y_center width height)
        - Los recortes completamente blancos son descartados automáticamente
        - La imagen y etiquetas originales son eliminadas después del procesamiento
        - Los recortes en los bordes se ajustan para mantenerse dentro de los límites de la imagen
    """
    img = cv2.imread(str(image_file))
    if img is None:
        raise ValueError(
            f"No se pudo leer la imagen {image_file}. Asegúrate de que el archivo existe y es una imagen válida."
        )

    image_height, image_width, _ = img.shape
    base_name = image_file.stem
    label_file = labels_dir / f"{base_name}.txt"

    annotations = []
    if label_file.exists():
        with open(label_file, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                x_center, y_center, width, height = parts[1:]
                annotations.append((class_id, x_center, y_center, width, height))
    else:
        LOGGER.debug(f"No se encontró archivo de etiquetas para {image_file}. Continuando sin anotaciones.")

    LOGGER.debug(f"Eliminando {image_file} y {label_file}")
    image_file.unlink(missing_ok=True)
    label_file.unlink(missing_ok=True)

    # Generar los recortes
    num_white_crops = 0
    num_crops = 0
    for y in range(0, image_height, crop_size - overlap):  # Recorrer filas con solapamiento
        actual_y = y
        if actual_y + crop_size > image_height:  # Ajustar para la última fila si se excede el tamaño
            actual_y = image_height - crop_size

        for x in range(0, image_width, crop_size - overlap):  # Recorrer columnas con solapamiento
            actual_x = x
            if actual_x + crop_size > image_width:  # Ajustar para la última columna si se excede el tamaño
                actual_x = image_width - crop_size

            if actual_x < 0 or actual_y < 0:
                continue  # Evitar coordenadas negativas

            # En este punto, actual_x y actual_y son las coordenadas del recorte
            # o sea, tengo un rectángulo de imagen de tamaño crop_size x crop_size
            # que comienza en (actual_x, actual_y) y termina en (actual_x + crop_size, actual_y + crop_size)
            crop_img = img[actual_y : actual_y + crop_size, actual_x : actual_x + crop_size]
            if Helpers.is_white_image(crop_img)[0]:
                LOGGER.debug(f"El recorte de la imagen {image_file} en ({actual_x}, {actual_y}) es blanco. Saltando.")
                num_white_crops += 1
                continue

            # Guardar el recorte de la imagen
            num_crops += 1
            new_image_name = f"{base_name}_crop_{num_crops}.jpg"
            cv2.imwrite(str(images_dir / new_image_name), crop_img)

            # Ajustar las anotaciones para este recorte
            new_annotations = []
            for class_id, bbox_x_center, bbox_y_center, bbox_width, bbox_height in annotations:
                new_annotation = _new_annotation(
                    bbox_x_center,
                    bbox_y_center,
                    bbox_width,
                    bbox_height,
                    actual_x,
                    actual_y,
                    image_width,
                    image_height,
                    crop_size,
                    threshold,
                )
                if new_annotation:
                    new_x_center, new_y_center, new_width, new_height = new_annotation
                    new_annotations.append(
                        f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}"
                    )

            # Guardar el archivo de etiquetas para el recorte
            new_label_name = f"{base_name}_crop_{num_crops}.txt"
            with open(labels_dir / new_label_name, "w") as f:
                for ann in new_annotations:
                    f.write(ann + "\n")

            # Si estamos en la última columna de una fila y no hay solapamiento para la siguiente imagen,
            # salimos del bucle interior para evitar procesar la misma región dos veces si w % (image_size - overlap) != 0
            if actual_x + crop_size >= image_width and image_width > crop_size:
                break

                # Similar a lo anterior para la última fila
        if actual_y + crop_size >= image_height and image_height > crop_size:
            break
    LOGGER.debug(f"Recortes realizados: {num_crops}, Recortes blancos omitidos: {num_white_crops}")
    return num_crops


def _new_annotation(
    bbox_x_center: float,
    bbox_y_center: float,
    bbox_width: float,
    bbox_height: float,
    crop_x: int,
    crop_y: int,
    image_width: int,
    image_height: int,
    crop_size: int,
    iou_threshold: float,
) -> Optional[tuple[float, float, float, float]]:
    """
    Calcula las nuevas coordenadas YOLO normalizadas para una detección después de ser recortada.
    Esta función toma una detección en formato YOLO (coordenadas normalizadas) y determina
    si la detección es válida en un recorte específico de la imagen. Si es válida, calcula
    las nuevas coordenadas YOLO normalizadas relativas al recorte.
    Args:
        bbox_x_center (float): Coordenada x del centro de la caja de detección (normalizada, 0-1).
        bbox_y_center (float): Coordenada y del centro de la caja de detección (normalizada, 0-1).
        bbox_width (float): Ancho de la caja de detección (normalizado, 0-1).
        bbox_height (float): Alto de la caja de detección (normalizado, 0-1).
        crop_x (int): Coordenada x del inicio del recorte en píxeles absolutos.
        crop_y (int): Coordenada y del inicio del recorte en píxeles absolutos.
        image_width (int): Ancho de la imagen original en píxeles.
        image_height (int): Alto de la imagen original en píxeles.
        crop_size (int): Tamaño del recorte cuadrado en píxeles.
        iou_threshold (float): Umbral mínimo de intersección sobre área original para considerar válida la detección.
    Returns:
        Optional[tuple[float, float, float, float]]: Tupla con las nuevas coordenadas YOLO
            (x_center, y_center, width, height) normalizadas al recorte, o None si la
            detección no es válida en el recorte.
    Examples:
        >>> # Detección en el centro de una imagen 640x640, recorte 320x320 en esquina superior izquierda
        >>> result = _new_annotation(0.5, 0.5, 0.2, 0.2, 0, 0, 640, 640, 320, 0.5)
        >>> # Retorna las nuevas coordenadas normalizadas al recorte de 320x320
        >>> # Detección que no intersecta suficientemente con el recorte
        >>> result = _new_annotation(0.9, 0.9, 0.1, 0.1, 0, 0, 640, 640, 320, 0.8)
        >>> # Retorna None si la intersección es menor al umbral
    Notes:
        - La función asume que el recorte es cuadrado (crop_size x crop_size).
        - Las coordenadas de entrada deben estar en formato YOLO (normalizadas entre 0 y 1).
        - La función registra errores si las coordenadas están fuera de los límites de la imagen
          o si el área de la caja de detección es cero.
        - El umbral iou_threshold se aplica como ratio de intersección sobre área original,
          no como IoU tradicional (intersección sobre unión).
    """
    # Convertir coordenadas YOLO (normalizadas) a píxeles absolutos (de la imagen original)
    # Coordenadas de la caja de detección con respecto a la imagen original
    abs_x_min = round((bbox_x_center - bbox_width / 2) * image_width)
    abs_y_min = round((bbox_y_center - bbox_height / 2) * image_height)
    abs_x_max = round((bbox_x_center + bbox_width / 2) * image_width)
    abs_y_max = round((bbox_y_center + bbox_height / 2) * image_height)

    # Verificar que las coordenadas estén dentro de los límites de la imagen.
    if abs_x_min < 0 or abs_y_min < 0 or abs_x_max > image_width or abs_y_max > image_height:
        LOGGER.error(
            f"Coordenadas de la caja de detección fuera de los límites de la imagen: "
            f"({abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}) en una imagen de tamaño ({image_width}, {image_height})."
        )

    # Coordenadas del recorte con respecto a la imagen original
    crop_x_min, crop_y_min = crop_x, crop_y
    crop_x_max, crop_y_max = crop_x + crop_size, crop_y + crop_size

    # Calcular la intersección entre la caja de detección y el recorte
    inter_x_min = max(abs_x_min, crop_x_min)
    inter_y_min = max(abs_y_min, crop_y_min)
    inter_x_max = min(abs_x_max, crop_x_max)
    inter_y_max = min(abs_y_max, crop_y_max)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return None  # No hay intersección

    # Calcular áreas
    intersection_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    original_box_area = (abs_x_max - abs_x_min) * (abs_y_max - abs_y_min)

    if original_box_area == 0:
        LOGGER.error(
            f"Área de la caja de detección es cero: "
            f"({abs_x_min}, {abs_y_min}, {abs_x_max}, {abs_y_max}) en una imagen de tamaño ({image_width}, {image_height})."
        )
        return None

    intersection_ratio = intersection_area / original_box_area

    # Umbral para considerar que una detección es válida en el recorte
    # Por ejemplo, si al menos el 50% de la detección está en el recorte
    if original_box_area > 0 and intersection_ratio > iou_threshold:
        # Convertir coordenadas de la detección al sistema de coordenadas del recorte
        new_abs_x_min = max(0, inter_x_min - crop_x)
        new_abs_y_min = max(0, inter_y_min - crop_y)
        new_abs_x_max = min(crop_size, inter_x_max - crop_x)
        new_abs_y_max = min(crop_size, inter_y_max - crop_y)

        new_box_width = new_abs_x_max - new_abs_x_min
        new_box_height = new_abs_y_max - new_abs_y_min

        new_x_center = (new_abs_x_min + new_abs_x_max) / 2 / crop_size
        new_y_center = (new_abs_y_min + new_abs_y_max) / 2 / crop_size
        new_width = new_box_width / crop_size
        new_height = new_box_height / crop_size

        return new_x_center, new_y_center, new_width, new_height
    return None


@deprecated(
    version="1.0.0",
    reason="Esta función está obsoleta y será eliminada en futuras versiones. Usa balance_dataset_v1 en su lugar.",
)
@app.command()
def balance_dataset(
    dataset_path: Path,
    dataset_format: DatasetFormat = DatasetFormat.YOLO,
    all_classes: bool = False,
) -> None:
    """
    Balancea un dataset YOLO eliminando imágenes para igualar la cantidad de imágenes con y sin detecciones,
    o para igualar la cantidad de imágenes por clase.

    Este proceso es útil para evitar sesgos en el entrenamiento de modelos de detección de objetos,
    asegurando que todas las clases tengan una representación similar en el dataset o que haya un balance
    entre imágenes con y sin detecciones.

    Args:
        dataset_path (Path, optional): Ruta al dataset YOLO que se desea balancear.
            Por defecto es la carpeta de datos interinos con el nombre y versión del dataset completo.
        dataset_format (DatasetFormat, optional): Formato del dataset.
            Por defecto es YOLO.
        all_classes (bool, optional): Si es True, balancea el dataset por clase, igualando la cantidad de imágenes
            para cada clase. Si es False, balancea entre imágenes con y sin detecciones.
            Por defecto es False.

    Raises:
        NotImplementedError: Si el formato del dataset no es YOLO.

    Notas:
        - Si `all_classes` es True, el balanceo se realiza considerando cada clase individualmente.
          Esto implica que se eliminarán imágenes para que todas las clases tengan la misma cantidad de imágenes.
        - Si `all_classes` es False, el balanceo se realiza entre imágenes con detecciones y sin detecciones,
          igualando la cantidad de imágenes en ambas categorías.
        - Las imágenes eliminadas se seleccionan de manera aleatoria.
        - Las imágenes sin anotaciones se consideran como imágenes sin detecciones.
        - El archivo `dataset.yaml` es necesario para identificar los nombres de las clases cuando se realiza
          el balanceo por clase.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"El dataset {dataset_path} no existe.")

    if dataset_format != DatasetFormat.YOLO:
        raise NotImplementedError(
            f"El formato de dataset {dataset_format} no está implementado para el balanceo de imágenes."
        )

    images_dir = dataset_path / "images" / "full"
    labels_dir = dataset_path / "labels" / "full"
    dataset_yaml_path = dataset_path / "dataset.yaml"

    if not images_dir.exists():
        raise FileNotFoundError(f"La carpeta 'images' no se encontró en {dataset_path}.")
    if not labels_dir.exists():
        raise FileNotFoundError(f"La carpeta 'labels' no se encontró en {dataset_path}.")

    class_names = []
    try:
        with open(dataset_yaml_path, "r") as f:
            data = yaml.safe_load(f)
            class_names = data.get("names", [])
            if all_classes and not class_names:
                raise ValueError(
                    f"No se encontraron nombres de clases en {dataset_yaml_path}. Necesario para balancear por clase."
                )
    except FileNotFoundError:
        if all_classes:  # Si se pide balanceo por clase y no hay YAML, salimos
            raise FileNotFoundError(
                f"No se encontró dataset.yaml en {dataset_path}. Necesario para balancear por clase."
            )
        LOGGER.warning(f"No se encontró dataset.yaml en {dataset_path}. Balanceo entre imágenes con y sin detecciones.")
    except yaml.YAMLError as e:
        if all_classes:  # Si se pide balanceo por clase y hay error de YAML, salimos
            raise ValueError(f"Error al parsear dataset.yaml en {dataset_path}: {e}")
        LOGGER.warning(f"Error al parsear dataset.yaml en {dataset_path}: {e}")

    # 1. Recopilar todas las imágenes y clasificar
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    # Key: class_id (int), Value: list[tuple[Path, Path]] (lista de (image_file, label_file))
    images_by_class: dict[int, list[tuple[Path, Path]]] = defaultdict(list)

    images_with_detections: list[tuple[Path, Path]] = []
    images_without_detections: list[tuple[Path, Path]] = []

    LOGGER.debug("Clasificando imágenes por presencia de anotaciones y por clase...")

    for image_file in image_files:
        base_name = image_file.stem
        label_file = labels_dir / f"{base_name}.txt"

        if label_file.exists() and label_file.stat().st_size > 0:
            with open(label_file, "r") as f:
                for line in f:
                    try:
                        class_id = int(line.strip().split()[0])
                        images_by_class[class_id].append((image_file, label_file))
                    except (ValueError, IndexError):
                        LOGGER.warning(
                            f"Línea de anotación mal formada en {label_file}: {line.strip()}. Ignorando esta anotación."
                        )
            images_with_detections.append((image_file, label_file))
        else:
            images_without_detections.append((image_file, label_file))

    LOGGER.debug(f"Imágenes con detecciones (total): {len(images_with_detections)}")
    LOGGER.debug(f"Imágenes sin detecciones (total): {len(images_without_detections)}")

    if all_classes:
        LOGGER.debug("Conteo de detecciones por clase (antes del balanceo):")
        for class_id, img_list in sorted(images_by_class.items()):
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_Class_{class_id}"
            LOGGER.debug(f"  Clase {class_id} ({class_name}): {len(img_list)} detecciones.")

        LOGGER.info("Balanceando el dataset por todas las clases individuales de objetos...")

        # Encontrar la cantidad de detecciones para la clase minoritaria
        min_class_count = float("inf")
        max_images_with_detections = float("-inf")
        if images_by_class:  # Asegurarse de que haya al menos una clase con detecciones
            min_class_count = min(len(img_list) for img_list in images_by_class.values())
            max_images_with_detections = max(len(set(img_list)) for img_list in images_by_class.values())
        else:
            LOGGER.warning("No se encontraron detecciones para ninguna clase. Balanceo por clase no aplicable.")
            return

        # Conjunto de imágenes a mantener para evitar eliminar duplicados o deseados
        files_to_keep: set[Path] = set()

        # Para cada clase, seleccionar `min_class_count` imágenes aleatoriamente
        for class_id, img_list in images_by_class.items():
            random.shuffle(img_list)
            for img_file, _ in img_list[:min_class_count]:
                files_to_keep.add(img_file)

        # Agregar imágenes sin detecciones al conjunto de archivos a mantener
        random.shuffle(images_without_detections)
        for img_file, _ in images_without_detections[:max_images_with_detections]:
            files_to_keep.add(img_file)

        all_processed_files = set(image_file for image_file, _ in images_with_detections) | set(
            image_file for image_file, _ in images_without_detections
        )

        # Las imágenes que deben eliminarse son todas las que no están en files_to_keep
        images_to_remove = [
            (img_path, labels_dir / f"{img_path.stem}.txt")
            for img_path in all_processed_files
            if img_path not in files_to_keep
        ]

        LOGGER.info(
            f"Eliminando {len(images_to_remove)} imágenes para balancear por clase. Objetivo por clase: {min_class_count}"
        )
        for img_file, lbl_file in images_to_remove:
            img_file.unlink(missing_ok=True)
            lbl_file.unlink(missing_ok=True)

        LOGGER.info(
            f"Imágenes con detecciones restantes: {len(files_to_keep & set(image_file for image_file, _ in images_with_detections))}"
        )
        LOGGER.info(
            f"Imágenes sin detecciones restantes: {len(files_to_keep & set(image_file for image_file, _ in images_without_detections))}"
        )
        LOGGER.success(f"Balanceo de dataset completado en {dataset_path}")

    else:
        LOGGER.info("Balanceando el dataset entre imágenes con y sin detecciones...")

        effective_target_count = min(len(images_with_detections), len(images_without_detections))

        # Eliminar exceso de imágenes con detecciones
        if len(images_with_detections) > effective_target_count:
            random.shuffle(images_with_detections)
            images_to_remove_with_detections = images_with_detections[effective_target_count:]
            LOGGER.info(f"Eliminando {len(images_to_remove_with_detections)} imágenes con detecciones.")
            for img_file, lbl_file in images_to_remove_with_detections:
                img_file.unlink(missing_ok=True)
                lbl_file.unlink(missing_ok=True)

        # Eliminar exceso de imágenes sin detecciones
        if len(images_without_detections) > effective_target_count:
            random.shuffle(images_without_detections)
            images_to_remove_without_detections = images_without_detections[effective_target_count:]
            LOGGER.info(f"Eliminando {len(images_to_remove_without_detections)} imágenes sin detecciones.")
            for img_file, lbl_file in images_to_remove_without_detections:
                img_file.unlink(missing_ok=True)
                lbl_file.unlink(missing_ok=True)

        LOGGER.info(f"Imágenes con detecciones restantes: {effective_target_count}")
        LOGGER.info(f"Imágenes sin detecciones restantes: {effective_target_count}")
        LOGGER.success(f"Balanceo de dataset completado en {dataset_path}")


@app.command()
def balance_dataset_v1(
    dataset_path: Path,
    output_path: Path,
    export_categories: list[dict],
    dataset_format: DatasetFormat = DatasetFormat.YOLO,
    background_percentage: float = 0.1,
    all_classes: bool = False,
) -> None:
    """
    Balancea un dataset de detección de objetos en formato YOLO.
    Esta función procesa un dataset de detección de objetos para crear una versión balanceada
    que puede incluir todas las clases con la misma cantidad de detecciones o eliminar imágenes
    sin detecciones según los parámetros especificados.
    Args:
        dataset_path (Path): Ruta al dataset original en formato YOLO.
        output_path (Path): Ruta donde se guardará el dataset balanceado.
        export_categories (list[dict]): Lista de diccionarios con las categorías a exportar.
            Cada diccionario debe contener al menos la clave 'name' con el nombre de la clase.
        dataset_format (DatasetFormat, optional): Formato del dataset. Por defecto DatasetFormat.YOLO.
        background_percentage (float, optional): Porcentaje de imágenes sin detecciones a incluir
            en el dataset balanceado. Por defecto 0.1 (10%).
        all_classes (bool, optional): Si True, balancea todas las clases para que tengan la misma
            cantidad de detecciones que la clase menos representada. Si False, elimina las imágenes
            sin detecciones. Por defecto False.
    Returns:
        None: La función no retorna valores, pero genera un dataset balanceado en output_path.
    Raises:
        ValueError: Si no se puede crear una vista de exportación válida o si el dataset
            no contiene imágenes con detecciones.
    Examples:
        Balancear dataset eliminando imágenes sin detecciones:
        >>> from pathlib import Path
        >>> categories = [{"name": "person"}, {"name": "car"}]
        >>> balance_dataset_v1(
        ...     Path("dataset_original"),
        ...     Path("dataset_balanceado"),
        ...     categories
        ... )
        Balancear todas las clases con 20% de imágenes de fondo:
        >>> balance_dataset_v1(
        ...     Path("dataset_original"),
        ...     Path("dataset_balanceado"),
        ...     categories,
        ...     all_classes=True,
        ...     background_percentage=0.2
        ... )
    Notes:
        - La función requiere que el dataset esté en formato YOLO con estructura estándar
          (directorios images/ y labels/).
        - Cuando all_classes=True, el balanceeo se realiza basándose en la clase con menor
          cantidad de detecciones.
        - Si output_path es igual a dataset_path, la función sobrescribirá el dataset original
          usando una carpeta temporal durante el proceso.
        - El porcentaje de imágenes de fondo (background_percentage) se calcula respecto al
          número total de imágenes con detecciones en el dataset balanceado.
    """
    _check_dataset(dataset_path, output_path, dataset_format)

    images_dir, labels_dir, dataset_yaml_path = _get_dataset_path_metadata(dataset_path)

    _check_dataset_directories(dataset_path, images_dir, labels_dir)

    _validate_class_names(dataset_path, all_classes, dataset_yaml_path)

    same_folder, dataset = _initialize_dataset(dataset_path, output_path)
    export_view = None

    no_detections_view = dataset.filter_field("ground_truth", F("detections").length() == 0)
    no_detections_samples_id = no_detections_view.values("id")
    no_detections_count = no_detections_view.count()
    with_detections_view = dataset.filter_field("ground_truth", F("detections").length() > 0)
    with_detections_count = with_detections_view.count()
    LOGGER.debug(
        f"Imágenes con detecciones: {with_detections_count}, " f"Imágenes sin detecciones: {no_detections_count}"
    )

    no_detections_to_add = 0
    if not all_classes:
        # Eliminamos las imágenes sin detecciones.
        export_view = dataset.exclude(no_detections_samples_id)
    else:
        LOGGER.debug("Balanceando el dataset por todas las clases individuales de objetos...")
        class_count = dataset.count_values("ground_truth.detections.label")
        class_count_ordered = {k: v for k, v in sorted(class_count.items(), key=lambda item: item[1])}
        min_class_count = min(class_count_ordered.values())
        LOGGER.debug(f"Conteo de detecciones por clase (antes del balanceo): {class_count_ordered}")

        detections_counts = {class_name: 0 for class_name in class_count}
        samples_to_include = []

        for class_name in class_count_ordered.keys():
            LOGGER.debug(f"Procesando clase '{class_name}' con {class_count_ordered[class_name]} detecciones.")
            LOGGER.debug(f"Conteo de detecciones totales actuales para todas las clases: {detections_counts}")
            filtered_view = dataset.match(F("ground_truth.detections.label").contains(class_name))
            filtered_view = filtered_view.shuffle()
            if detections_counts[class_name] >= min_class_count:
                LOGGER.info(f"Ya se alcanzó el mínimo requerido de detecciones para la clase '{class_name}'.")
                continue
            for sample in filtered_view:
                samples_to_include.append(sample.id)
                sample_detections = sample.ground_truth.detections
                labels = [detection.label for detection in sample_detections]
                sample_detections_count = Counter(labels)
                for label, count in sample_detections_count.items():
                    detections_counts[label] += count
                if detections_counts[class_name] >= min_class_count:
                    LOGGER.debug(
                        f"Se alcanzó el mínimo requerido de {min_class_count} detecciones para la clase '{class_name}'."
                    )
                    # Salir del bucle interior si se alcanza el mínimo requerido
                    break

        # Incluir las imágenes con detecciones balanceadas
        export_view = dataset.select(samples_to_include)
        with_detections_count = export_view.count()

    if not export_view:
        raise ValueError(
            "No se pudo crear una vista de exportación. Asegúrate de que el dataset tenga imágenes con detecciones."
        )

    # Agregamos el porcentaje de imágenes sin detecciones si se especifica
    if background_percentage > 0:
        random.shuffle(no_detections_samples_id)
        no_detections_to_add = int(with_detections_count * background_percentage)
        if no_detections_to_add > no_detections_count:
            LOGGER.warning(
                f"Se solicitó agregar {no_detections_to_add} imágenes sin detecciones, "
                f"pero solo hay {no_detections_count} disponibles. Se agregarán todas."
            )
            no_detections_to_add = no_detections_count
        else:
            LOGGER.debug(f"Se agregarán {no_detections_to_add} imágenes sin detecciones al dataset balanceado.")
        no_detections_samples_id = no_detections_samples_id[:no_detections_to_add]
        no_detections_view = dataset.select(no_detections_samples_id)
        export_view += no_detections_view

    # Finalmente, exportamos el dataset balanceado
    _export_dataset(dataset_path, output_path, export_categories, same_folder, export_view)

def _export_dataset(dataset_path, output_path, export_categories, same_folder, export_view):
    """Exporta el dataset balanceado a la ruta especificada, manejando el caso de sobrescribir en la misma carpeta."""
    export_categories_list = [cat["name"] for cat in export_categories]
    if same_folder:
        temp_path = TEMP_DATA_FOLDER / "temp_dataset"
        export_view.export(
            export_dir=str(temp_path),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            overwrite=True,
            split="full",
            classes=export_categories_list,
        )
        LOGGER.debug(f"Eliminando archivos originales en {dataset_path}.")
        shutil.rmtree(dataset_path)

        LOGGER.debug(f"Copiando archivos balanceados a {output_path}.")
        shutil.copytree(temp_path, output_path, dirs_exist_ok=True)

        LOGGER.debug(f"Actualizando dataset.yaml en {output_path}.")
        dataset_yaml = output_path / "dataset.yaml"
        with open(dataset_yaml, "r") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["path"] = str(output_path)
        with open(dataset_yaml, "w") as f:
            yaml.safe_dump(yaml_data, f)

        LOGGER.debug(f"Eliminando carpeta temporal {temp_path}.")
        shutil.rmtree(temp_path)
    else:
        export_view.export(
            export_dir=str(output_path),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            overwrite=True,
            split="full",
            classes=export_categories_list,
        )


def _initialize_dataset(dataset_path: Path, output_path: Path) -> tuple[bool, fo.Dataset]:
    """Inicializa el dataset de Fiftyone. También devuelve si se desea grabar en la misma carpeta."""
    same_folder = False
    if output_path == dataset_path:
        same_folder = True
        LOGGER.warning(
            "La ruta de salida es la misma que la del dataset original. "
            "Se eliminarán imágenes y etiquetas originales."
        )

    dataset_name = "temp_dataset"
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_path,
        dataset_type=fo.types.YOLOv5Dataset,
        overwrite=True,
        name=dataset_name,
        split="full",
        label_field="ground_truth",
    )

    return same_folder, dataset


def _check_dataset_directories(dataset_path: Path, images_dir: Path, labels_dir: Path) -> None:
    """Chequea que existan los directorios del dataset."""
    if not images_dir.exists():
        raise FileNotFoundError(f"La carpeta 'images' no se encontró en {dataset_path}.")
    if not labels_dir.exists():
        raise FileNotFoundError(f"La carpeta 'labels' no se encontró en {dataset_path}.")


def _get_dataset_path_metadata(dataset_path: Path) -> tuple[Path, Path, Path]:
    """Obtiene las rutas de los datos del dataset (imagenes y labels)."""
    images_dir = dataset_path / "images" / "full"
    labels_dir = dataset_path / "labels" / "full"
    dataset_yaml_path = dataset_path / "dataset.yaml"
    return images_dir, labels_dir, dataset_yaml_path


def _check_dataset(dataset_path: Path, output_path: Path, dataset_format: DatasetFormat) -> None:
    """Chequea la existencia del dataset, formato y existencia de la carpeta de salida (la crea si no existe)."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"El dataset {dataset_path} no existe.")
    if not output_path.exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)
        LOGGER.debug(f"Creando carpeta de salida {output_path}.")
    if dataset_format != DatasetFormat.YOLO:
        raise NotImplementedError(
            f"El formato de dataset {dataset_format} no está implementado para el balanceo de imágenes."
        )


def _validate_class_names(dataset_path: Path, all_classes: bool, dataset_yaml_path: Path) -> None:
    """
    Valida los nombres de clases en un dataset de YOLO.
    Esta función verifica que exista un archivo dataset.yaml válido y que contenga
    nombres de clases cuando se requiere balanceo por todas las clases.
    Args:
        dataset_path (Path): Ruta al directorio del dataset.
        all_classes (bool): Si True, requiere que existan nombres de clases válidos.
                           Si False, permite continuar sin archivo yaml o con errores.
        dataset_yaml_path (Path): Ruta completa al archivo dataset.yaml.
    Raises:
        ValueError: Si all_classes es True y no se encuentran nombres de clases
                   en el archivo yaml, o si hay errores al parsear el yaml.
        FileNotFoundError: Si all_classes es True y no se encuentra el archivo
                          dataset.yaml.
    Note:
        Si all_classes es False, los errores se registran como warnings y la
        función continúa sin lanzar excepciones, permitiendo balanceo básico
        entre imágenes con y sin detecciones.
    """
    try:
        with open(dataset_yaml_path, "r") as f:
            data = yaml.safe_load(f)
            class_names = data.get("names", [])
            if all_classes and not class_names:
                raise ValueError(
                    f"No se encontraron nombres de clases en {dataset_yaml_path}. Necesario para balancear por clase."
                )
    except FileNotFoundError:
        if all_classes:  # Si se pide balanceo por clase y no hay YAML, salimos
            raise FileNotFoundError(
                f"No se encontró dataset.yaml en {dataset_path}. Necesario para balancear por clase."
            )
        LOGGER.warning(f"No se encontró dataset.yaml en {dataset_path}. Balanceo entre imágenes con y sin detecciones.")
    except yaml.YAMLError as e:
        if all_classes:  # Si se pide balanceo por clase y hay error de YAML, salimos
            raise ValueError(f"Error al parsear dataset.yaml en {dataset_path}: {e}")
        LOGGER.warning(f"Error al parsear dataset.yaml en {dataset_path}: {e}")


def under_sample_dataset(
    dataset_path: Path, output_path: Path, export_categories: list[str], target_size: int, dataset_format: DatasetFormat = DatasetFormat.YOLO
) -> None:
    """
    Aplica undersampling a un dataset de YOLO.
    Args:
        dataset_path (Path): Ruta al directorio del dataset.
        output_path (Path): Ruta de salida para el dataset balanceado.
        target_size (int): Tamaño objetivo del dataset tras el undersampling.
        dataset_format (DatasetFormat): Formato del dataset (por defecto YOLO).
    """
    _check_dataset(dataset_path, output_path, dataset_format)

    images_dir, labels_dir, dataset_yaml_path = _get_dataset_path_metadata(dataset_path)

    _check_dataset_directories(dataset_path, images_dir, labels_dir)

    same_folder, dataset = _initialize_dataset(dataset_path, output_path)

    images_count = dataset.count()
    if images_count <= target_size:
        LOGGER.warning(f"El dataset ya tiene {images_count} imágenes, menor o igual al tamaño objetivo {target_size}.")
        return
    
    images_to_delete_count = images_count - target_size
    LOGGER.info(f"Reduciendo el dataset de {images_count} a {target_size} imágenes mediante undersampling.")
    random_samples_view = dataset.shuffle().take(images_to_delete_count)
    samples_to_delete_id = random_samples_view.values("id")
    export_view = dataset.exclude(samples_to_delete_id)

    _export_dataset(dataset_path, output_path, export_categories, same_folder, export_view)

if __name__ == "__main__":
    # app()
    CLASS_NAMES = {0: "palmera"}
    COLOR_MAP = {
        "palmera": (0, 255, 0),  # Verde
    }
    CATEGORIES = [{"id": id, "name": name, "supercategory": ""} for id, name in CLASS_NAMES.items()]
    dataset_path = Path(
        "E:/Documentos/Git Repositories/uba-ceia-proy-final/ceia-proyecto-final/modulo-IA/data/interim/coco_palm_dataset_v1.1_step"
    )
    output_path = Path("E:/Documentos/Git Repositories/uba-ceia-proy-final/ceia-proyecto-final/modulo-IA/data/interim/coco_palm_dataset_v1.1_under_sample")
    target_size = 1000
    under_sample_dataset(dataset_path, output_path, CATEGORIES, target_size)
