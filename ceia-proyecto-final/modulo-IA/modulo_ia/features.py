from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger as LOGGER
from modulo_ia.config import config as CONFIG

import cv2  # Debe venir después de la importación del config por configuraciones de variables de entorno.

import yaml

import typer

import torch
from tqdm import tqdm
import typer

from PIL import Image

from transformers import ViTImageProcessor, ViTImageProcessorFast

# Transformaciones de imágenes (Torchvision)
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Resize,
    CenterCrop,
    RandomApply,
    RandomRotation,
    RandomCrop,
    ColorJitter,
)

from modulo_ia.utils.types import DatasetFormat
import modulo_apps.utils.helpers as Helpers

RAW_DATA_FOLDER = CONFIG.folders.raw_data_folder
EXTERNAL_DATA_FOLDER = CONFIG.folders.external_data_folder
INTERIM_DATA_FOLDER = CONFIG.folders.interim_data_folder
PROCESSED_DATA_FOLDER = CONFIG.folders.processed_data_folder

FULL_DATASET_NAME = CONFIG.names.palm_detection_dataset_name
FULL_DATASET_VERSION = CONFIG.versions.detection_dataset_version

PARTIAL_DATASET_NAME = CONFIG.names.partial_dataset_name
PARTIAL_DATASET_VERSION = CONFIG.versions.detection_dataset_version

CUTOUTS_DATASET_NAME = CONFIG.names.cutouts_dataset_name
CUTOUTS_DATASET_VERSION = CONFIG.versions.cutouts_dataset_version

app = typer.Typer()


class SwinV2Transforms:

    def __init__(self, image_processor: Union[ViTImageProcessor, ViTImageProcessorFast]) -> None:
        self.image_processor = image_processor
        self.mean = image_processor.image_mean
        self.std = image_processor.image_std

        if "height" in image_processor.size:
            self.size = (image_processor.size["height"], image_processor.size["width"])
            self.crop_size = self.size
            self.max_size = None
        elif "shortest_edge" in image_processor.size:
            self.size = image_processor.size["shortest_edge"]
            self.crop_size = (self.size, self.size)
            self.max_size = image_processor.size.get("longest_edge")

        self.train_transforms = Compose(
            [
                RandomResizedCrop(self.crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ]
        )

        self.val_transforms = Compose(
            [
                Resize(self.size),
                CenterCrop(self.crop_size),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ]
        )

        self._unnormalize = Normalize(
            mean=[-m / s for m, s in zip(self.mean, self.std)],
            std=[1 / s for s in self.std],
        )

    def __call__(self, batch, train=True):
        return self.transforms(batch, train)

    def transforms(self, batch, train=True):
        batch["pixel_values"] = (
            [self.train_transforms(img) for img in batch["image"]]
            if train
            else [self.val_transforms(img) for img in batch["image"]]
        )
        del batch["image"]

        return batch

    def unnormalize(self, img: torch.Tensor) -> torch.Tensor:
        return self._unnormalize(img)

    def transforms_to_string(self) -> str:
        def format_compose(compose):
            return "\n  - " + "\n  - ".join(str(t) for t in compose.transforms)

        return (
            f"Transformaciones de entrenamiento:{format_compose(self.train_transforms)}\n\n"
            f"Transformaciones de validacion:{format_compose(self.val_transforms)}"
        )

    def transforms_to_dict(self) -> dict:
        """Devuelve un diccionario con las transformaciones de entrenamiento y validación."""
        return {
            "train_transforms": [str(t) for t in self.train_transforms],
            "val_transforms": [str(t) for t in self.val_transforms],
        }


@app.command()
def crop_dataset(
    dataset_path: Path = INTERIM_DATA_FOLDER / f"{FULL_DATASET_NAME}_{FULL_DATASET_VERSION}_{DatasetFormat.YOLO.value}",
    dataset_format: DatasetFormat = DatasetFormat.YOLO,
    image_size: int = 640,
    overlap: int = 200,
    threshold: float = 0.5,
) -> None:
    """
    Recorta un dataset YOLO en parches más pequeños y ajusta las anotaciones para cada parche.

    Argumentos:
        dataset_path (Path): Ruta al dataset YOLO que se desea recortar.
        image_size (int): Tamaño de los parches cuadrados en los que se recortará cada imagen.
        overlap (int): Cantidad de píxeles de solapamiento entre parches adyacentes.
        threshold (float): Proporción mínima de intersección sobre unión (IoU) requerida para incluir una anotación en un parche.

    Notas:
        - Las imágenes y anotaciones recortadas se guardan en las carpetas correspondientes dentro del dataset.
        - Las imágenes blancas y los parches blancos se omiten.
        - Las anotaciones se ajustan al sistema de coordenadas de cada parche y se incluyen solo si cumplen con el umbral de IoU.
    """
    # TODO: Mejorar performance (concurrent.futures o multiprocessing)
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
        cant_crops += _crop_and_adjust_annotations(image_size, overlap, threshold, images_dir, labels_dir, image_file)

    LOGGER.success(f"Recorte de dataset completado en {dataset_path}")


def _crop_and_adjust_annotations(
    image_size: int, overlap: int, threshold: float, images_dir: Path, labels_dir: Path, image_file: Path
) -> int:
    """
    Recorta una imagen en parches más pequeños y ajusta las anotaciones para cada parche.
    Esta función toma una imagen de entrada y sus anotaciones correspondientes, divide la imagen en parches
    más pequeños de un tamaño especificado con solapamiento opcional, y ajusta las anotaciones para cada parche.
    También guarda las imágenes recortadas y sus anotaciones correspondientes en los directorios especificados.
    Argumentos:
        image_size (int): El tamaño de los parches cuadrados en los que se recortará la imagen.
        overlap (int): La cantidad de píxeles por la cual los parches adyacentes se solapan.
        threshold (float): La proporción mínima de intersección sobre unión (IoU) requerida para que una anotación
            sea incluida en un parche.
        images_dir (Path): El directorio donde se guardarán las imágenes recortadas.
        labels_dir (Path): El directorio donde se guardarán las anotaciones ajustadas.
        image_file (Path): La ruta al archivo de imagen de entrada.
    Retorna:
        int: El número de parches recortados generados.
    Lanza:
        ValueError: Si la imagen de entrada no puede ser leída o es inválida.
    Notas:
        - La función omite imágenes blancas y parches blancos.
        - Las anotaciones se ajustan para adaptarse al sistema de coordenadas de cada parche.
        - Las anotaciones se incluyen en un parche solo si su IoU con el parche supera el umbral especificado.
    """
    img = cv2.imread(str(image_file))
    if img is None:
        raise ValueError(
            f"No se pudo leer la imagen {image_file}. Asegúrate de que el archivo existe y es una imagen válida."
        )

    h, w, _ = img.shape
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
    for y in range(0, h, image_size - overlap):  # Recorrer filas con solapamiento
        if y + image_size > h:  # Ajustar para la última fila si se excede el tamaño
            y = h - image_size

        for x in range(0, w, image_size - overlap):  # Recorrer columnas con solapamiento
            if x + image_size > w:  # Ajustar para la última columna si se excede el tamaño
                x = w - image_size

            # En este punto, x e y son las coordenadas del recorte
            # o sea, tengo un rectángulo de imagen de tamaño image_size x image_size
            # que comienza en (x, y) y termina en (x + image_size, y + image_size)
            crop_img = img[y : y + image_size, x : x + image_size]
            if Helpers.is_white_image(crop_img)[0]:
                LOGGER.debug(f"El recorte de la imagen {image_file} en ({x}, {y}) es blanco. Saltando.")
                num_white_crops += 1
                continue

            # Guardar el recorte de la imagen
            new_image_name = f"{base_name}_crop_{num_crops + 1}.jpg"
            cv2.imwrite(str(images_dir / new_image_name), crop_img)

            # Ajustar las anotaciones para este recorte
            new_annotations = []
            for class_id, x_c, y_c, box_w, box_h in annotations:
                # Convertir coordenadas YOLO (normalizadas) a píxeles absolutos (de la imagen original)
                abs_x_min = int((x_c - box_w / 2) * w)
                abs_y_min = int((y_c - box_h / 2) * h)
                abs_x_max = int((x_c + box_w / 2) * w)
                abs_y_max = int((y_c + box_h / 2) * h)

                # Verificar si la detección está dentro del recorte
                # Solo incluimos detecciones que están completamente dentro del recorte
                # O que al menos una parte significativa está dentro

                # Intersección de la caja de detección con el recorte
                crop_x_min, crop_y_min = x, y
                crop_x_max, crop_y_max = x + image_size, y + image_size

                inter_x_min = max(abs_x_min, crop_x_min)
                inter_y_min = max(abs_y_min, crop_y_min)
                inter_x_max = min(abs_x_max, crop_x_max)
                inter_y_max = min(abs_y_max, crop_y_max)

                # Calcular el área de la intersección
                inter_width = max(0, inter_x_max - inter_x_min)
                inter_height = max(0, inter_y_max - inter_y_min)
                intersection_area = inter_width * inter_height

                # Área de la caja original
                original_box_area = (abs_x_max - abs_x_min) * (abs_y_max - abs_y_min)

                # Umbral para considerar que una detección es válida en el recorte
                # Por ejemplo, si al menos el 50% de la detección está en el recorte
                if original_box_area > 0 and (intersection_area / original_box_area) > threshold:
                    # Convertir coordenadas de la detección al sistema de coordenadas del recorte
                    new_abs_x_min = max(0, inter_x_min - x)
                    new_abs_y_min = max(0, inter_y_min - y)
                    new_abs_x_max = min(image_size, inter_x_max - x)
                    new_abs_y_max = min(image_size, inter_y_max - y)

                    new_box_width = new_abs_x_max - new_abs_x_min
                    new_box_height = new_abs_y_max - new_abs_y_min

                    new_x_center = (new_abs_x_min + new_abs_x_max) / 2 / image_size
                    new_y_center = (new_abs_y_min + new_abs_y_max) / 2 / image_size
                    new_width = new_box_width / image_size
                    new_height = new_box_height / image_size

                    new_annotations.append(
                        f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}"
                    )

            # Guardar el archivo de etiquetas para el recorte
            new_label_name = f"{base_name}_crop_{num_crops + 1}.txt"
            with open(labels_dir / new_label_name, "w") as f:
                for ann in new_annotations:
                    f.write(ann + "\n")
            num_crops += 1

            # Si estamos en la última columna de una fila y no hay solapamiento para la siguiente imagen,
            # salimos del bucle interior para evitar procesar la misma región dos veces si w % (image_size - overlap) != 0
            if x + image_size >= w and w > image_size:
                break

                # Similar a lo anterior para la última fila
        if y + image_size >= h and h > image_size:
            break
    LOGGER.debug(f"Recortes realizados: {num_crops}, Recortes blancos omitidos: {num_white_crops}")
    return num_crops


@app.command()
def balance_dataset(
    dataset_path: Path = INTERIM_DATA_FOLDER / f"{FULL_DATASET_NAME}_{FULL_DATASET_VERSION}_{DatasetFormat.YOLO.value}",
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
        LOGGER.error(f"El dataset {dataset_path} no existe.")
        return

    if dataset_format != DatasetFormat.YOLO:
        raise NotImplementedError(
            f"El formato de dataset {dataset_format} no está implementado para el balanceo de imágenes."
        )

    images_dir = dataset_path / "images" / "full"
    labels_dir = dataset_path / "labels" / "full"
    dataset_yaml_path = dataset_path / "dataset.yaml"

    if not images_dir.exists() or not labels_dir.exists():
        LOGGER.error(f"Las carpetas 'images' o 'labels' no se encontraron en {dataset_path}.")
        return

    class_names = []
    try:
        with open(dataset_yaml_path, "r") as f:
            data = yaml.safe_load(f)
            class_names = data.get("names", [])
    except FileNotFoundError:
        LOGGER.error(f"No se encontró dataset.yaml en {dataset_path}. Necesario para balancear por clase.")
        if all_classes:  # Si se pide balanceo por clase y no hay YAML, salimos
            return
    except yaml.YAMLError as e:
        LOGGER.error(f"Error al parsear dataset.yaml: {e}")
        if all_classes:  # Si se pide balanceo por clase y hay error de YAML, salimos
            return

    # 1. Recopilar todas las imágenes y clasificar
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    # Key: class_id (int), Value: List[Tuple[Path, Path]] (lista de (image_file, label_file))
    images_by_class: Dict[int, List[Tuple[Path, Path]]] = defaultdict(list)

    images_with_detections: List[Tuple[Path, Path]] = []
    images_without_detections: List[Tuple[Path, Path]] = []

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


if __name__ == "__main__":
    app()
