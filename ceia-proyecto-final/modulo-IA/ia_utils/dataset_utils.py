import os, sys, yaml, shutil

sys.path.append(os.path.abspath("../../modulo-apps"))  # Se agrega modulo-mini-apps

from pathlib import Path
from typing import List, Optional
from sklearn.model_selection import train_test_split

from apps_utils.logging import Logging
from apps_config.settings import Config
from apps_com_db.mongodb_client import MongoDB

from apps_com_s3.procesador_s3 import ProcesadorS3
from tqdm import tqdm
import apps_etiquetado.procesador_anotaciones_mongodb as ProcesadorCocoDataset

CONFIG = Config().config_data
LOGGER = Logging().logger
DB = MongoDB().db

download_folder = Path(CONFIG["folders"]["download_folder"])
DOWNLOAD_RAW_DATASET_FOLDER = Path(CONFIG["folders"]["raw_dataset_folder"])
DOWNLOAD_YOLO_DATASET_FOLDER = Path(CONFIG["folders"]["yolo_dataset"])


def download_full_dataset(
    for_patches: bool = True,
    folder_path: Optional[Path] = None,
    with_annotations: bool = True,
    annotations_field_name: str = "cvat",
    annotations_output_filename: str = None,
) -> List[str]:
    procesadorS3 = ProcesadorS3()
    if folder_path is None:
        Path(DOWNLOAD_RAW_DATASET_FOLDER).mkdir(parents=True, exist_ok=True)
        folder_path = DOWNLOAD_RAW_DATASET_FOLDER
    if annotations_output_filename is None:
        annotations_output_filename = folder_path / "labels.json"

    data_folder_path = folder_path / "data"
    Path(data_folder_path).mkdir(parents=True, exist_ok=True)
    if not for_patches:
        images_names = ProcesadorCocoDataset.list_images_w_ann_from_mongodb()
        procesadorS3.download_images_from_minio(images_names, data_folder_path)
    else:
        patch_names = ProcesadorCocoDataset.list_patches_w_ann_from_mongodb()
        procesadorS3.download_patches_from_minio(patch_names, data_folder_path)

    if with_annotations:
        if for_patches:
            ProcesadorCocoDataset.download_annotations_as_coco_from_mongodb(
                patches_names=patch_names,
                field_name=annotations_field_name,
                output_filename=annotations_output_filename,
            )
        else:
            ProcesadorCocoDataset.download_annotations_as_coco_from_mongodb(
                field_name=annotations_field_name,
                images_names=images_names,
                output_filename=annotations_output_filename,
            )

    return patch_names if for_patches else images_names


def download_partial_dataset(
    images_names: List[str] = None,
    patches_names: List[str] = None,
    folder_path: Optional[Path] = None,
    with_annotations: bool = True,
    annotations_field_name: str = "cvat",
    annotations_output_filename: str = None,
) -> List[str]:
    if bool(patches_names is None) == bool(images_names is None):  # xor
        raise ValueError("Se debe proporcionar una lista de nombres de parches o imágenes.")

    procesadorS3 = ProcesadorS3()
    if folder_path is None:
        Path(DOWNLOAD_RAW_DATASET_FOLDER).mkdir(parents=True, exist_ok=True)
        folder_path = DOWNLOAD_RAW_DATASET_FOLDER
    if annotations_output_filename is None:
        annotations_output_filename = folder_path / "labels.json"

    data_folder_path = folder_path / "data"
    Path(data_folder_path).mkdir(parents=True, exist_ok=True)
    if not patches_names:
        procesadorS3.download_images_from_minio(images_names, data_folder_path)
    else:
        procesadorS3.download_patches_from_minio(patches_names, data_folder_path)

    if with_annotations:
        if patches_names:
            ProcesadorCocoDataset.download_annotations_as_coco_from_mongodb(
                patches_names=patches_names,
                field_name=annotations_field_name,
                output_filename=annotations_output_filename,
            )
        else:
            ProcesadorCocoDataset.download_annotations_as_coco_from_mongodb(
                field_name=annotations_field_name,
                images_names=images_names,
                output_filename=annotations_output_filename,
            )

    return patches_names if patches_names else images_names


# TODO: Implementar la función para descargar el dataset de recortes
def download_cutouts_dataset(
    folder_path: Optional[Path] = None,
    with_annotations: bool = True,
    annotations_field_name: str = "cvat",
    annotations_output_filename: str = None,
) -> List[str]:

    procesadorS3 = ProcesadorS3()
    if folder_path is None:
        Path(DOWNLOAD_RAW_DATASET_FOLDER).mkdir(parents=True, exist_ok=True)
        folder_path = DOWNLOAD_RAW_DATASET_FOLDER
    if annotations_output_filename is None:
        annotations_output_filename = folder_path / "labels.json"

    data_folder_path = folder_path / "data"
    Path(data_folder_path).mkdir(parents=True, exist_ok=True)
