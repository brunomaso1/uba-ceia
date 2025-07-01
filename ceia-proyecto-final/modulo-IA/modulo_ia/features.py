from pathlib import Path
from typing import List, Optional

from loguru import logger as LOGGER

from modulo_ia.config import config as CONFIG

from modulo_apps.database_comunication.mongodb_client import mongodb as DB

import modulo_apps.s3_comunication.procesador_s3 as ProcesadorS3
import modulo_apps.labeling.procesador_anotaciones_mongodb as ProcesadorCocoDataset
import modulo_apps.labeling.procesador_recortes as ProcesadorRecortes

import typer

from pyparsing import Union
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

RAW_DATA_FOLDER = CONFIG.folders.raw_data_folder
EXTERNAL_DATA_FOLDER = CONFIG.folders.external_data_folder
INTERIM_DATA_FOLDER = CONFIG.folders.interim_data_folder
PROCESSED_DATA_FOLDER = CONFIG.folders.processed_data_folder

FULL_DATASET_NAME = CONFIG.names.detection_dataset_name
FULL_DATASET_VERSION = CONFIG.versions.detection_dataset_version

PARTIAL_DATASET_NAME = CONFIG.names.partial_dataset_name
PARTIAL_DATASET_VERSION = CONFIG.versions.detection_dataset_version

CUTOUTS_DATASET_NAME = CONFIG.names.cutouts_dataset_name
CUTOUTS_DATASET_VERSION = CONFIG.versions.cutouts_dataset_version

app = typer.Typer()


class SwinV2Transforms:
    """Clase para aplicar transformaciones a las imágenes usando la configuración de un `ViTImageProcessor`.

    Args:
        image_processor (Union[ViTImageProcessor, ViTImageProcessorFast]): Procesador de imágenes ViT para obtener la configuración de las transformaciones.
    """

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
def apply_offline_transforms(
    folder_path: Path,
    transforms: List[torch.nn.Module],
    train: bool = True,
) -> torch.Tensor:
    raise NotImplementedError()


if __name__ == "__main__":
    app()
