from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results

from supervision import Detections, InferenceSlicer, OverlapFilter, BoxAnnotator

import modulo_apps.labeling.procesador_anotaciones_coco_dataset as CocoDatasetProcessor


@dataclass
class PredictionResult:
    detections: Results | Detections
    img_size_hw: Tuple[int, int]

    def __post_init__(self):
        if isinstance(self.detections, Results):
            self.detections: Detections = Detections.from_ultralytics(self.detections)

    def is_empty(self) -> bool:
        return self.detections.is_empty()

    def filter_by_confidence(self, min_confidence: float = 0.5) -> "PredictionResult":
        if self.is_empty():
            return self

        filtered_detections = self.detections[self.detections.confidence >= min_confidence]
        return PredictionResult(filtered_detections, self.img_size_hw)

    def as_pandas(self) -> pd.DataFrame:
        if self.detections.is_empty():
            return pd.DataFrame()

        data = {
            "x1": self.detections.xyxy[:, 0],
            "y1": self.detections.xyxy[:, 1],
            "x2": self.detections.xyxy[:, 2],
            "y2": self.detections.xyxy[:, 3],
            "confidence": self.detections.confidence,
            "class_id": self.detections.class_id,
            "class_name": self.detections.data["class_name"],
        }
        return pd.DataFrame(data)

    def get_annotated_image(self, image: np.ndarray) -> np.ndarray:
        if self.detections.is_empty():
            return image.copy()
        return BoxAnnotator().annotate(scene=image.copy(), detections=self.detections)

    def as_coco_annotations(
        self, pic_name: str, should_download: Optional[bool] = None, output_filename: Optional[Path] = None
    ) -> List[dict]:
        if self.detections.is_empty():
            return []

        kwargs = {}
        if output_filename is not None:
            kwargs["output_filename"] = output_filename
        if should_download is not None:
            kwargs["should_download"] = should_download

        return CocoDatasetProcessor.create_coco_annotations_from_detections(
            detections=self.detections, image_size_hw=self.img_size_hw, pic_name=pic_name, **kwargs
        )


@dataclass
class DetectionModelPredictor:
    model: Path | Any
    target_img_size_wh: Tuple[int, int] = (640, 640)
    overlap_ratio_wh: Tuple[float, float] = (0.4, 0.4)
    overlap_filter: OverlapFilter = OverlapFilter.NON_MAX_MERGE
    iou_threshold: float = 0.5

    def __post_init__(self):
        if not hasattr(self.model, "predict"):
            raise ValueError("El modelo debe tener un método 'predict'.")
        if isinstance(self.model, Path):
            self.model = self._load_model(self.model)
        self.overlap_wh: Tuple[int, int] = (
            int(self.overlap_ratio_wh[0] * self.target_img_size_wh[0]),
            int(self.overlap_ratio_wh[1] * self.target_img_size_wh[1]),
        )

    def predict(self, image: np.ndarray | Path, should_slice: bool = True) -> PredictionResult:
        if isinstance(image, Path):
            if not image.exists():
                raise FileNotFoundError(f"La imagen {image} no existe.")
            image = cv2.imread(str(image))

        if image is None:
            raise ValueError("La imagen no se pudo cargar correctamente.")

        img_size_hw = (image.shape[0], image.shape[1])

        if not should_slice:
            results = self.model.predict(image, verbose=False)[0].cpu()
            return PredictionResult(results, img_size_hw)

        slicer = InferenceSlicer(
            callback=self._slicer_callback,
            slice_wh=(self.target_img_size_wh[0], self.target_img_size_wh[1]),
            overlap_wh=self.overlap_wh,
            overlap_ratio_wh=None,
            overlap_filter=self.overlap_filter,
            iou_threshold=self.iou_threshold,
        )

        detections = slicer(image)
        return PredictionResult(detections, img_size_hw)

    def _load_model(self, model_path: Path) -> Any:
        if not model_path.exists():
            raise FileNotFoundError(f"El modelo {model_path} no existe.")

        model = YOLO(model_path)
        return model

    def _slicer_callback(self, img_slice: np.ndarray) -> Detections:
        result = self.model.predict(img_slice)[0]
        detections = Detections.from_ultralytics(result)
        return detections
