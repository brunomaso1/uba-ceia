# Dependencias del sistema
from dataclasses import dataclass
from pathlib import Path
from re import VERBOSE
from tabnanny import verbose
from typing import Any, Optional, Literal

# Dependencias propias
from modulo_utilidades.core.labeling.procesador_anotaciones_coco_dataset_core import (
    create_coco_annotations_from_detections,
)

# Dependencias de terceros
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results
from supervision import Detections, InferenceSlicer, OverlapFilter, BoxAnnotator, LabelAnnotator

VERBOSE = False # Cambiar a True para ver el log de cada paso del predictor, o a False para solo ver los resultados finales.


@dataclass
class PredictionResult:
    """Clase para manejar los resultados de las predicciones del modelo."""

    detections: Results | Detections
    img_size_hw: tuple[int, int]

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

    def filter_by_nms(self, iou_threshold: float = 0.5, class_agnostic: bool = False) -> "PredictionResult":
        if self.is_empty():
            return self

        filtered_detections = self.detections.with_nms(threshold=iou_threshold, class_agnostic=class_agnostic)
        return PredictionResult(filtered_detections, self.img_size_hw)

    def filter_by_nmm(self, iou_threshold: float = 0.5, class_agnostic: bool = False) -> "PredictionResult":
        if self.is_empty():
            return self

        filtered_detections = self.detections.with_nmm(threshold=iou_threshold, class_agnostic=class_agnostic)
        return PredictionResult(filtered_detections, self.img_size_hw)

    def filter_by_square_ratio(self, min_ratio: float = 0.7) -> "PredictionResult":
        if self.is_empty():
            return self

        ratios = np.array([self._square_ratio(box) for box in self.detections.xyxy])
        mask = ratios >= min_ratio
        filtered_detections = self.detections[mask]
        return PredictionResult(filtered_detections, self.img_size_hw)

    def filter_by_containment(
        self, threshold: float = 0.8, class_agnostic: bool = False, check_confidence: bool = True
    ) -> "PredictionResult":
        """
        Elimina cajas cuya área esté contenida en otra caja con mayor confianza.
        threshold: proporción mínima de contención para considerar duplicado.
        class_agnostic: si es True, no verifica que las cajas sean de la misma clase.
        check_confidence: si es False, no verifica que la caja contenedora tenga mayor confianza.
        """
        if self.is_empty():
            return self

        boxes = self.detections.xyxy
        confs = self.detections.confidence
        class_ids = self.detections.class_id if hasattr(self.detections, "class_id") else None
        keep = np.ones(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j or not keep[i]:
                    continue
                # Si no es class_agnostic, verificar que sean de la misma clase
                if not class_agnostic and class_ids is not None:
                    if class_ids[i] != class_ids[j]:
                        continue

                cr = self._containment_ratio(boxes[i], boxes[j])

                # Verificar contención y opcionalmente la confianza
                containment_condition = cr >= threshold
                confidence_condition = not check_confidence or confs[j] >= confs[i]

                if containment_condition and confidence_condition:
                    # Si la caja i está contenida en j, y (si check_confidence=True) j tiene más confianza → descartar i
                    keep[i] = False
                    break

        filtered_detections = self.detections[keep]
        return PredictionResult(filtered_detections, self.img_size_hw)

    def as_pandas(self) -> pd.DataFrame:
        """Convierte las detecciones a un DataFrame de pandas."""
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

    def get_annotated_image(self, image: np.ndarray, include_labels: bool = True) -> np.ndarray:
        """
        Genera una imagen anotada con las detecciones realizadas.
        Toma la imagen original proporcionada y la anota con los resultados de las
        predicciones almacenadas en self.detections. Se dibujan cuadros delimitadores
        (bounding boxes) alrededor de los objetos detectados y, opcionalmente, etiquetas
        con el nombre de la clase y la confianza de la predicción.
        Args:
            image (np.ndarray): La imagen original que será anotada con las predicciones.
                               Debe ser un array de numpy en formato BGR (si es de OpenCV).
            include_labels (bool, optional): Si True, incluye etiquetas de texto con el nombre
                                             de la clase y el nivel de confianza de cada detección.
                                             Por defecto es True.
        Returns:
            np.ndarray: La imagen anotada con los cuadros delimitadores y opcionalmente
                       las etiquetas de las detecciones. Si no hay detecciones, retorna
                       una copia de la imagen original sin modificaciones.
        Examples:
            >>> annotated_image = predictor.get_annotated_image(original_image)
            >>> annotated_image = predictor.get_annotated_image(original_image, include_labels=False)
        """
        """Devuelve la imagen anotada con las detecciones."""
        if self.detections.is_empty():
            return image.copy()
        annotated_frame = None
        if include_labels:
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(self.detections["class_name"], self.detections.confidence)
            ]

            label_annotator = LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=image.copy(), detections=self.detections, labels=labels)

        box_annotator = BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=image.copy() if annotated_frame is None else annotated_frame, detections=self.detections
        )
        return annotated_frame

    def as_coco_annotations(
        self,
        pic_name: str,
        categories: Optional[list[dict]] = None,
        output_file_path: Optional[Path] = None,
    ):
        if self.detections.is_empty():
            return []

        return create_coco_annotations_from_detections(
            detections=self.detections,
            image_size_hw=self.img_size_hw,
            pic_name=pic_name,
            categories=categories,
            output_file_path=output_file_path,
        )

    def _square_ratio(self, box: np.ndarray) -> float:
        """Calcula la relación de aspecto de una caja delimitadora."""
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w <= 0 or h <= 0:
            return 0.0
        return min(w, h) / max(w, h)

    def _containment_ratio(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """
        Calcula el containment ratio de box_a respecto a box_b. Ej:
        Si box_a está completamente dentro de box_b, el ratio es 1.0.
        Si box_a está 0.8 dentro de box_b, el ratio es 0.8.
        box_a, box_b: [x1, y1, x2, y2]
        """
        # Coordenadas de intersección
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = max(0, (box_a[2] - box_a[0])) * max(0, (box_a[3] - box_a[1]))
        if area_a == 0:
            return 0.0

        return inter_area / area_a


@dataclass
class DetectionModelPredictor:
    model: Path | Any
    target_img_size_wh: tuple[int, int] = (640, 640)
    overlap_ratio_wh: tuple[float, float] = (0.4, 0.4)
    overlap_filter_name: Literal["NMS", "NMM", "NONE"] = "NONE"
    iou_threshold: float = 0.5

    def __post_init__(self):

        if isinstance(self.model, Path):
            self.model = self._load_model(self.model)
        if not hasattr(self.model, "predict"):
            raise ValueError("El modelo debe tener un método 'predict'.")
        """ Nota importante sobre overlap_filter_name:
        En la documentación de Supervision, los valores posibles son:
        - NONE: Do not filter detections based on overlap.
        - NON_MAX_SUPPRESSION: Filter detections using non-max suppression. This means, detections that overlap by more than a set threshold will be discarded, except for the one with the highest confidence.
        - NON_MAX_MERGE: Merge detections with non-max merging. This means, detections that overlap by more than a set threshold will be merged into a single detection.
        """
        overlap_filter_name = self.overlap_filter_name.upper()
        match overlap_filter_name:
            case "NMS":
                self.overlap_filter = OverlapFilter.NON_MAX_SUPPRESSION
            case "NMM":
                self.overlap_filter = OverlapFilter.NON_MAX_MERGE
            case "NONE":
                self.overlap_filter = OverlapFilter.NONE
            case _:
                raise ValueError(f"Unsupported overlap filter: {self.overlap_filter_name}")
        self.overlap_wh: tuple[int, int] = (
            int(self.overlap_ratio_wh[0] * self.target_img_size_wh[0]),
            int(self.overlap_ratio_wh[1] * self.target_img_size_wh[1]),
        )

    def predict(self, image: np.ndarray | Path) -> PredictionResult:
        if isinstance(image, Path):
            if not image.exists():
                raise FileNotFoundError(f"La imagen {image} no existe.")
            image = cv2.imread(str(image))

        if image is None:
            raise ValueError("La imagen no se pudo cargar correctamente.")

        img_size_hw = (image.shape[0], image.shape[1])

        slicer = InferenceSlicer(
            callback=self._slicer_callback,
            slice_wh=(self.target_img_size_wh[0], self.target_img_size_wh[1]),
            overlap_wh=self.overlap_wh,
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
        result = self.model.predict(img_slice, verbose=VERBOSE)[0]
        detections = Detections.from_ultralytics(result)
        return detections
