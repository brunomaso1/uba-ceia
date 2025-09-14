import torch
from ultralytics import YOLO
import numpy as np

from fastapi_backend.detectors.detector import Detector
from fastapi_backend.config import YOLOV11X_MODEL_PATH


class YoloDetector(Detector):
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(YOLOV11X_MODEL_PATH)
        super().__init__(model, device)

    def predict(self, image: np.ndarray) -> dict:
        self.model.to(self.device)
        return self.model(image)
