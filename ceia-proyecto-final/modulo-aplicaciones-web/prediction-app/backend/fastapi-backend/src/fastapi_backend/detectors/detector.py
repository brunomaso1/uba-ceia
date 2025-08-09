from typing import override
import numpy as np


class Detector:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @override
    def predict(self, image: np.ndarray) -> dict[str, any]:
        # To be implemented in subclasses
        raise NotImplementedError("Subclasses should implement this method")
