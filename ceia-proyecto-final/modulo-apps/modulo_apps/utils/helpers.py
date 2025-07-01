from typing import Tuple
import cv2
import numpy as np


def is_white_image(image: np.ndarray, threshold_percent=50, white_threshold=250) -> Tuple[bool, float]:
    """
    Detecta imágenes blancas basándose en un porcentaje de píxeles blancos.

    Args:
        image: Imagen en formato BGR (OpenCV)
        threshold_percent: Porcentaje de blanco requerido
        white_threshold: Valor mínimo para considerar un pixel como blanco (0-255)

    Returns:
        bool: True si la imagen es considerada blanca
        float: Porcentaje de blanco en la imagen
    """
    # Convertir a HSV para mejor manejo del brillo
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Criterios para considerar un pixel como blanco
    white_mask = cv2.inRange(
        hsv, np.array([0, 0, white_threshold]), np.array([180, 30, 255])  # Mínimo HSV
    )  # Máximo HSV

    total_pixels = image.shape[0] * image.shape[1]
    white_pixel_count = cv2.countNonZero(white_mask)
    white_percentage = (white_pixel_count / total_pixels) * 100

    return white_percentage > threshold_percent, white_percentage
