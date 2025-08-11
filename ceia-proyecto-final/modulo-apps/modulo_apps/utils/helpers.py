import sys
import cv2
import numpy as np
import loguru as LOGGER


def is_white_image(image: np.ndarray, threshold_percent=50, white_threshold=250) -> tuple[bool, float]:
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

def set_log_to_file(log_file: str, level: str) -> None:
    """
    Configura el logger para escribir en un archivo.

    Args:
        log_file: Ruta del archivo de log
    """
    try:
        LOGGER.remove()  # Elimina cualquier configuración previa
    finally:
        LOGGER.add(log_file, rotation="500 MB", level=level, backtrace=True, diagnose=True)
        LOGGER.info(f"Logger configurado para escribir en {log_file}")
        LOGGER.debug("Logger configurado correctamente.")