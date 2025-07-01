# Enum para los formatos de dataset
from enum import Enum


class DatasetFormat(str, Enum):
    YOLO = "yolo"
    HUGGINGFACE = "huggingface"
    COCO = "coco"

    def __str__(self):
        return self.value

    @classmethod
    def list_formats(cls):
        """Retorna una lista de todos los formatos disponibles"""
        return [fmt.value for fmt in cls]

    @classmethod
    def from_string(cls, format_str: str):
        """Crea un enum desde un string, case-insensitive"""
        format_str = format_str.lower()
        for fmt in cls:
            if fmt.value.lower() == format_str:
                return fmt
        raise ValueError(f"Formato '{format_str}' no válido. Formatos disponibles: {cls.list_formats()}")
