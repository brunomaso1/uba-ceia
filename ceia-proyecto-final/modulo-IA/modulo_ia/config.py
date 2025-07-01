import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass

OPENCV_IO_MAX_IMAGE_PIXELS = 50000 * 50000  # Para imágenes grandes, ej: barrio3Ombues_20180801_dji_pc_3cm.jpg
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(OPENCV_IO_MAX_IMAGE_PIXELS)

ROOT_DIR = Path(__file__).resolve().parent

# Construir las rutas con respecto al directorio raíz.
env_dev_path = ROOT_DIR / ".env.dev"
env_prod_path = ROOT_DIR / ".env.prod"

logger.info(f"Directorio de configuración raíz: {ROOT_DIR}")

if env_dev_path.exists():
    load_dotenv(env_dev_path)
    logger.info("Variables de entorno cargadas desde .env.dev.")
elif env_prod_path.exists():
    load_dotenv(env_prod_path)
    logger.info("Variables de entorno cargadas desde .env.prod.")
else:
    logger.info("No se encontraron archivos .env. Cargando variables desde el entorno del sistema.")


@dataclass
class FoldersConfig:
    data_folder: Path = ROOT_DIR / "data"
    models_folder: Path = ROOT_DIR / "models"

    def __post_init__(self):
        self.raw_data_folder: Path = self.data_folder / "raw"
        self.external_data_folder: Path = self.data_folder / "external"
        self.interim_data_folder: Path = self.data_folder / "interim"
        self.processed_data_folder: Path = self.data_folder / "processed"

@dataclass
class NamesConfig:
    detection_dataset_name: str = "coco_rpw_dataset"
    cutouts_dataset_name: str = "coco_cutouts_dataset"
    partial_dataset_name: str = "coco_rpw_dataset_partial"


@dataclass
class VersionsConfig:
    detection_dataset_version: str = "v1.0"
    cutouts_dataset_version: str = "v1.0"
    partial_dataset_name: str = "v1.0"

class Config:
    """Clase principal de configuración del sistema"""

    def __init__(self):
        # Configuración general
        self.environment = os.getenv("ENVIRONMENT", "dev")
        self.seed = 42

        self.folders = FoldersConfig()
        self.names = NamesConfig()
        self.versions = VersionsConfig()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(1)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except Exception:
    pass

# Instancia global de configuración
config = Config()
