import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass

OPENCV_IO_MAX_IMAGE_PIXELS = 50000 * 50000  # Para imágenes grandes, ej: barrio3Ombues_20180801_dji_pc_3cm.jpg
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(OPENCV_IO_MAX_IMAGE_PIXELS)

from modulo_apps.config import config as MODULO_APPS_CONFIG

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent

# Construir las rutas con respecto al directorio raíz.
env_dev_path = PROJECT_DIR / ".env.dev"
env_prod_path = PROJECT_DIR / ".env.prod"

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
    palm_detection_dataset_name: str = "coco_palm_detection_dataset"
    cutouts_dataset_name: str = "coco_cutouts_dataset"
    partial_dataset_name: str = "coco_rpw_dataset_partial"
    rpw_detection_dataset_name: str = "rpw_dataset_name"


@dataclass
class VersionsConfig:
    detection_dataset_version: str = "v1.0"
    cutouts_dataset_version: str = "v1.0"
    partial_dataset_name: str = "v1.0"
    rpw_dataset_version: str = "v1.0"


@dataclass
class FiftyoneConfig:
    host: str
    port: int
    data_quality_folder: Path = ROOT_DIR.parent / "modulo-calidad-datos" / "fiftyone" / "data"

    def __post_init__(self):
        self.address: str = f"http://{self.host}:{self.port}"


@dataclass
class MLFlowConfig:
    host: str
    port: int

    def __post_init__(self):
        self.tracking_uri: str = f"http://{self.host}:{self.port}"


class Config:
    """Clase principal de configuración del sistema"""

    def __init__(self):
        # Configuración general
        self.environment = os.getenv("ENVIRONMENT", "dev")
        self.seed = 42

        self.folders = FoldersConfig()
        self.names = NamesConfig()
        self.versions = VersionsConfig()
        self.fiftyone = self._get_fiftyone_config()
        self.mlflow = self._get_mlflow_config()

        self.coco_dataset = MODULO_APPS_CONFIG.coco_dataset

    def _get_fiftyone_config(self) -> FiftyoneConfig:
        """Obtiene la configuración de FiftyOne desde las variables de entorno"""
        return FiftyoneConfig(host=os.getenv("FIFTYONE_HOST", "localhost"), port=int(os.getenv("FIFTYONE_PORT", 5151)))

    def _get_mlflow_config(self) -> MLFlowConfig:
        """Obtiene la configuración de MLflow desde las variables de entorno"""
        return MLFlowConfig(host=os.getenv("MLFLOW_HOST", "localhost"), port=int(os.getenv("MLFLOW_PORT", 5000)))


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
