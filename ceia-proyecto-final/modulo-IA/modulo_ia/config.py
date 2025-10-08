import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger as LOGGER
from dataclasses import dataclass

OPENCV_IO_MAX_IMAGE_PIXELS = 50000 * 50000  # Para imágenes grandes, ej: barrio3Ombues_20180801_dji_pc_3cm.jpg
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(OPENCV_IO_MAX_IMAGE_PIXELS)

# Colab
# PROJECT_DIR = Path('/content/drive/MyDrive/ModuloIA/modulo_ia')
PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent

env_dev_path = PROJECT_DIR / ".env.dev"
env_prod_path = PROJECT_DIR / ".env.prod"

LOGGER.info(f"Directorio de configuración raíz: {ROOT_DIR}")
LOGGER.info(f"Directorio de configuración del proyecto: {PROJECT_DIR}")

if env_dev_path.exists():
    load_dotenv(env_dev_path)
    LOGGER.info("Variables de entorno cargadas desde .env.dev.")
elif env_prod_path.exists():
    load_dotenv(env_prod_path)
    LOGGER.info("Variables de entorno cargadas desde .env.prod.")
else:
    LOGGER.info("No se encontraron archivos .env. Cargando variables desde el entorno del sistema.")


@dataclass
class FoldersConfig:
    data_folder: Path = ROOT_DIR / "data"
    models_folder: Path = ROOT_DIR / "models"
    palm_detection_yolov11_folder: Path = ROOT_DIR / "notebooks" / "deteccion_palmeras" / "yolov11"
    rpw_detection_yolov11_folder: Path = ROOT_DIR / "notebooks" / "deteccion_picudo_rojo" / "yolov11"

    def __post_init__(self):
        self.raw_data_folder: Path = self.data_folder / "raw"
        self.external_data_folder: Path = self.data_folder / "external"
        self.interim_data_folder: Path = self.data_folder / "interim"
        self.processed_data_folder: Path = self.data_folder / "processed"
        self.temp_data_folder: Path = self.data_folder / "temp"


@dataclass
class NamesConfig:
    palm_dataset_name: str = "coco_palm_dataset"


@dataclass
class RawVersionsConfig:
    v11: str = "v1.1"

@dataclass
class ProcessedVersionsConfig:
    v111: str = "v1.1.1"
    v112: str = "v1.1.2" # Undersampling = 5000


@dataclass
class DatasetsProcessedFormatConfig:
    yolo: str = "yolo"
    huggingface: str = "huggingface"


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
    mlflow_tracking_username: str
    mlflow_tracking_password: str
    schema: str

    def __post_init__(self):
        self.tracking_uri: str = f"{self.schema}://{self.host}:{self.port}"


class Config:
    """Clase principal de configuración del sistema"""

    def __init__(self):
        # Configuración general
        self.environment = os.getenv("ENVIRONMENT", "dev")
        self.seed = 42

        self.folders = FoldersConfig()
        self.names = NamesConfig()
        self.raw_versions = RawVersionsConfig()
        self.processed_versions = ProcessedVersionsConfig()
        self.datasets_processed_format = DatasetsProcessedFormatConfig()
        self.fiftyone = self._get_fiftyone_config()
        self.mlflow = self._get_mlflow_config()

    def _get_fiftyone_config(self) -> FiftyoneConfig:
        """Obtiene la configuración de FiftyOne desde las variables de entorno"""
        return FiftyoneConfig(host=os.getenv("FIFTYONE_HOST", "localhost"), port=int(os.getenv("FIFTYONE_PORT", 5151)))

    def _get_mlflow_config(self) -> MLFlowConfig:
        """Obtiene la configuración de MLflow desde las variables de entorno"""
        return MLFlowConfig(
            host=os.getenv("MLFLOW_HOST", "localhost"),
            port=int(os.getenv("MLFLOW_PORT", 5000)),
            schema=os.getenv("MLFLOW_SCHEMA", "http"),
            mlflow_tracking_username=os.getenv("MLFLOW_TRACKING_USERNAME", ""),
            mlflow_tracking_password=os.getenv("MLFLOW_TRACKING_PASSWORD", ""),
        )


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    LOGGER.remove(1)
    LOGGER.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except Exception:
    pass

# Instancia global de configuración
config = Config()
