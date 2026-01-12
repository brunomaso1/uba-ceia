# Dependencias del sistema
from pathlib import Path
import json

# Dependencias locales
from modulo_ia.core_config import core_settings

# Dependencias de terceros
from loguru import logger as LOGGER
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

LOGGER.debug("Loading modulo-ia config...")

PROJECT_DIR = core_settings.folders.project_dir

# Env resolution.
env_file_path = PROJECT_DIR / ".env"
env_file = str(env_file_path)
if env_file_path.exists():
    LOGGER.warning(f"Using .env file at {env_file_path.resolve()} for configuration.")

class FoldersConfig(BaseModel):
    data_folder: Path = PROJECT_DIR / "data"
    models_folder: Path = PROJECT_DIR / "models"
    palm_detection_yolov11_folder: Path = PROJECT_DIR / "notebooks" / "deteccion_palmeras" / "yolov11"
    rpw_detection_yolov11_folder: Path = PROJECT_DIR / "notebooks" / "deteccion_picudo_rojo" / "yolov11"

    @computed_field
    def raw_data_folder(self) -> Path:
        return self.data_folder / "raw"

    @computed_field
    def external_data_folder(self) -> Path:
        return self.data_folder / "external"

    @computed_field
    def interim_data_folder(self) -> Path:
        return self.data_folder / "interim"

    @computed_field
    def processed_data_folder(self) -> Path:
        return self.data_folder / "processed"

    @computed_field
    def temp_data_folder(self) -> Path:
        return self.data_folder / "temp"


class NamesConfig(BaseModel):
    palm_dataset_name: str = "coco_palm_dataset"


class RawVersionsConfig(BaseModel):
    v11: str = "v1.1"


class ProcessedVersionsConfig(BaseModel):
    v111: str = "v1.1.1"
    v112: str = Field(default="v1.1.2", description="Undersampling = 5000")


class DatasetsProcessedFormatConfig(BaseModel):
    yolo: str = "yolo"
    huggingface: str = "huggingface"


class FiftyoneConfig(BaseModel):
    host: str = "localhost"
    port: int = 5151
    schema: str = "http"
    data_quality_folder: Path = PROJECT_DIR.parent / "modulo-calidad-datos" / "fiftyone" / "data"

    @computed_field
    def address(self) -> str:
        return f"{self.schema}://{self.host}:{self.port}"


class MLFlowConfig(BaseModel):
    host: str = "localhost"
    port: int = 5000
    tracking_username: str
    tracking_password: str
    schema: str = "http"

    @computed_field
    def tracking_uri(self) -> str:
        return f"{self.schema}://{self.host}:{self.port}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file,
        env_nested_delimiter="_",
        env_nested_max_split=1,
        env_file_encoding="utf-8",
        extra="allow",
    )

    environment: str = "dev"
    seed: int = 42

    folders: FoldersConfig = Field(default_factory=FoldersConfig)
    names: NamesConfig = Field(default_factory=NamesConfig)
    raw_dataset_versions: RawVersionsConfig = Field(default_factory=RawVersionsConfig)
    processed_dataset_versions: ProcessedVersionsConfig = Field(default_factory=ProcessedVersionsConfig)
    processed_dataset_format: DatasetsProcessedFormatConfig = Field(default_factory=DatasetsProcessedFormatConfig)
    fiftyone: FiftyoneConfig = Field(default_factory=FiftyoneConfig)
    mlflow: MLFlowConfig = Field(default_factory=MLFlowConfig)


settings = Settings()
LOGGER.debug(f"Settings (modulo-ia) loaded: {json.dumps(settings.model_dump(), indent=2, default=str)}")

