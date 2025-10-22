# Dependencias del sistema
import json, os
from pathlib import Path

# Dependencias de terceros
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configuraciones
PROJECT_DIR: Path = Path(__file__).parent.parent.resolve()
ROOT_DIR: Path = PROJECT_DIR / "app"
RESOURCES_DIR: Path = ROOT_DIR / "resources"
MODELS_DIR: Path = RESOURCES_DIR / "ia_models"
OPENCV_IO_MAX_IMAGE_PIXELS = 50000 * 50000  # Para imágenes grandes, ej: barrio3Ombues_20180801_dji_pc_3cm.jpg
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(OPENCV_IO_MAX_IMAGE_PIXELS)

# Env resolution.
env_file_path: Path = PROJECT_DIR / ".env"
env_file: str = str(env_file_path)
if env_file_path.exists():
    logger.warning(f"Using .env file at {env_file_path.resolve()} for configuration.")


class FoldersConfig(BaseSettings):
    root_dir: Path = ROOT_DIR
    resources_dir: Path = RESOURCES_DIR
    models_dir: Path = MODELS_DIR


class CorsConfig(BaseSettings):
    allow_credentials: bool = True
    allow_origins: list[str] = ["*"]
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]


class KeycloakConfig(BaseSettings):
    url: str = "http://localhost:7000"
    realm: str = "mis-palmeras-app"
    client_id: str = "prediction-app-backend"
    client_secret: str


class PalmModelConfig(BaseSettings):
    model_name: str = "palm_detection_yolo11x_640_a3a50bd4646e4044bed83f02f8bb03f4"
    model_path: Path = MODELS_DIR / f"{model_name}.pt"
    class_names: dict[int, str] = {0: "palmera"}
    min_ratio: float = 0.8
    nms_iou_threshold: float = 0.8
    containerment_threshold: float = 0.8
    confidence: float = 0.5


class RPWModelConfig(BaseSettings):
    model_name: str = "rpw_detection_yolo11x_640_freeze_learning_5008788e1d99471e97b877bca45f169f"
    model_path: Path = MODELS_DIR / f"{model_name}.pt"
    class_names: dict[int, str] = {0: "palmera-sana", 1: "palmera-infectada", 2: "palmera-muerta"}
    min_ratio: float = 0.8
    nms_iou_threshold: float = 0.8
    containerment_threshold: float = 0.8
    confidence: float = 0.5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file,
        env_nested_delimiter="_",
        env_nested_max_split=1,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "dev"
    api_version: str = "v1"
    timeout_keep_alive: int = 600
    port: int = 8000

    target_img_size_wh: tuple[int, int] = (640, 640)
    overlap_ratio_wh: tuple[float, float] = (0.4, 0.4)

    folders: FoldersConfig = Field(default_factory=FoldersConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    keycloak: KeycloakConfig = Field(default_factory=KeycloakConfig)
    palm_model: PalmModelConfig = Field(default_factory=PalmModelConfig)
    rpw_model: RPWModelConfig = Field(default_factory=RPWModelConfig)


settings = Settings()
logger.debug(f"Settings loaded: {json.dumps(settings.model_dump(), indent=2, default=str)}")
