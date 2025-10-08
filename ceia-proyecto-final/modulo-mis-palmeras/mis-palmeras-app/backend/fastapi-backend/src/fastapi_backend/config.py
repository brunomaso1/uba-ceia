import json
from pathlib import Path
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent
env_file_path = ROOT_DIR / ".env"
env_file = str(env_file_path)

if env_file_path.exists():
    logger.warning(f"Using .env file at {env_file_path.resolve()} for configuration.")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8")

    resources_dir: Path = ROOT_DIR / "resources"
    models_dir: Path = resources_dir / "ia_models"

    api_version: str = "v1"
    port: int = 80

    cors_allow_credentials: bool = True
    cors_allow_origins: list[str] = ["*"]
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    keycloak_url: str = "http://localhost:7000"
    keycloak_realm: str = "mis-palmeras-app"
    keycloak_client_id: str = "prediction-app-backend"
    keycloak_client_secret: str

    target_img_size_wh: tuple[int, int] = (640, 640)
    overlap_ratio_wh: tuple[float, float] = (0.4, 0.4)

    palm_model_name: str = "palm_detection_yolo11x_640_a3a50bd4646e4044bed83f02f8bb03f4"
    palm_model_path: Path = models_dir / f"{palm_model_name}.pt"
    palm_model_class_names: dict[int, str] = {0: "palmera"}
    palm_min_ratio: float = 0.8
    palm_nms_iou_threshold: float = 0.8
    palm_containerment_threshold: float = 0.8
    palm_confidence: float = 0.5

    rpw_model_name: str = "rpw_detection_yolo11x_640_freeze_learning_5008788e1d99471e97b877bca45f169f"
    rpw_model_path: Path = models_dir / f"{rpw_model_name}.pt"
    rpw_model_class_names: dict[int, str] = {0: "palmera-sana", 1: "palmera-infectada", 2: "palmera-muerta"}
    rpw_min_ratio: float = 0.8
    rpw_nms_iou_threshold: float = 0.8
    rpw_containerment_threshold: float = 0.8
    rpw_confidence: float = 0.5

    timeout_keep_alive: int = 600


settings = Settings()
logger.debug(f"Settings loaded: {json.dumps(settings.model_dump(), indent=2, default=str)}")
