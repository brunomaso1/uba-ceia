import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Folders
ROOT_DIR = Path(__file__).resolve().parent
logger.debug(f"Directorio raiz: {ROOT_DIR}")
RESOURCES_DIR = ROOT_DIR / "resources"

try:
    load_dotenv("../.env.dev")
    logger.info("Cargando variables de entorno desde .env.dev.")
except FileNotFoundError:
    try:
        load_dotenv("../.env.prod")
        logger.info("Cargando variables de entorno desde .env.prod.")
    except FileNotFoundError:
        logger.info("Cargando variables desde el entorno del sistema.")

API_VERSION = "v1"

CORS_ALLOW_CREDENTIALS = os.environ.get("CORS_ALLOW_CREDENTIALS", "True")
CORS_ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
CORS_ALLOW_METHODS = os.environ.get("CORS_ALLOW_METHODS", "*").split(",")
CORS_ALLOW_HEADERS = os.environ.get("CORS_ALLOW_HEADERS", "*").split(",")

logger.debug(f"CORS_ALLOW_CREDENTIALS: {CORS_ALLOW_CREDENTIALS}")
logger.debug(f"CORS_ALLOW_ORIGINS: {CORS_ALLOW_ORIGINS}")
logger.debug(f"CORS_ALLOW_METHODS: {CORS_ALLOW_METHODS}")
logger.debug(f"CORS_ALLOW_HEADERS: {CORS_ALLOW_HEADERS}")

PORT = int(os.environ.get("PORT", "80"))

# Names
MODEL_NAME = "coco_palm_dataset_v1.0_palm_detection_yolo11n_640_1705a3dd294c499e8cfd8db7415a20c0.pt"

# Paths
MODEL_PATH = RESOURCES_DIR / "ia_models" / MODEL_NAME

# Model parameters
TARGET_IMG_SIZE_WH = (640, 640)  # Width, Height

