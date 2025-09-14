import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# FOLDERS
ROOT_DIR = Path(__file__).resolve().parent
logger.debug(f"Directorio raiz: {ROOT_DIR}")
RESOURCES_DIR = ROOT_DIR / "resources"
MODELS_DIR = RESOURCES_DIR / "ia_models"

# ENVIRONMENT VARIABLES
try:
    load_dotenv("../.env.dev")
    logger.info("Cargando variables de entorno desde .env.dev.")
except FileNotFoundError:
    try:
        load_dotenv("../.env.prod")
        logger.info("Cargando variables de entorno desde .env.prod.")
    except FileNotFoundError:
        logger.info("Cargando variables desde el entorno del sistema.")

# API CONFIGURATION
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

# MODELS CONFIGURATION
# Palm detection model
PALM_MODEL_NAME = "coco_palm_dataset_v1.0_palm_detection_yolo11x_640_b7218a073a3942339689b2e7f4e0b543"
PALM_MODEL_CLASS_NAMES = {0: "palmera"}
PALM_MODEL_PATH = MODELS_DIR / f"{PALM_MODEL_NAME}.pt"
PALM_MODEL_OVERLAP_FILTER = "NMS"
PALM_MODEL_NMS_THRESHOLD = 0.25
PALM_MODEL_MIN_CONFIDENCE = 0.75

# RPW detection model
RPW_MODEL_NAME = "coco_palm_dataset_v1.0_rpw_detection_yolo11x_640_stage_training_4b1667b39a5140749b939fc7ec743b84"
RPW_MODEL_CLASS_NAMES = {0: "palmera-sana", 1: "palmera-infectada", 2: "palmera-muerta"}
RPW_MODEL_PATH = MODELS_DIR / f"{RPW_MODEL_NAME}.pt"
RPW_MODEL_OVERLAP_FILTER = "NMM"
RPW_MODEL_NMM_THRESHOLD = 0.75
RPW_MODEL_MIN_CONFIDENCE = 0.75

# COMMON MODEL CONFIGURATION
TARGET_IMG_SIZE_WH = (640, 640)
OVERLAP_RATIO_WH = (0.4, 0.4)
