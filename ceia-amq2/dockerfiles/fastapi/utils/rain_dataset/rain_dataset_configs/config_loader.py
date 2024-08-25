import os
import json
from dotenv import load_dotenv
from airflow.models import Variable

class RainDatasetConfigs:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RainDatasetConfigs, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Método privado para cargar la configuración desde diferentes fuentes."""
        # Cargar las variables desde un archivo .env
        load_dotenv()

        # Variables de entorno
        self.MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

        # Variables desde Airflow
        self.RAW_DATA_FOLDER = Variable.get("RAW_DATA_FOLDER")
        self.DATA_FOLDER = Variable.get("DATA_FOLDER")
        self.S3_PREPROCESED_DATA_FOLDER = Variable.get("S3_PREPROCESED_DATA_FOLDER")
        self.S3_RAW_DATA_FOLDER = Variable.get("S3_RAW_DATA_FOLDER")
        self.S3_INFO_DATA_FOLDER = Variable.get("S3_INFO_DATA_FOLDER")
        self.S3_PIPES_DATA_FOLDER = Variable.get("S3_PIPES_DATA_FOLDER")
        self.S3_FINAL_DATA_FOLDER = Variable.get("S3_FINAL_DATA_FOLDER")
        self.INFO_DATA_FOLDER = Variable.get("INFO_DATA_FOLDER")
        self.PIPES_DATA_FOLDER = Variable.get("PIPES_DATA_FOLDER")
        self.TEST_SIZE = float(Variable.get("TEST_SIZE"))
        self.MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")

        # Otras constantes fijas
        self.DATASET_NAME = "rain"
        self.DATASET_NAME_W_EXTENSION = self.DATASET_NAME + ".csv"
        self.BUCKET_DATA = "data"
        self.BOTO3_CLIENT = "s3"
        self.MLFLOW_EXPERIMENT_NAME = "Rain Dataset"
        self.X_TRAIN_NAME = "X_train.csv"
        self.X_TEST_NAME = "X_test.csv"
        self.Y_TRAIN_NAME = "y_train.csv"
        self.Y_TEST_NAME = "y_test.csv"

        # Constantes derivadas
        self.S3_RAW_DATA_PATH = self.S3_RAW_DATA_FOLDER + self.DATASET_NAME_W_EXTENSION
        self.S3_DF_PATH = self.S3_PREPROCESED_DATA_FOLDER + self.DATASET_NAME_W_EXTENSION
        self.S3_GDF_LOCATIONS_PATH = self.INFO_DATA_FOLDER + "gdf_locations.json"
        self.S3_COLUMNS_PATH = self.INFO_DATA_FOLDER + "columnsTypes.json"
        self.S3_INPUT_PIPELINE_PATH = self.PIPES_DATA_FOLDER + "inputs_pipeline.pkl"
        self.S3_TARGET_PIPELINE_PATH = self.PIPES_DATA_FOLDER + "target_pipeline.pkl"

    def get(self, key):
        """Método para obtener la configuración mediante una clave."""
        return getattr(self, key, None)
