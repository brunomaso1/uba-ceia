import datetime
import os
from dotenv import load_dotenv
from airflow.models import Variable


class RainDatasetConfigs:
    """
    Clase Singleton encargada de manejar la configuración relacionada con el dataset de Rain.
    Contiene las variables de entonro cargadas, así como constantes extras.
    """

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

        self.DAG_DEFAULT_CONF = {
            "owner": "AMQ2",
            "depends_on_past": False,
            "schedule_interval": None,
            "schedule": None,
            "retries": 1,
            "retry_delay": datetime.timedelta(minutes=5),
            "dagrun_timeout": datetime.timedelta(minutes=15),
        }

        self.PARAM_GRID = {
            "learning_rate": [0.1],
            "max_depth": [3],
            "n_estimators": [100],
        }
        
        # Variables de entorno
        self.MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        self.MLFLOW_INPUT_PIPELINE_ALIAS = os.getenv("MLFLOW_INPUT_PIPELINE_ALIAS")
        self.MLFLOW_TARGET_PIPELINE_ALIAS = os.getenv("MLFLOW_TARGET_PIPELINE_ALIAS")

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
        self.DATASET_EXTENSION = ".csv"
        self.DATASET_NAME = "rain"
        self.BUCKET_DATA = "data"
        self.BOTO3_CLIENT = "s3"
        self.MLFLOW_EXPERIMENT_NAME = "Rain Dataset"
        self.X_TRAIN_NAME = "X_train.csv"
        self.X_TEST_NAME = "X_test.csv"
        self.Y_TRAIN_NAME = "y_train.csv"
        self.Y_TEST_NAME = "y_test.csv"
        self.PIPELINE_EXTENSION = ".pkl"
        self.INPUTS_PIPELINE_NAME = "inputs_pipeline"
        self.TARGET_PIPELINE_NAME = "target_pipeline"
        self.MODEL_PROD_NAME = "Rain_dataset_model_prod"
        self.MODEL_DEV_NAME = "Rain_dataset_model_dev"
        self.MODEL_PROD_DESC = "Modelo de predicción de lluvia"
        self.PROD_ALIAS = "prod_best"
        self.MODEL_ARTIFACT_PATH = "model_xgboost"
        self.CURRENT_MODEL = "XGBoost"

        # Constantes derivadas
        self.DATASET_NAME_W_EXTENSION = self.DATASET_NAME + self.DATASET_EXTENSION
        self.S3_RAW_DATA_PATH = self.S3_RAW_DATA_FOLDER + self.DATASET_NAME_W_EXTENSION
        self.S3_DF_PATH = (
            self.S3_PREPROCESED_DATA_FOLDER + self.DATASET_NAME_W_EXTENSION
        )
        self.MLFLOW_INPUT_PIPELINE_MODEL_REGISTRED_NAME = (
            "rain_dataset_etl_" + self.INPUTS_PIPELINE_NAME
        )
        self.MLFLOW_TARGET_PIPELINE_MODEL_REGISTRED_NAME = (
            "rain_dataset_etl_" + self.TARGET_PIPELINE_NAME
        )
        self.S3_GDF_LOCATIONS_PATH = self.INFO_DATA_FOLDER + "gdf_locations.json"
        self.S3_COLUMNS_PATH = self.INFO_DATA_FOLDER + "columnsTypes.json"
        self.S3_INPUT_PIPELINE_PATH = (
            self.PIPES_DATA_FOLDER + self.INPUTS_PIPELINE_NAME + self.PIPELINE_EXTENSION
        )
        self.S3_TARGET_PIPELINE_PATH = (
            self.PIPES_DATA_FOLDER + self.TARGET_PIPELINE_NAME + self.PIPELINE_EXTENSION
        )

    def get(self, key):
        """Método para obtener la configuración mediante una clave."""
        return getattr(self, key, None)
