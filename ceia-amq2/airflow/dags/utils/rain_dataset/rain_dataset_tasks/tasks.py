import mlflow.data.pandas_dataset
from utils.rain_dataset.rain_dataset_tasks.tasks_utils import (
    aux_functions,
    custom_transformers,
    types
)

import re
from airflow.decorators import task
import pickle
from dotenv import load_dotenv
import pandas as pd
import logging
import awswrangler as wr
from airflow.models import Variable
import boto3
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import mlflow
import logging
import os
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.sources import LocalArtifactDatasetSource

logger = logging.getLogger(__name__)

load_dotenv()

DATASET_NAME = "rain"
DATASET_NAME_W_EXTENSION = DATASET_NAME + ".csv"
COLUMNS_TYPE_FILE_NAME = "columnsTypes.json"
TARGET_PIPELINE_NAME = "target_pipeline.pkl"
INPUTS_PIPELINE_NAME = "inputs_pipeline.pkl"
BUCKET_DATA = "data"
BOTO3_CLIENT = "s3"
X_TRAIN_NAME = "X_train.csv"
X_TEST_NAME = "X_test.csv"
Y_TRAIN_NAME = "y_train.csv"
Y_TEST_NAME = "y_test.csv"
GDF_LOCATIONS_NAME = "gdf_locations.json"
MLFLOW_EXPERIMENT_NAME = "Rain Dataset"
TARGET = "RainTomorrow"

MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
RAW_DATA_FOLDER = Variable.get("RAW_DATA_FOLDER")
DATA_FOLDER = Variable.get("DATA_FOLDER")

S3_PREPROCESED_DATA_FOLDER = Variable.get("S3_PREPROCESED_DATA_FOLDER")
S3_RAW_DATA_FOLDER = Variable.get("S3_RAW_DATA_FOLDER")
S3_INFO_DATA_FOLDER = Variable.get("S3_INFO_DATA_FOLDER")
S3_PIPES_DATA_FOLDER = Variable.get("S3_PIPES_DATA_FOLDER")
S3_FINAL_DATA_FOLDER = Variable.get("S3_FINAL_DATA_FOLDER")

INFO_DATA_FOLDER = Variable.get("INFO_DATA_FOLDER")
PIPES_DATA_FOLDER = Variable.get("PIPES_DATA_FOLDER")
TEST_SIZE = Variable.get("TEST_SIZE")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")

S3_RAW_DATA_PATH = S3_RAW_DATA_FOLDER + DATASET_NAME_W_EXTENSION
S3_DF_PATH = S3_PREPROCESED_DATA_FOLDER + DATASET_NAME_W_EXTENSION
S3_GDF_LOCATIONS_PATH=INFO_DATA_FOLDER + GDF_LOCATIONS_NAME
S3_COLUMNS_PATH = INFO_DATA_FOLDER + COLUMNS_TYPE_FILE_NAME
S3_INPUT_PIPELINE_PATH = PIPES_DATA_FOLDER + INPUTS_PIPELINE_NAME
S3_TARGET_PIPELINE_PATH = PIPES_DATA_FOLDER + TARGET_PIPELINE_NAME

class RainTasks:
    @task.virtualenv(requirements=["kagglehub"], system_site_packages=True)
    def download_raw_data_from_internet():
        import kagglehub
        from airflow.models import Variable

        kagglehub_repo_location = Variable.get("KAGGLEHUB_REPO_LOCATION")
        kagglehub_data_name = Variable.get("KAGGLEHUB_DATA_NAME")

        path = kagglehub.dataset_download(
            kagglehub_repo_location, path=kagglehub_data_name, force_download=True
        )

        return path

    @task
    def search_upload_locations(dummy):
        logger = logging.getLogger(__name__)

        client = boto3.client(BOTO3_CLIENT)
        df = wr.s3.read_csv(S3_RAW_DATA_PATH)

        country = "Australia"
        mapping_dict = {"Dartmoor": "DartmoorVillage", "Richmond": "RichmondSydney"}
        df["Location"] = df["Location"].map(mapping_dict).fillna(df["Location"])

        locations = df["Location"].unique()

        locations = [re.sub(r"([a-z])([A-Z])", r"\1 \2", l) for l in locations]

        locs = []
        lats = []
        lons = []
        logger.info(f"len(locations)={len(locations)}")
        for location in locations:
            try:
                logger.info(f"location={location}")
                lat, lon = ox.geocode(location + f", {country}")

                locs.append(location.replace(" ", ""))
                lats.append(lat)
                lons.append(lon)
            except Exception as e:
                print(f"Error retrieving coordinates for {location}: {e}")

        df_locations = pd.DataFrame({"Location": locs, "Lat": lats, "Lon": lons})
        logger.info("df_locations=")
        logger.info(df_locations.head())
        geometry = [
            Point(lon, lat)
            for lon, lat in zip(df_locations["Lon"], df_locations["Lat"])
        ]
        gdf_locations = gpd.GeoDataFrame(
            df_locations, geometry=geometry, crs="EPSG:4326"
        )

        gdf_locations_json = json.loads(gdf_locations.to_json())

        client.put_object(
            Bucket=BUCKET_DATA,
            Key=S3_GDF_LOCATIONS_PATH,
            Body=json.dumps(gdf_locations_json),
        )

        return 'dummy'

    @task
    def process_column_types():
        client = boto3.client(BOTO3_CLIENT)
        client.put_object(
            Bucket=BUCKET_DATA, Key=S3_COLUMNS_PATH, Body=json.dumps(types.COLUMNS_TYPES)
        )

        return 'dummy'

    @task
    def upload_raw_data_to_S3(local_path):
        df = pd.read_csv(local_path, compression="zip")

        wr.s3.to_csv(df, path=S3_RAW_DATA_PATH, index=False)

        return 'dummy'

    @task
    def process_target_drop_na(dummy):
        df = wr.s3.read_csv(S3_RAW_DATA_PATH)
        df.dropna(subset=["RainTomorrow"], inplace=True, ignore_index=True)

        wr.s3.to_csv(df, path=S3_DF_PATH, index=False)

        return 'dummy'

    @task
    def create_target_pipe():
        target_pipeline = Pipeline(steps=[])
        target_pipeline.steps.append(
            ("mapping", FunctionTransformer(aux_functions.map_bool))
        )

        
        client = boto3.client(BOTO3_CLIENT)
        client.put_object(
            Bucket=BUCKET_DATA,
            Key=S3_TARGET_PIPELINE_PATH,
            Body=pickle.dumps(target_pipeline),
        )

        return 'dummy'

    @task
    def create_inputs_pipe(*dummy):
        inputs_pipeline = Pipeline(steps=[])

        client = boto3.client(BOTO3_CLIENT)
        obj = client.get_object(Bucket=BUCKET_DATA, Key=S3_COLUMNS_PATH)
        columns_types = json.load(obj["Body"])
        cat_columns = columns_types["cat_columns"]
        bool_columns = columns_types["bool_columns"]
        date_columns = columns_types["date_columns"]
        cont_columns = columns_types["cont_columns"]

        # Limpieza de datos
        # Transformar tipos de datos
        inputs_pipeline.steps.append(
            (
                "feature_transf",
                custom_transformers.convertDataTypeTransformer(
                    cat_columns, date_columns, bool_columns
                ),
            )
        )

        # Eliminar outliers
        columns = ["Rainfall", "Evaporation", "WindGustSpeed", "WindSpeed9am"]
        inputs_pipeline.steps.append(
            (
                "clap_outliers_irq_transf",
                custom_transformers.ClapOutliersIRQTransformer(columns=columns),
            )
        )

        # Valores faltantes
        inputs_pipeline.steps.append(
            (
                "missing_values_transf",
                custom_transformers.missingValuesTransformer(
                    cat_columns, bool_columns, cont_columns
                ),
            )
        )

        # Codificar variables categóricas
        # Date
        inputs_pipeline.steps.append(
            ("cyclical_date_transformer", custom_transformers.cyclicalDateTransformer())
        )

        # Location
        obj = client.get_object(Bucket=BUCKET_DATA, Key=S3_GDF_LOCATIONS_PATH)
        gdf_locations = gpd.read_file(obj["Body"])
        location_pieline = Pipeline(
            [
                ("fix_location", custom_transformers.fixLocationsTransformer()),
                (
                    "encode_location_transformer",
                    custom_transformers.encodeLocationTransformer(gdf_locations),
                ),
            ]
        )
        inputs_pipeline.steps.append(("location_pieline", location_pieline))

        # WindDir
        inputs_pipeline.steps.append(
            (
                "encode_wind_dir_transformer",
                custom_transformers.encodeWindDirTransformer(),
            )
        )

        # Eliminar columnas
        columnas_codificadas = [
            "WindGustDir",
            "WindDir9am",
            "WindDir3pm",
            "Date",
            "Location",
            "id",
        ]
        inputs_pipeline.steps.append(
            (
                "eliminar_columnas_transformer",
                custom_transformers.removeColumnsTransformer(columnas_codificadas),
            )
        )

        # Scaler
        inputs_pipeline.steps.append(
            (
                "StandardScaler",
                StandardScaler(with_mean=True, with_std=True).set_output(
                    transform="pandas"
                ),
            )
        )

        # Subimos el pipeline
        client.put_object(
            Bucket=BUCKET_DATA,
            Key=S3_INPUT_PIPELINE_PATH,
            Body=pickle.dumps(inputs_pipeline),
        )

        return 'dummy'

    @task
    def split_dataset(dummy):
        df = wr.s3.read_csv(S3_DF_PATH)
        X = df.drop(columns="RainTomorrow")
        y = df["RainTomorrow"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        train_test_split_preprocesed_path = {
            "X_train": S3_PREPROCESED_DATA_FOLDER + X_TRAIN_NAME,
            "X_test": S3_PREPROCESED_DATA_FOLDER + X_TEST_NAME,
            "y_train": S3_PREPROCESED_DATA_FOLDER + Y_TRAIN_NAME,
            "y_test": S3_PREPROCESED_DATA_FOLDER + Y_TEST_NAME,
        }

        aux_functions.upload_split_to_s3(
            X_train, X_test, y_train, y_test, train_test_split_preprocesed_path
        )

        return train_test_split_preprocesed_path

    @task
    def fit_transform_pipes(        train_test_split_preprocesed_path, *dummy    ):
        X_train, X_test, y_train, y_test = aux_functions.download_split_from_s3(
            train_test_split_preprocesed_path
        )

        client = boto3.client(BOTO3_CLIENT)
        obj = client.get_object(Bucket=BUCKET_DATA, Key=S3_INPUT_PIPELINE_PATH)
        inputs_pipeline = pickle.load(obj["Body"])
        obj = client.get_object(Bucket=BUCKET_DATA, Key=S3_TARGET_PIPELINE_PATH)
        target_pipeline = pickle.load(obj["Body"])

        X_train = inputs_pipeline.fit_transform(X_train)
        X_test = inputs_pipeline.transform(X_test)
        y_train = target_pipeline.fit_transform(y_train)
        y_test = target_pipeline.transform(y_test)

        client.put_object(
            Bucket=BUCKET_DATA,
            Key=S3_INPUT_PIPELINE_PATH,
            Body=pickle.dumps(inputs_pipeline),
        )

        client.put_object(
            Bucket=BUCKET_DATA,
            Key=S3_TARGET_PIPELINE_PATH,
            Body=pickle.dumps(target_pipeline),
        )

        train_test_split_final_path = {
            "X_train": S3_FINAL_DATA_FOLDER + X_TRAIN_NAME,
            "X_test": S3_FINAL_DATA_FOLDER + X_TEST_NAME,
            "y_train": S3_FINAL_DATA_FOLDER + Y_TRAIN_NAME,
            "y_test": S3_FINAL_DATA_FOLDER + Y_TEST_NAME,
        }

        aux_functions.upload_split_to_s3(
            X_train, X_test, y_train, y_test, train_test_split_final_path
        )

        # No se puede devovler una tupa debido al siguiente issue: https://github.com/apache/airflow/discussions/31680
        # Por esto el boilerplate de código para serializarlo por JSON.
        final_paths = {
            "train_test_split_preprocesed_path": train_test_split_preprocesed_path,
            "train_test_split_final_path": train_test_split_final_path,
            "s3_input_pipeline_path": S3_INPUT_PIPELINE_PATH,
            "s3_target_pipeline_path": S3_TARGET_PIPELINE_PATH,
        }

        return final_paths

    @task
    def register_to_mlflow(final_paths):
        MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        RAW_DATA_FOLDER = Variable.get("RAW_DATA_FOLDER")
        logger.info(f"MLFLOW_S3_ENDPOINT_URL={MLFLOW_S3_ENDPOINT_URL}")
        logger.info(f"RAW_DATA_FOLDER={RAW_DATA_FOLDER}")

        s3_raw_data_path = S3_RAW_DATA_FOLDER + DATASET_NAME_W_EXTENSION
        df = wr.s3.read_csv(s3_raw_data_path)

        train_test_split_final_path = final_paths["train_test_split_final_path"]
        s3_input_pipeline_path = final_paths["s3_input_pipeline_path"]

        X_train, X_test, _, _ = aux_functions.download_split_from_s3(
            train_test_split_final_path
        )

        client = boto3.client(BOTO3_CLIENT)
        obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_input_pipeline_path)
        inputs_pipeline: Pipeline = pickle.load(obj["Body"])

        sc_X = inputs_pipeline["StandardScaler"]

        # source_dataset = MLFLOW_S3_ENDPOINT_URL + "/" + DATA_FOLDER + RAW_DATA_FOLDER + DATASET_NAME_W_EXTENSION
        # logger.info(f"source_dataset={source_dataset}")
        # dataset: PandasDataset = mlflow.data.from_pandas(df, source=LocalArtifactDatasetSource(source_dataset), name=DATASET_NAME, targets=TARGET)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run():
            mlflow.set_tag("Training info", "Training data for Rain DataSet")

            # mlflow.log_input(dataset, context="Dataset")

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", sc_X.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", sc_X.mean_)
            mlflow.log_param("Standard Scaler scale values", sc_X.scale_)
