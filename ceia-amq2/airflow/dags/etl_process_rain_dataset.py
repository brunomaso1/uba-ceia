import datetime
import pickle

from airflow.decorators import dag, task
from airflow import DAG
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import logging
import awswrangler as wr
from airflow.models import Variable
import boto3
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler
import geopandas as gpd

logger = logging.getLogger(__name__)
load_dotenv()

# TODO: Arreglar types hints

DATASET_NAME = "rain.csv"
COLUMNS_TYPE_FILE_NAME = "columnsTypes.json"
TARGET_PIPELINE_NAME = "target_pipeline.pkl"
INPUTS_PIPELINE_NAME = "inputs_pipeline.pkl"
GDF_LOCATIONS_NAME = "gdf_locations.json"
BUCKET_DATA = "data"
BOTO3_CLIENT = "s3"
X_TRAIN_NAME = "X_train.csv"
X_TEST_NAME = "X_test.csv"
Y_TRAIN_NAME = "y_train.csv"
Y_TEST_NAME = "y_test.csv"

S3_PREPROCESED_DATA_FOLDER = Variable.get("S3_PREPROCESED_DATA_FOLDER")
S3_RAW_DATA_FOLDER = Variable.get("S3_RAW_DATA_FOLDER")
S3_INFO_DATA_FOLDER = Variable.get("S3_INFO_DATA_FOLDER")
S3_PIPES_DATA_FOLDER = Variable.get("S3_PIPES_DATA_FOLDER")
S3_FINAL_DATA_FOLDER = Variable.get("S3_FINAL_DATA_FOLDER")

INFO_DATA_FOLDER = Variable.get("INFO_DATA_FOLDER")
PIPES_DATA_FOLDER = Variable.get("PIPES_DATA_FOLDER")
TEST_SIZE = Variable.get("TEST_SIZE")


markdown_text = """
### ETL Process for Rain Dataset

TODO: Escribir un resumen
"""

default_args = {
    "owner": "AMQ2",
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}


def map_bool(x):
    if isinstance(x, pd.DataFrame):
        return x.applymap(lambda y: {"Yes": 1, "No": 0}.get(y, y))
    else:
        return x.map({"Yes": 1, "No": 0})


def to_category(x):
    return x.astype("category")


def to_datetime(x):
    return x.astype("datetime64[ns]")


def save_to_csv(df, path):
    wr.s3.to_csv(df=df, path=path, index=False)


def encode_location(df: pd.DataFrame, gdf_locations) -> pd.DataFrame:
    return pd.merge(df, gdf_locations.drop(columns="geometry"), on="Location")

def eliminar_columnas(df, columnas_a_eliminar):
    return df.drop(columns=columnas_a_eliminar)


class ClapOutliersTransformerIRQ(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.IRQ_saved = {}
        self.columns = columns
        self.fitted = False

    def fit(self, X, y=None):
        for col in self.columns:
            # Rango itercuartílico
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IRQ = Q3 - Q1
            irq_lower_bound = Q1 - 3 * IRQ
            irq_upper_bound = Q3 + 3 * IRQ

            # Ajusto los valores al mínimo o máximo según corresponda.
            # Esto es para no pasarse de los valores mínimos o máximos con el IRQ.
            min_value = X[col].min()
            max_value = X[col].max()
            irq_lower_bound = max(irq_lower_bound, min_value)
            irq_upper_bound = min(irq_upper_bound, max_value)

            self.IRQ_saved[col + "irq_lower_bound"] = irq_lower_bound
            self.IRQ_saved[col + "irq_upper_bound"] = irq_upper_bound

        self.fitted = True

        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Fit the transformer first using fit().")

        X_transf = X.copy()

        for col in self.columns:
            irq_lower_bound = self.IRQ_saved[col + "irq_lower_bound"]
            irq_upper_bound = self.IRQ_saved[col + "irq_upper_bound"]
            X_transf[col] = X_transf[col].clip(
                upper=irq_upper_bound, lower=irq_lower_bound
            )

        return X_transf


def encode_cyclical_date(df, date_column="Date"):
    """
    Encodes a date column into cyclical features using sine and cosine transformations.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the date column.
    date_column (str): The name of the date column. Default is 'Date'.

    Returns:
    pandas.DataFrame: The dataframe with new 'DayCos' and 'DaySin' columns added,
                      and intermediate columns removed.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()

    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Calculate day of year
    df["DayOfYear"] = df[date_column].dt.dayofyear

    # Determine the number of days in the year for each date (taking leap years into account)
    df["DaysInYear"] = df[date_column].dt.is_leap_year.apply(
        lambda leap: 366 if leap else 365
    )

    # Convert day of the year to angle in radians, dividing by DaysInYear + 1
    df["Angle"] = 2 * np.pi * (df["DayOfYear"] - 1) / df["DaysInYear"]

    # Calculate sine and cosine features
    df["DayCos"] = np.cos(df["Angle"])
    df["DaySin"] = np.sin(df["Angle"])

    # Remove intermediate columns
    df = df.drop(columns=["DayOfYear", "DaysInYear", "Angle"])

    return df


def fix_location(df):
    mapping_dict = {"Dartmoor": "DartmoorVillage", "Richmond": "RichmondSydney"}
    df_out = df.copy()
    df_out["Location"] = df_out["Location"].map(mapping_dict).fillna(df["Location"])
    return df_out


def encode_wind_dir(df):
    dirs = [
        "E",
        "ENE",
        "NE",
        "NNE",
        "N",
        "NNW",
        "NW",
        "WNW",
        "W",
        "WSW",
        "SW",
        "SSW",
        "S",
        "SSE",
        "SE",
        "ESE",
    ]
    angles = np.radians(np.arange(0, 360, 22.5))
    mapping_dict = {d: a for (d, a) in zip(dirs, angles)}

    wind_dir_columns = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    for column in wind_dir_columns:
        df[f"{column}Angle"] = df[column].map(mapping_dict)

        df[f"{column}Cos"] = np.cos(df[f"{column}Angle"].astype(float))
        df[f"{column}Sin"] = np.sin(df[f"{column}Angle"].astype(float))

        df = df.drop(columns=f"{column}Angle")

    return df


with DAG(
    dag_id="process_etl_rain_dataset",
    # TODO: Escribir resumen
    description="TODO: Escribir resumen",
    doc_md=markdown_text,
    tags=["ETL", "Rain datset", "Dataset"],
    default_args=default_args,
    catchup=False,
) as dag:

    @task.virtualenv(requirements=["kagglehub"], system_site_packages=True)
    def download_raw_data_from_internet():
        import os
        import logging
        import kagglehub
        from airflow.models import Variable

        logger = logging.getLogger("airflow.task")

        kagglehub_repo_location = Variable.get("KAGGLEHUB_REPO_LOCATION")
        kagglehub_data_name = Variable.get("KAGGLEHUB_DATA_NAME")

        path = kagglehub.dataset_download(
            kagglehub_repo_location, path=kagglehub_data_name, force_download=True
        )

        return path

    @task.virtualenv(
        requirements=["awswrangler", "boto3", "osmnx", "geopandas"],
        system_site_packages=True,
    )
    def search_upload_locations(
        s3_raw_data_path, s3_gdf_locations_path, boto3_client, bucket_data
    ):
        import awswrangler as wr
        import boto3
        import re
        import osmnx as ox
        from shapely.geometry import Point
        import geopandas as gpd
        import pandas as pd
        import logging
        import json

        logger = logging.getLogger(__name__)

        client = boto3.client(boto3_client)
        df = wr.s3.read_csv(s3_raw_data_path)

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
            Bucket=bucket_data,
            Key=s3_gdf_locations_path,
            Body=json.dumps(gdf_locations_json),
        )

        return s3_gdf_locations_path

    @task
    def process_column_types():
        columns_types = {
            "cat_columns": ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"],
            "bool_columns": ["RainToday"],
            "date_columns": ["Date"],
            "cont_columns": [
                "MinTemp",
                "MaxTemp",
                "Rainfall",
                "Evaporation",
                "Sunshine",
                "WindGustSpeed",
                "WindSpeed9am",
                "WindSpeed3pm",
                "Humidity9am",
                "Humidity3pm",
                "Pressure9am",
                "Pressure3pm",
                "Cloud9am",
                "Cloud3pm",
                "Temp9am",
                "Temp3pm",
            ],
            "target_columns": ["RainTomorrow"],
        }

        s3_columns_path = INFO_DATA_FOLDER + COLUMNS_TYPE_FILE_NAME

        client = boto3.client(BOTO3_CLIENT)
        client.put_object(
            Bucket=BUCKET_DATA, Key=s3_columns_path, Body=json.dumps(columns_types)
        )

        return s3_columns_path

    @task
    def upload_raw_data_to_S3(local_path):
        # TODO: Add a bash operator for un-compressing the file.
        logger = logging.getLogger("airflow.task")

        df = pd.read_csv(local_path, compression="zip")

        s3_raw_data_path = S3_RAW_DATA_FOLDER + DATASET_NAME
        wr.s3.to_csv(df, path=s3_raw_data_path, index=False)

        return s3_raw_data_path

    @task
    def process_target_drop_na(s3_raw_data_path):
        s3_raw_data_path = S3_RAW_DATA_FOLDER + DATASET_NAME

        df = wr.s3.read_csv(s3_raw_data_path)
        df.dropna(subset=["RainTomorrow"], inplace=True, ignore_index=True)

        s3_df_path = S3_PREPROCESED_DATA_FOLDER + DATASET_NAME

        wr.s3.to_csv(df, path=s3_df_path, index=False)

        return s3_df_path

    @task
    def create_target_pipe():
        target_pipeline = Pipeline(steps=[])
        target_pipeline.steps.append(("mapping", FunctionTransformer(map_bool)))

        s3_target_pipeline_path = PIPES_DATA_FOLDER + TARGET_PIPELINE_NAME
        client = boto3.client(BOTO3_CLIENT)
        client.put_object(
            Bucket=BUCKET_DATA,
            Key=s3_target_pipeline_path,
            Body=pickle.dumps(target_pipeline),
        )

        return s3_target_pipeline_path

    @task
    def create_inputs_pipe(s3_columns_path, s3_gdf_locations_path):
        inputs_pipeline = Pipeline(steps=[])

        client = boto3.client(BOTO3_CLIENT)
        obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_columns_path)
        columns_types = json.load(obj["Body"])
        cat_columns = columns_types["cat_columns"]
        bool_columns = columns_types["bool_columns"]
        date_columns = columns_types["date_columns"]
        cont_columns = columns_types["cont_columns"]
        target_columns = columns_types["target_columns"]

        # Limpieza de datos
        # Transformar tipos de datos
        col_types_transf = ColumnTransformer(
            [
                ("categories", FunctionTransformer(to_category), cat_columns),
                ("date", FunctionTransformer(to_datetime), date_columns),
                ("bool", FunctionTransformer(map_bool), bool_columns),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        inputs_pipeline.steps.append(("feature_transf", col_types_transf))

        # Eliminar outliers
        columns = ["Rainfall", "Evaporation", "WindGustSpeed", "WindSpeed9am"]
        clap_outliers_irq_transf = ClapOutliersTransformerIRQ(columns=columns)

        inputs_pipeline.steps.append(
            ("clap_outliers_irq_transf", clap_outliers_irq_transf)
        )

        # Valores faltantes
        cat_imputer = (
            "cat_missing_values_imputer",
            SimpleImputer(strategy="most_frequent"),
        )
        cont_imputer = ("cont_missing_values_imptuer", SimpleImputer(strategy="mean"))
        missing_values_transf = ColumnTransformer(
            [
                ("cat_imputer", Pipeline([cat_imputer]), cat_columns + bool_columns),
                ("cont_imputer", Pipeline([cont_imputer]), cont_columns),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        inputs_pipeline.steps.append(("missing_values_transf", missing_values_transf))

        # Codificar variables categóricas
        # Date
        cyclical_date_transformer = FunctionTransformer(
            func=encode_cyclical_date, kw_args={"date_column": "Date"}, validate=False
        )
        inputs_pipeline.steps.append(
            ("cyclical_date_transformer", cyclical_date_transformer)
        )

        # Location
        obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_gdf_locations_path)
        gdf_locations = gpd.read_file(obj["Body"])
        encode_location_transformer = FunctionTransformer(
            func=encode_location,
            kw_args={"gdf_locations": gdf_locations},
            validate=False,
        )

        location_pieline = Pipeline(
            [
                ("fix_location", FunctionTransformer(fix_location)),
                ("encode_location_transformer", encode_location_transformer),
            ]
        )
        inputs_pipeline.steps.append(("location_pieline", location_pieline))

        # WindDir
        encode_wind_dir_transformer = FunctionTransformer(
            encode_wind_dir, validate=False
        )
        inputs_pipeline.steps.append(
            ("encode_wind_dir_transformer", encode_wind_dir_transformer)
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
        eliminar_columnas_transformer = FunctionTransformer(
            eliminar_columnas, kw_args={"columnas_a_eliminar": columnas_codificadas}
        )
        inputs_pipeline.steps.append(('eliminar_columnas_transformer', eliminar_columnas_transformer))

        # Scaler
        inputs_pipeline.steps.append(('StandardScaler', StandardScaler(with_mean=True, with_std=True).set_output(transform='pandas')))

        # Subimos el pipeline
        s3_input_pipeline_path = PIPES_DATA_FOLDER + INPUTS_PIPELINE_NAME

        client.put_object(
            Bucket=BUCKET_DATA,
            Key=s3_input_pipeline_path,
            Body=pickle.dumps(inputs_pipeline),
        )

        return s3_input_pipeline_path

    @task
    def split_dataset(s3_df_path):
        # TODO: Register to MLFLow?
        df = wr.s3.read_csv(s3_df_path)
        X = df.drop(columns="RainTomorrow")
        y = df["RainTomorrow"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        train_test_split_preprocesed_path = {
            "X_train": S3_PREPROCESED_DATA_FOLDER + X_TRAIN_NAME,
            "X_test": S3_PREPROCESED_DATA_FOLDER + X_TEST_NAME,
            "y_train": S3_PREPROCESED_DATA_FOLDER + Y_TRAIN_NAME,
            "y_test": S3_PREPROCESED_DATA_FOLDER + Y_TEST_NAME,
        }

        save_to_csv(X_train, train_test_split_preprocesed_path["X_train"])
        save_to_csv(X_test, train_test_split_preprocesed_path["X_test"])
        save_to_csv(y_train, train_test_split_preprocesed_path["y_train"])
        save_to_csv(y_test, train_test_split_preprocesed_path["y_test"])

        return train_test_split_preprocesed_path

    @task
    def fit_transform_pipes(
        train_test_split_preprocesed_path,
        s3_input_pipeline_path,
        s3_target_pipeline_path,
    ):
        #TODO: Register to MLFlow?
        X_train = wr.s3.read_csv(train_test_split_preprocesed_path["X_train"])
        X_test = wr.s3.read_csv(train_test_split_preprocesed_path["X_test"])
        y_train = wr.s3.read_csv(train_test_split_preprocesed_path["y_train"])
        y_test = wr.s3.read_csv(train_test_split_preprocesed_path["y_test"])

        client = boto3.client(BOTO3_CLIENT)
        obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_input_pipeline_path)
        inputs_pipeline = pickle.load(obj["Body"])
        obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_target_pipeline_path)
        target_pipeline = pickle.load(obj["Body"])

        X_train = inputs_pipeline.fit_transform(X_train)
        X_test = inputs_pipeline.transform(X_test)
        y_train = target_pipeline.fit_transform(y_train)
        y_test = target_pipeline.transform(y_test)

        train_test_split_final_path = {
            "X_train": S3_FINAL_DATA_FOLDER + X_TRAIN_NAME,
            "X_test": S3_FINAL_DATA_FOLDER + X_TEST_NAME,
            "y_train": S3_FINAL_DATA_FOLDER + Y_TRAIN_NAME,
            "y_test": S3_FINAL_DATA_FOLDER + Y_TEST_NAME,
        }

        save_to_csv(X_train, train_test_split_final_path["X_train"])
        save_to_csv(X_test, train_test_split_final_path["X_test"])
        save_to_csv(y_train, train_test_split_final_path["y_train"])
        save_to_csv(y_test, train_test_split_final_path["y_test"])

    local_path = download_raw_data_from_internet()
    s3_raw_data_path = upload_raw_data_to_S3(local_path)
    s3_df_path = process_target_drop_na(s3_raw_data_path)
    s3_gdf_locations_path = search_upload_locations(
        s3_raw_data_path,
        INFO_DATA_FOLDER + GDF_LOCATIONS_NAME,
        BOTO3_CLIENT,
        BUCKET_DATA,
    )

    s3_columns_path = process_column_types()
    s3_input_pipeline_path = create_inputs_pipe(s3_columns_path, s3_gdf_locations_path)

    s3_target_pipeline_path = create_target_pipe()

    train_test_split_paths = split_dataset(s3_df_path)
    fit_transform_pipes(
        train_test_split_paths, s3_input_pipeline_path, s3_target_pipeline_path
    )
