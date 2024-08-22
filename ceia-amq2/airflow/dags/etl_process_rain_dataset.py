import datetime
import pickle

from airflow.decorators import dag, task
from airflow import DAG
from dotenv import load_dotenv
import pandas as pd
import logging
import awswrangler as wr
from airflow.models import Variable
import boto3
import json
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline

logger = logging.getLogger(__name__)
load_dotenv()

# TODO: Arreglar types hints

DATASET_NAME = 'rain.csv'
COLUMNS_TYPE_FILE_NAME = 'columnsTypes.json'
TARGET_PIPELINE_NAME = 'target_pipeline.pkl'
INPUTS_PIPELINE_NAME = 'inputs_pipeline.pkl'
BUCKET_DATA = 'data'
BOTO3_CLIENT = 's3'
X_TRAIN_NAME = 'X_train.csv'
X_TEST_NAME = 'X_test.csv'
Y_TRAIN_NAME = 'y_train.csv'
Y_TEST_NAME = 'y_test.csv'

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
    'owner': "AMQ2",
    'depends_on_past': False,
    'schedule_interval': None,
    'schedule': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


def map_bool(x):
    if isinstance(x, pd.DataFrame):
        return x.applymap(lambda y: {"Yes": 1, "No": 0}.get(y, y))
    else:
        return x.map({"Yes": 1, "No": 0})


def to_category(x):
    return x.astype('category')


def to_datetime(x):
    return x.astype('datetime64[ns]')


def save_to_csv(df, path):
    wr.s3.to_csv(df=df, path=path, index=False)


with DAG(
    dag_id="process_etl_rain_dataset",
    # TODO: Escribir resumen
    description="TODO: Escribir resumen",
    doc_md=markdown_text,
    tags=["ETL", "Rain datset", "Dataset"],
    default_args=default_args,
    catchup=False,
) as dag:
    @task.virtualenv(
        requirements=["kagglehub"],
        system_site_packages=True
    )
    def download_raw_data_from_internet():
        import os
        import logging
        import kagglehub
        from airflow.models import Variable

        logger = logging.getLogger("airflow.task")

        kagglehub_repo_location = Variable.get("KAGGLEHUB_REPO_LOCATION")
        kagglehub_data_name = Variable.get("KAGGLEHUB_DATA_NAME")

        path = kagglehub.dataset_download(
            kagglehub_repo_location, path=kagglehub_data_name, force_download=True)

        return path

    @task
    def process_column_types():
        columns_types = {
            'cat_columns': ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],
            'bool_columns': ['RainToday'],
            'date_columns': ['Date'],
            'cont_columns': ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                             'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                             'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                             'Cloud3pm', 'Temp9am', 'Temp3pm'],
            'target_columns': ['RainTomorrow']
        }

        s3_columns_path = INFO_DATA_FOLDER + COLUMNS_TYPE_FILE_NAME

        client = boto3.client(BOTO3_CLIENT)
        client.put_object(Bucket=BUCKET_DATA, Key=s3_columns_path,
                          Body=json.dumps(columns_types))

        return s3_columns_path

    @task
    def upload_raw_data_to_S3(local_path):
        # TODO: Add a bash operator for un-compressing the file.
        logger = logging.getLogger("airflow.task")

        df = pd.read_csv(local_path, compression='zip')

        s3_raw_data_path = S3_RAW_DATA_FOLDER + DATASET_NAME
        wr.s3.to_csv(df, path=s3_raw_data_path, index=False)

        return s3_raw_data_path

    @task
    def process_target_drop_na(s3_raw_data_path):
        s3_raw_data_path = S3_RAW_DATA_FOLDER + DATASET_NAME

        df = wr.s3.read_csv(s3_raw_data_path)
        df.dropna(subset=['RainTomorrow'], inplace=True, ignore_index=True)

        s3_df_path = S3_PREPROCESED_DATA_FOLDER + DATASET_NAME

        wr.s3.to_csv(df, path=s3_df_path, index=False)

        return s3_df_path

    @task
    def create_target_pipe():
        target_pipeline = Pipeline(steps=[])
        target_pipeline.steps.append(
            ('mapping', FunctionTransformer(map_bool)))

        s3_target_pipeline_path = PIPES_DATA_FOLDER + TARGET_PIPELINE_NAME
        client = boto3.client(BOTO3_CLIENT)
        client.put_object(Bucket=BUCKET_DATA, Key=s3_target_pipeline_path,
                          Body=pickle.dumps(target_pipeline))

        return s3_target_pipeline_path

    @task
    def create_inputs_pipe(s3_columns_path):
        inputs_pipeline = Pipeline(steps=[])

        client = boto3.client(BOTO3_CLIENT)
        obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_columns_path)
        columns_types = json.load(obj['Body'])
        cat_columns = columns_types['cat_columns']
        bool_columns = columns_types['bool_columns']
        date_columns = columns_types['date_columns']
        cont_columns = columns_types['cont_columns']
        target_columns = columns_types['target_columns']

        col_types_transf = ColumnTransformer(
            [('categories', FunctionTransformer(to_category), cat_columns),
             ('date', FunctionTransformer(to_datetime), date_columns),
             ('bool', FunctionTransformer(map_bool), bool_columns)],
            remainder='passthrough',
            verbose_feature_names_out=False).set_output(transform='pandas')

        inputs_pipeline.steps.append(('feature_transf', col_types_transf))

        # TODO: Agregar todas las tranformaciones faltantes.

        s3_input_pipeline_path = PIPES_DATA_FOLDER + INPUTS_PIPELINE_NAME

        client.put_object(Bucket=BUCKET_DATA, Key=s3_input_pipeline_path,
                          Body=pickle.dumps(inputs_pipeline))

        return s3_input_pipeline_path

    @task
    def split_dataset(s3_df_path):
        # TODO: Register to MLFLow?
        df = wr.s3.read_csv(s3_df_path)
        X = df.drop(columns="RainTomorrow")
        y = df["RainTomorrow"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        train_test_split_preprocesed_path = {
            'X_train': S3_PREPROCESED_DATA_FOLDER + X_TRAIN_NAME,
            'X_test': S3_PREPROCESED_DATA_FOLDER + X_TEST_NAME,
            'y_train': S3_PREPROCESED_DATA_FOLDER + Y_TRAIN_NAME,
            'y_test': S3_PREPROCESED_DATA_FOLDER + Y_TEST_NAME
        }

        save_to_csv(X_train, train_test_split_preprocesed_path['X_train'])
        save_to_csv(X_test, train_test_split_preprocesed_path['X_test'])
        save_to_csv(y_train, train_test_split_preprocesed_path['y_train'])
        save_to_csv(y_test, train_test_split_preprocesed_path['y_test'])

        return train_test_split_preprocesed_path

    @task
    def fit_transform_pipes(train_test_split_preprocesed_path, s3_input_pipeline_path, s3_target_pipeline_path):
        X_train = wr.s3.read_csv(train_test_split_preprocesed_path['X_train'])
        X_test = wr.s3.read_csv(train_test_split_preprocesed_path['X_test'])
        y_train = wr.s3.read_csv(train_test_split_preprocesed_path['y_train'])
        y_test = wr.s3.read_csv(train_test_split_preprocesed_path['y_test'])

        client = boto3.client(BOTO3_CLIENT)
        obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_input_pipeline_path)
        inputs_pipeline = pickle.load(obj['Body'])
        obj = client.get_object(
            Bucket=BUCKET_DATA, Key=s3_target_pipeline_path)
        target_pipeline = pickle.load(obj['Body'])

        X_train = inputs_pipeline.fit_transform(X_train)
        X_test = inputs_pipeline.transform(X_test)
        y_train = target_pipeline.fit_transform(y_train)
        y_test = target_pipeline.transform(y_test)

        train_test_split_final_path = {
            'X_train': S3_FINAL_DATA_FOLDER + X_TRAIN_NAME,
            'X_test': S3_FINAL_DATA_FOLDER + X_TEST_NAME,
            'y_train': S3_FINAL_DATA_FOLDER + Y_TRAIN_NAME,
            'y_test': S3_FINAL_DATA_FOLDER + Y_TEST_NAME
        }

        save_to_csv(X_train, train_test_split_final_path['X_train'])
        save_to_csv(X_test, train_test_split_final_path['X_test'])
        save_to_csv(y_train, train_test_split_final_path['y_train'])
        save_to_csv(y_test, train_test_split_final_path['y_test'])

    local_path = download_raw_data_from_internet()
    s3_raw_data_path = upload_raw_data_to_S3(local_path)
    s3_df_path = process_target_drop_na(s3_raw_data_path)

    s3_columns_path = process_column_types()
    s3_input_pipeline_path = create_inputs_pipe(s3_columns_path)

    s3_target_pipeline_path = create_target_pipe()

    train_test_split_paths = split_dataset(s3_df_path)
    fit_transform_pipes(train_test_split_paths, s3_input_pipeline_path,
                        s3_target_pipeline_path)
