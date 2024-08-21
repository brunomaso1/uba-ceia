import datetime

from airflow.decorators import dag, task
from dotenv import load_dotenv
import pandas as pd
import logging
import awswrangler as wr
from airflow.models import Variable

logger = logging.getLogger(__name__)
load_dotenv()

# TODO: Arreglar types hints

DATASET_NAME = 'raw_rain.csv'
COLUMNS_TYPE_FILE_NAME = 'columnsTypes.json'
S3_PREPROCESED_FOLDER = Variable.get("S3_PREPROCESED_DATA_FOLDER")
S3_RAW_DATA_FOLDER = Variable.get("S3_RAW_DATA_FOLDER")

logger.debug(f"S3_RAW_DATA_FOLDER={S3_RAW_DATA_FOLDER}")
logger.debug(f"S3_PREPROCESED_FOLDER={S3_PREPROCESED_FOLDER}")

markdown_text = """
### ETL Process for Rain Dataset

TODO: Escribir un resumen
"""

default_args = {
    'owner': "AMQ2",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="process_etl_rain_dataset",
    # TODO: Escribir resumen
    description="TODO: Escribir resumen",
    doc_md=markdown_text,
    tags=["ETL", "Rain datset", "Dataset"],
    default_args=default_args,
    catchup=False,
)
def process_etl_rain_dataset():
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

        logger.debug(f"KAGGLEHUB_CACHE={os.getenv('KAGGLEHUB_CACHE')}")

        kagglehub_repo_location = Variable.get("KAGGLEHUB_REPO_LOCATION")
        logger.debug(f"kagglehub_repo_location={kagglehub_repo_location}")
        kagglehub_data_name = Variable.get("KAGGLEHUB_DATA_NAME")
        logger.debug(f"kagglehub_repo_location={kagglehub_data_name}")

        path = kagglehub.dataset_download(
            kagglehub_repo_location, path=kagglehub_data_name, force_download=True)

        return path

    @task
    def upload_raw_data_to_S3(local_path):
        # TODO: Add a bash operator for un-compressing the file.
        logger = logging.getLogger("airflow.task")

        logger.debug(f"local_path={local_path}")

        df = pd.read_csv(local_path, compression='zip')

        wr.s3.to_csv(df, path=S3_RAW_DATA_FOLDER + DATASET_NAME, index=False)

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

        s3_columns_path = S3_PREPROCESED_FOLDER + COLUMNS_TYPE_FILE_NAME

        wr.s3.to_json(pd.DataFrame(columns_types), path=s3_columns_path,
                      index=False)
        
        return s3_columns_path

    @task
    def process_target(s3_columns_path):
        logger = logging.getLogger("airflow.task")

        logger.debug(f"s3_columns_path={s3_columns_path}")

        columns_types = wr.s3.read_json(s3_columns_path)
        df = wr.s3.read_csv(S3_RAW_DATA_FOLDER + DATASET_NAME)

        bool_columns = columns_types['bool_columns']
        target_columns = columns_types['target_columns']
        mapping_dict = {"Yes": 1, "No": 0}

        df[bool_columns + target_columns] = df[bool_columns + target_columns].apply(lambda x: mapping_dict.get(x, x))
        logger.info(df.head(5))

        s3_df_path = S3_PREPROCESED_FOLDER + DATASET_NAME

        wr.s3.to_csv(df, path=s3_df_path, index=False)

    local_path = download_raw_data_from_internet()
    upload_raw_data_to_S3(local_path)
    # s3_columns_path = process_column_types()

dag = process_etl_rain_dataset()
