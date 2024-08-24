import datetime
from airflow import DAG

from utils.rain_dataset.rain_dataset_tasks.tasks import RainTasks
from utils.rain_dataset.ran_dataset_dags_docs.rain_dataset_doc import (
    DESCRIPTION,
    FULL_DESCRIPTION_MD,
)

default_args = {
    "owner": "AMQ2",
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}

with DAG(
    dag_id="process_etl_rain_dataset",
    description=DESCRIPTION,
    doc_md=FULL_DESCRIPTION_MD,
    tags=["ETL", "Rain datset", "Dataset"],
    default_args=default_args,
    catchup=False,
) as dag:
    rain_tasks = RainTasks()

    local_path = rain_tasks.download_raw_data_from_internet()
    s3_raw_data_path = rain_tasks.upload_raw_data_to_S3(local_path)
    s3_df_path = rain_tasks.process_target_drop_na(s3_raw_data_path)
    s3_gdf_locations_path = rain_tasks.search_upload_locations(s3_raw_data_path)

    s3_columns_path = rain_tasks.process_column_types()
    s3_input_pipeline_path = rain_tasks.create_inputs_pipe(
        s3_columns_path, s3_gdf_locations_path
    )

    s3_target_pipeline_path = rain_tasks.create_target_pipe()

    train_test_split_paths = rain_tasks.split_dataset(s3_df_path)
    rain_tasks.fit_transform_pipes(
        train_test_split_paths, s3_input_pipeline_path, s3_target_pipeline_path
    )
