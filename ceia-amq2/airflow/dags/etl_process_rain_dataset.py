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

    task_data01 = rain_tasks.download_raw_data_from_internet()
    task_data10 = rain_tasks.upload_raw_data_to_S3(task_data01)
    task_data21 = rain_tasks.process_target_drop_na(task_data10)
    task_data22 = rain_tasks.search_upload_locations(task_data10)

    task_data02 = rain_tasks.process_column_types()
    task_data30 = rain_tasks.create_inputs_pipe(task_data02, task_data22)

    task_data03 = rain_tasks.create_target_pipe()

    tast_data31 = rain_tasks.split_dataset(task_data21)
    task_data40 = rain_tasks.fit_transform_pipes(tast_data31, task_data30, task_data03)
    rain_tasks.register_to_mlflow(task_data40)
