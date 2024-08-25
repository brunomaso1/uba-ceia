import datetime
from airflow import DAG
import sklearn
from airflow.decorators import task

from utils.rain_dataset.rain_dataset_tasks.etl_tasks import RainTasks
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

@task
def test():
    print('The scikit-learn version is {}.'.format(sklearn.__version__))

with DAG(
    dag_id="test_etl",
    description=DESCRIPTION,
    doc_md=FULL_DESCRIPTION_MD,
    tags=["ETL", "Rain datset", "Dataset"],
    default_args=default_args,
    catchup=False,
) as dag:
    test()