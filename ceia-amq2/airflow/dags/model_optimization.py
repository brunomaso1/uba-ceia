import datetime
import mlflow
import awswrangler as wr
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from airflow.decorators import dag, task

from utils.rain_dataset.rain_dataset_tasks.tasks_utils import (
    aux_functions,
    custom_transformers,
    types,
)

from utils.rain_dataset.rain_dataset_configs.config_loader import RainDatasetConfigs

config = RainDatasetConfigs()


mlflow.set_tracking_uri('http://mlflow:5000')

markdown_text = """
### Hyperparameter Optimization

This DAG performs hyperparameter optimization.
###TODO: Add more information

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

@dag(
    dag_id="train_optimize_rain_dataset",
    description="Perform hyperparameter optimization",
    doc_md=markdown_text,
    tags=["Optimization", "Rain dataset"],
    default_args=default_args,
    catchup=False,
)

def optimization_dag():

    @task(multiple_outputs=True)
    def load_train_test_dataset():
        X_train = wr.s3.read_csv("s3://data/final/X_train.csv")
        y_train = wr.s3.read_csv("s3://data/final/y_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/y_test.csv")

        return {"X_train":X_train, "y_train": y_train, 
                "X_test": X_test, "y_test": y_test}

    @task
    def experiment_creation():
        print("Creating experiment")
        # Se crea el experimento en MLflow, verificando si ya existe.
        if experiment := mlflow.get_experiment_by_name("Rain Dataset"):
            print("Found experiment")
            return experiment.experiment_id
        else:
            print("Creating new experiment")
            return mlflow.create_experiment("Rain Dataset")

    @task
    def find_best_model(X_train, y_train, X_test, y_test, experiment_id):

        from mlflow import MlflowClient
        client = MlflowClient()

        run_name_parent = "best_hyperparam_"  + datetime.datetime.today().strftime('%Y%m%d_%H%M%S"')

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name_parent):
            # Definir los hiperparámetros a ajustar
            # param_grid = {
            #     'learning_rate': [0.1, 0.01],
            #     'max_depth': [3, 6 ,9],
            #     'n_estimators': [100, 500, 1000]
            # }

            param_grid = {
                'learning_rate': [0.1],
                'max_depth': [3],
                'n_estimators': [100]
            }

            # Inicializar el modelo XGBoost
            xgb_model = xgb.XGBClassifier(objective='binary:logistic')

            # Configurar la búsqueda grid con validación cruzada
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train.astype(float), y_train)

            # Obtener los mejores hiperparámetros
            best_params = grid_search.best_params_
            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy", grid_search.best_score_)

            mlflow.set_tags(
                tags={
                    "project": "Rain dataset",
                    "model": "XGBoost",
                    "optimizer": "GridSearchCV"}
                )

            # Entrenar el modelo con los mejores hiperparámetros
            best_model = xgb.XGBClassifier(**best_params)
            best_model.fit(X_train.astype(float), y_train)

            # Se calculan las métricas de evaluación con el conjunto de prueba
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)

            # Se registran las métricas en MLflow
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            # Se registran la matriz de confusión en MLflow
            #TODO: Log confusion matrix

            # Se guarda el artefacto del modelo
            artifact_path = "model_xgboost"
            signature = mlflow.models.signature.infer_signature(X_train, best_model.predict(X_train))

            # Se guarda el modelo en MLflow
            mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path=artifact_path,
                signature=signature,
                serialization_format="cloudpickle",
                registered_model_name="Rain_dataset_model_dev",
                metadata={"task": "Classification", "dataset": "Rain dataset"}
                )
            
            # Registrar el pipeline en MLFlow
            inputs_pipeline, target_pipeline = aux_functions.load_pipelines_from_s3()
            mlflow.sklearn.log_model(inputs_pipeline, "inputs_pipeline")
            mlflow.sklearn.log_model(target_pipeline, "target_pipeline")
            
            # Se obtiene la ubicación del modelo guardado en MLflow
            model_uri = mlflow.get_artifact_uri(artifact_path)

            # TODO: Agregar un alias
            # client.set_registered_model_alias(name="Rain_dataset_model_dev", alias="dev_best", version=1)

            return model_uri

    @task
    def test_model(model_uri, experiment_id):

        run_name_parent = "test_run_"  + datetime.datetime.today().strftime('%Y%m%d_%H%M%S"')

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name_parent):
            # Se carga el modelo guardado en MLflow
            model = mlflow.sklearn.load_model(model_uri)

            # Se cargan los datos de prueba
            X_test = wr.s3.read_csv("s3://data/final/X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/y_test.csv")

            # Se realiza la predicción con el modelo
            y_pred = model.predict(X_test)

            # Se calculan las métricas de evaluación
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)

            # Se registran las métricas en MLflow
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            # Se registran la matriz de confusión en MLflow
            #TODO: Log confusion matrix

    @task
    def register_model(model_uri):

        from mlflow import MlflowClient
        
        client = mlflow.MlflowClient()
        name = "rain_dataset_model_prod"
        desc = "Modelo de predicción de lluvia"

        # Se carga el modelo guardado en MLflow
        model = mlflow.sklearn.load_model(model_uri)

        # Creamos el modelo en productivo
        client.create_registered_model(name=name, description=desc)

        # Guardamos como tag los hiperparámetros del modelo
        tags = model.get_params()
        tags["model"] = type(model).__name__
        
        # Guardamos la versión del modelo
        result = client.create_model_version(
            name=name,
            source=model_uri,
            run_id=model_uri.split("/")[-3],
            tags=tags
        )

        # Y creamos como la version con el alias de prod_best para poder levantarlo en nuestro
        # proceso de servicio del modelo on-line.
        client.set_registered_model_alias(name, "prod_best", result.version)

    # Cargar el conjunto de datos de entrenamiento
    load_train_test = load_train_test_dataset()
    X_train = load_train_test["X_train"]
    y_train = load_train_test["y_train"]
    X_test = load_train_test["X_test"]
    y_test = load_train_test["y_test"]

    # Crear el experimento en MLflow
    experiment_id = experiment_creation()

    # Encuentra los mejores hiperparámetros
    model_uri_path = find_best_model(X_train, y_train, X_test, y_test, experiment_id)
    
    test_model(model_uri_path, experiment_id)

    register_model(model_uri_path)

dag = optimization_dag()