"""
OPTIMIZE: Crea y optimiza el modelo.
"""
import datetime
import mlflow
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from airflow.decorators import dag, task
from rain_dataset_utils import (
    aux_functions,
    config_loader
)
from rain_dataset_utils.rain_dataset_doc import (
    DESCRIPTION_OPTIMIZE,
    FULL_DESCRIPTION_MD_OPTIMIZE,
)
import logging

logger = logging.getLogger(__name__)
config = config_loader.RainDatasetConfigs()

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

@dag(
    dag_id="train_optimize_rain_dataset",
    description=DESCRIPTION_OPTIMIZE,
    doc_md=FULL_DESCRIPTION_MD_OPTIMIZE,
    tags=["Optimization", config.MLFLOW_EXPERIMENT_NAME],
    default_args=config.DAG_DEFAULT_CONF,
    catchup=False,
)
def optimization_dag():

    @task(multiple_outputs=True)
    def load_train_test_dataset():
        X_train, X_test, y_train, y_test = aux_functions.download_split_from_s3_final()

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    @task
    def experiment_creation():
        print("Creating experiment")
        # Se crea el experimento en MLflow, verificando si ya existe.
        if experiment := mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME):
            logger.info("Found experiment")
            return experiment.experiment_id
        else:
            logger.info("Creating new experiment")
            return mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME)

    @task
    def find_best_model(X_train, y_train, X_test, y_test, experiment_id):

        from mlflow import MlflowClient
        run_name_parent = "best_hyperparam_" + datetime.datetime.today().strftime(
            '%Y%m%d_%H%M%S"'
        )

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name_parent):
            # Inicializar el modelo XGBoost
            xgb_model = xgb.XGBClassifier(objective="binary:logistic")

            # Configurar la búsqueda grid con validación cruzada
            grid_search = GridSearchCV(
                estimator=xgb_model, param_grid=config.PARAM_GRID, cv=5, scoring="accuracy"
            )
            grid_search.fit(X_train.astype(float), y_train)

            # Obtener los mejores hiperparámetros
            best_params = grid_search.best_params_
            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy", grid_search.best_score_)

            mlflow.set_tags(
                tags={
                    "project": config.MLFLOW_EXPERIMENT_NAME,
                    "model": config.CURRENT_MODEL,
                    "optimizer": "GridSearchCV",
                }
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
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

            signature = mlflow.models.signature.infer_signature(
                X_train, best_model.predict(X_train)
            )

            # Se guarda el modelo en MLflow
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=config.MODEL_ARTIFACT_PATH,
                signature=signature,
                serialization_format="cloudpickle",
                registered_model_name=config.MODEL_DEV_NAME,
                metadata={"task": "Classification", "dataset": config.MLFLOW_EXPERIMENT_NAME},
            )

            # Se obtiene la ubicación del modelo guardado en MLflow
            model_uri = mlflow.get_artifact_uri(config.MODEL_ARTIFACT_PATH)

            return model_uri

    @task
    def test_model(model_uri, experiment_id):

        run_name_parent = "test_run_" + datetime.datetime.today().strftime(
            '%Y%m%d_%H%M%S"'
        )

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name_parent):
            # Se carga el modelo guardado en MLflow
            model = mlflow.sklearn.load_model(model_uri)

            # Se cargan los datos de prueba
            _, X_test, _, y_test = aux_functions.download_split_from_s3_final()

            # Se realiza la predicción con el modelo
            y_pred = model.predict(X_test)

            # Se calculan las métricas de evaluación
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)

            # Se registran las métricas en MLflow
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

    @task
    def register_model(model_uri):
        client = mlflow.MlflowClient()
        config.MODEL_PROD_DESC = "Modelo de predicción de lluvia"

        # Se carga el modelo guardado en MLflow
        model = mlflow.sklearn.load_model(model_uri)

        # Creamos el modelo en productivo
        try:
            client.create_registered_model(name=config.MODEL_PROD_NAME, description=config.MODEL_PROD_DESC)
        except:
            print("Model already exists")
            pass

        # Guardamos como tag los hiperparámetros del modelo
        tags = model.get_params()
        tags["model"] = type(model).__name__

        # Guardamos la versión del modelo
        result = client.create_model_version(
            name=config.MODEL_PROD_NAME, source=model_uri, run_id=model_uri.split("/")[-3], tags=tags
        )

        # Y creamos como la version con el alias de prod_best para poder levantarlo en nuestro
        # proceso de servicio del modelo on-line.
        client.set_registered_model_alias(config.MODEL_PROD_NAME, "prod_best", result.version)

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
