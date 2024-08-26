"""
RETRAIN: Re-entrena el modelo
"""

from airflow.decorators import dag, task
from rain_dataset_utils import (
    aux_functions,
    config_loader
)
from rain_dataset_utils.rain_dataset_doc import (
    DESCRIPTION_RETRAIN,
    FULL_DESCRIPTION_MD_RETRAIN,
)

config = config_loader.RainDatasetConfigs()

@dag(
    dag_id="retrain_model_rain_dataset",
    description=DESCRIPTION_RETRAIN,
    doc_md=FULL_DESCRIPTION_MD_RETRAIN,
    tags=["Re-Train", config.MLFLOW_EXPERIMENT_NAME],
    default_args=config.DAG_DEFAULT_CONF,
    catchup=False,
)
def processing_dag():

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=["scikit-learn==1.3.2", "mlflow==2.10.2", "awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def train_the_challenger_model():
        import datetime
        import mlflow
        import awswrangler as wr

        from sklearn.base import clone
        from sklearn.metrics import f1_score
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

        def load_the_champion_model():
            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(config.MODEL_PROD_NAME, config.PROD_ALIAS)

            champion_version = mlflow.sklearn.load_model(model_data.source)

            return champion_version

        def mlflow_track_experiment(model, X):
            # Track the experiment
            experiment = mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

            mlflow.start_run(
                run_name="Challenger_run_"
                + datetime.datetime.today().strftime('%Y%m%d_%H%M%S"'),
                experiment_id=experiment.experiment_id,
                tags={"experiment": "challenger models", "dataset": config.MLFLOW_EXPERIMENT_NAME},
                log_system_metrics=True,
            )

            params = model.get_params()
            params["model"] = type(model).__name__

            mlflow.log_params(params)

            signature = mlflow.models.signature.infer_signature(X, model.predict(X))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=config.MODEL_ARTIFACT_PATH,
                signature=signature,
                serialization_format="cloudpickle",
                registered_model_name=config.MODEL_DEV_NAME,
                metadata={"model_data_version": 1},
            )

            # TODO: Eliminar esto?
            # Registrar el pipeline en MLFlow
            inputs_pipeline, target_pipeline = aux_functions.load_pipelines_from_s3()
            mlflow.sklearn.log_model(inputs_pipeline, config.INPUTS_PIPELINE_NAME)
            mlflow.sklearn.log_model(target_pipeline, config.TARGET_PIPELINE_NAME)

            # Obtain the model URI
            return mlflow.get_artifact_uri(config.MODEL_ARTIFACT_PATH)

        def register_challenger(model, f1_score, model_uri):

            client = mlflow.MlflowClient()

            # Save the model params as tags
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["f1-score"] = f1_score

            # Save the version of the model
            result = client.create_model_version(
                name=config.MODEL_PROD_NAME, source=model_uri, run_id=model_uri.split("/")[-3], tags=tags
            )

            # Save the alias as challenger
            client.set_registered_model_alias(config.MODEL_PROD_NAME, "challenger", result.version)

        # Load the champion model
        champion_model = load_the_champion_model()

        # Clone the model
        challenger_model = clone(champion_model)

        # Load the dataset
        X_train, y_train, X_test, y_test = aux_functions.download_split_from_s3_final()

        # Fit the training model
        challenger_model.fit(X_train, y_train.to_numpy().ravel())

        # Obtain the metric of the model
        y_pred = challenger_model.predict(X_test)
        f1_score = f1_score(y_test.to_numpy().ravel(), y_pred)

        # Track the experiment
        artifact_uri = mlflow_track_experiment(challenger_model, X_train)

        # Record the model
        register_challenger(challenger_model, f1_score, artifact_uri)

    @task.virtualenv(
        task_id="evaluate_champion_challenge",
        requirements=["scikit-learn==1.3.2", "mlflow==2.10.2", "awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def evaluate_champion_challenge():
        import mlflow
        import awswrangler as wr

        from sklearn.metrics import f1_score

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

        def load_the_model(alias):
            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(config.MODEL_PROD_NAME, alias)

            model = mlflow.sklearn.load_model(model_data.source)

            return model

        def load_the_test_data():
            _, X_test, _, y_test = aux_functions.download_split_from_s3_final()

            return X_test, y_test

        def promote_challenger(name):

            client = mlflow.MlflowClient()

            # Demote the champion
            client.delete_registered_model_alias(name, config.PROD_ALIAS)

            # Load the challenger from registry
            challenger_version = client.get_model_version_by_alias(name, "challenger")

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

            # Transform in champion
            client.set_registered_model_alias(
                name, config.PROD_ALIAS, challenger_version.version
            )

        def demote_challenger(name):

            client = mlflow.MlflowClient()

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

        # Load the champion model
        champion_model = load_the_model(config.PROD_ALIAS)

        # Load the challenger model
        challenger_model = load_the_model("challenger")

        # Load the dataset
        X_test, y_test = load_the_test_data()

        # Obtain the metric of the models
        y_pred_champion = champion_model.predict(X_test)
        f1_score_champion = f1_score(y_test.to_numpy().ravel(), y_pred_champion)

        y_pred_challenger = challenger_model.predict(X_test)
        f1_score_challenger = f1_score(y_test.to_numpy().ravel(), y_pred_challenger)

        experiment = mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_f1_challenger", f1_score_challenger)
            mlflow.log_metric("test_f1_champion", f1_score_champion)

            if f1_score_challenger > f1_score_champion:
                mlflow.log_param("Winner", "Challenger")
            else:
                mlflow.log_param("Winner", "Champion")

            # Registrar el pipeline en MLFlow
            inputs_pipeline, target_pipeline = aux_functions.load_pipelines_from_s3()
            mlflow.sklearn.log_model(inputs_pipeline, config.INPUTS_PIPELINE_NAME)
            mlflow.sklearn.log_model(target_pipeline, config.TARGET_PIPELINE_NAME)

        if f1_score_challenger > f1_score_champion:
            promote_challenger(config.MODEL_PROD_NAME)
        else:
            demote_challenger(config.MODEL_PROD_NAME)

    train_the_challenger_model() >> evaluate_champion_challenge()


my_dag = processing_dag()
