from mlflow.tracking import MlflowClient

# Dirección de tu servidor de MLflow
TRACKING_URI = "http://192.168.0.4:5000"

# ID del run que querés actualizar
RUN_ID = "b7218a073a3942339689b2e7f4e0b543"


def fix_run_status():
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Forzamos el estado a FINISHED
    client.set_terminated(run_id=RUN_ID, status="FINISHED")

    print(f"✅ Run {RUN_ID} actualizado a FINISHED")


if __name__ == "__main__":
    fix_run_status()
