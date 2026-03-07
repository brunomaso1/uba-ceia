# System imports
import os

# Local imports

# Third-party imports
from fastapi import FastAPI
from celery import Celery

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "pyamqp://guest:guest@rabbitmq:5672/ai_vhost")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://:redispassword@redis:6379/0")
CELERY_MAIN_MODULE_NAME = os.getenv("CELERY_MAIN_MODULE_NAME", "tasks")


app = FastAPI()
celery = Celery(CELERY_MAIN_MODULE_NAME, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


@app.get("/health/")
async def health():
    # Check if the Celery broker is reachable
    try:
        celery.control.ping(timeout=1)
        broker_status = "reachable"
    except Exception as e:
        broker_status = f"unreachable: {str(e)}"

    return {"status": "ok", "broker_status": broker_status}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/tasks/add/")
async def create_add_task(x: int, y: int):
    # Envía el mensaje al broker: "ejecuta app.tasks.add con x, y"
    result = celery.send_task("app.tasks.add", args=(x, y))
    return {"task_id": result.id}


@app.get("/tasks/{task_id}/")
async def get_task_result(task_id: str):
    result = celery.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result}
