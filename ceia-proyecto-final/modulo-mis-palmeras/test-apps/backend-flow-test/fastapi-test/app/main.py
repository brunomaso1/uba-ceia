# System imports
import os
from pathlib import Path
from uuid import UUID
from contextlib import asynccontextmanager

# Local imports
from .db import connect_db, close_db, get_database_url, get_db_connection

# Third-party imports
from fastapi import FastAPI, Depends
from celery import Celery
from psycopg import AsyncConnection

ROOT_DIR = Path(__file__).parent.parent  # fastapi-test
RESOURCES_DIR = ROOT_DIR / "app" / "resources"

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "pyamqp://guest:guest@rabbitmq:5672/ai_vhost")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://:redispassword@redis:6379/0")
CELERY_MAIN_MODULE_NAME = os.getenv("CELERY_MAIN_MODULE_NAME", "tasks")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of the application."""
    # Startup: Initialize database pool
    print("App iniciando...")
    await connect_db(get_database_url())
    yield  # ← App corre aquí (acepta requests)
    # Shutdown: Close database pool
    print("App cerrando...")
    await close_db()


app = FastAPI(lifespan=lifespan)
celery = Celery(CELERY_MAIN_MODULE_NAME, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


@app.get("/health/")
async def health(conn: AsyncConnection = Depends(get_db_connection)):
    # Check if the Celery broker is reachable
    try:
        celery.control.ping(timeout=1)
        broker_status = "reachable"
    except Exception as e:
        broker_status = f"unreachable: {str(e)}"

    # Check if the database is reachable
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
        db_status = "reachable"
    except Exception as e:
        db_status = f"unreachable: {str(e)}"

    return {"status": "ok", "broker_status": broker_status, "database_status": db_status}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/tasks/add/")
async def create_add_task(x: int, y: int, conn: AsyncConnection = Depends(get_db_connection)):
    try:
        async with conn.cursor() as cur:
            user_id = UUID("123e4567-e89b-12d3-a456-426614174000")  # Example user ID
            endpoint = "/tasks/add/"
            await cur.execute(
                "INSERT INTO requests (user_id, endpoint) VALUES (%s, %s)",
                (str(user_id), endpoint),
            )
        await conn.commit()
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}

    # Envía el mensaje al broker: "ejecuta app.tasks.add con x, y"
    result = celery.send_task("app.tasks.add", args=(x, y))
    return {"task_id": result.id}


@app.get("/tasks/{task_id}/")
async def get_task_result(task_id: str):
    result = celery.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result}
