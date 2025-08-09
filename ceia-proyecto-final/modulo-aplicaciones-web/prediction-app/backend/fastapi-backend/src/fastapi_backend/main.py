import json
from uvicorn import Server, Config
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from fastapi_backend.routers import api
from fastapi_backend.config import (
    PORT,
)
from fastapi_keycloak_middleware import KeycloakConfiguration, setup_keycloak_middleware


keycloak_config = KeycloakConfiguration(
    url="http://localhost:7000",
    realm="prediction-app",
    client_id="prediction-app-backend",
    client_secret="tKRGTPVME5JN3m4aklYA0EfFGzgDa4Y9",
)

app = FastAPI()

# NOTA DE 7 HORAS DE TRABAJO: El orden de los middlewares es importante. Keycloak debe estar antes de CORS, sino da error con OPTIONS.
# En FastAPI, los middlewares se ejecutan de abajo hacia arriba.
setup_keycloak_middleware(app, keycloak_configuration=keycloak_config)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins="*",
    allow_methods="*",
    allow_headers="*",
)

app.include_router(api.router)

if __name__ == "__main__":
    server = Server(Config(app, host="0.0.0.0", port=PORT, lifespan="on", timeout_keep_alive=600))
    server.run()
