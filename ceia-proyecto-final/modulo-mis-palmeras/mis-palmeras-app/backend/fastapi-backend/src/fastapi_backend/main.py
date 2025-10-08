from uvicorn import Server, Config
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi_keycloak_middleware import KeycloakConfiguration, setup_keycloak_middleware

# Local imports
from .config import settings
from .routers import api

PORT = settings.port
KEYCLOAK_URL = settings.keycloak_url
KEYCLOAK_REALM = settings.keycloak_realm
KEYCLOAK_CLIENT_ID = settings.keycloak_client_id
KEYCLOAK_CLIENT_SECRET = settings.keycloak_client_secret

ALLOW_CREDENTIALS = settings.cors_allow_credentials
ALLOW_ORIGINS = settings.cors_allow_origins
ALLOW_METHODS = settings.cors_allow_methods
ALLOW_HEADERS = settings.cors_allow_headers

TIMEOUT_KEEP_ALIVE = settings.timeout_keep_alive

keycloak_config = KeycloakConfiguration(
    url=KEYCLOAK_URL,
    realm=KEYCLOAK_REALM,
    client_id=KEYCLOAK_CLIENT_ID,
    client_secret=KEYCLOAK_CLIENT_SECRET,
)

app = FastAPI()

# NOTA DE 7 HORAS DE TRABAJO: El orden de los middlewares es importante. Keycloak debe estar antes de CORS, sino da error con OPTIONS.
# En FastAPI, los middlewares se ejecutan de abajo hacia arriba.
setup_keycloak_middleware(app, keycloak_configuration=keycloak_config)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=ALLOW_METHODS,
    allow_headers=ALLOW_HEADERS,
)

app.include_router(api.router)

if __name__ == "__main__":
    server = Server(Config(app, host="0.0.0.0", port=PORT, lifespan="on", timeout_keep_alive=TIMEOUT_KEEP_ALIVE))
    server.run()
