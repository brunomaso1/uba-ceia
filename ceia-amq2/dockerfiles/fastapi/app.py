import json
import pickle
import mlflow
from dotenv import load_dotenv

import numpy as np
import pandas as pd

import boto3

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
import os

BUCKET_DATA = "data"
BOTO3_CLIENT = "s3"
INPUTS_PIPELINE_NAME = "inputs_pipeline.pkl"
TARGET_PIPELINE_NAME = "target_pipeline.pkl"
PIPES_DATA_FOLDER="pipes/"
s3_input_pipeline_path = PIPES_DATA_FOLDER + INPUTS_PIPELINE_NAME
s3_target_pipeline_path = PIPES_DATA_FOLDER + TARGET_PIPELINE_NAME


class ModelInput(BaseModel):
    """
    Esta clase define la estructura de datos de entrada para el modelo de predicción de lluvia.
    """

    date: str = Field(
        description="Fecha de los datos.",
    )
    location: Literal[
        "Adelaide",
        "Albany",
        "Albury",
        "AliceSprings",
        "BadgerysCreek",
        "Ballarat",
        "Bendigo",
        "Brisbane",
        "Cairns",
        "Canberra",
        "Cobar",
        "CoffsHarbour",
        "Dartmoor",
        "Darwin",
        "GoldCoast",
        "Hobart",
        "Katherine",
        "Launceston",
        "Melbourne",
        "MelbourneAirport",
        "Mildura",
        "Moree",
        "MountGambier",
        "MountGinini",
        "Newcastle",
        "Nhil",
        "NorahHead",
        "NorfolkIsland",
        "Nuriootpa",
        "PearceRAAF",
        "Penrith",
        "Perth",
        "PerthAirport",
        "Portland",
        "Richmond",
        "Sale",
        "SalmonGums",
        "Sydney",
        "SydneyAirport",
        "Townsville",
        "Tuggeranong",
        "Uluru",
        "WaggaWagga",
        "Walpole",
        "Watsonia",
        "Williamtown",
        "Witchcliffe",
        "Wollongong",
        "Woomera",
    ] = Field(
        description="Ubicación de la estación meteorológica.",
    )
    mintemp: float = Field(
        description="Temperatura mínima de hoy.",
        ge=-10,
        le=50,
    )
    maxtemp: float = Field(
        description="Temperatura máxima de hoy.",
        ge=-20,
        le=70,
    )
    rainfall: float = Field(
        description="Cantidad de lluvia caída hoy.",
    )
    evaporation: float = Field(
        description="Evaporación hoy.",
    )
    sunshine: float = Field(
        description="Horas de sol hoy.",
        ge=0,
        le=24,
    )
    windgustdir: Literal[
        "E",
        "ENE",
        "ESE",
        "N",
        "NE",
        "NNE",
        "NNW",
        "NW",
        "S",
        "SE",
        "SSE",
        "SSW",
        "SW",
        "W",
        "WNW",
        "WSW",
    ] = Field(
        description="Dirección de las ráfagas.",
    )
    windgustspeed: float = Field(
        description="Velocidad máxima de las ráfagas hoy.",
    )
    winddir9am: Literal[
        "E",
        "ENE",
        "ESE",
        "N",
        "NE",
        "NNE",
        "NNW",
        "NW",
        "S",
        "SE",
        "SSE",
        "SSW",
        "SW",
        "W",
        "WNW",
        "WSW",
    ] = Field(
        description="Dirección del viento a las 9am.",
    )
    winddir3pm: Literal[
        "E",
        "ENE",
        "ESE",
        "N",
        "NE",
        "NNE",
        "NNW",
        "NW",
        "S",
        "SE",
        "SSE",
        "SSW",
        "SW",
        "W",
        "WNW",
        "WSW",
    ] = Field(
        description="Dirección del viento a las 3pm.",
    )
    windspeed9am: float = Field(
        description="Velocidad del viento a las 9am.",
    )
    windspeed3pm: float = Field(
        description="Velocidad del viento a las 3pm.",
    )
    humidity9am: float = Field(
        description="Humedad a las 9am.",
    )
    humidity3pm: float = Field(
        description="Humedad a las 3pm.",
    )
    pressure9am: float = Field(
        description="Presión a las 9am.",
    )
    pressure3pm: float = Field(
        description="Presión a las 3pm.",
    )
    cloud9am: float = Field(
        description="Cobertura de nubes a las 9am.",
    )
    cloud3pm: float = Field(
        description="Cobertura de nubes a las 3pm.",
    )
    temp9am: float = Field(
        description="Temperatura a las 9am.",
    )
    temp3pm: float = Field(
        description="Temperatura a las 3pm.",
    )
    raintoday: float = Field(
        description="Llovio hoy? 1: si llovió, 0: no llovió.",
        ge=0,
    )

    # TODO: Change inputs to Sin + Cos for winddir9am, winddir3pm, location, date

    model_config = {
        "json_schema_extra": {
            "ejemplos": [
                {
                    "date": "2021-01-01",
                    "location": "Sydney",
                    "mintemp": 15.0,
                    "maxtemp": 25.0,
                    "rainfall": 0.0,
                    "evaporation": 5.0,
                    "sunshine": 10.0,
                    "windgustdir": "N",
                    "windgustspeed": 30.0,
                    "winddir9am": "N",
                    "winddir3pm": "N",
                    "windspeed9am": 10.0,
                    "windspeed3pm": 15.0,
                    "humidity9am": 50.0,
                    "humidity3pm": 60.0,
                    "pressure9am": 1010.0,
                    "pressure3pm": 1005.0,
                    "cloud9am": 5.0,
                    "cloud3pm": 5.0,
                    "temp9am": 20.0,
                    "temp3pm": 23.0,
                    "raintoday": 0,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Modelo de salida de la API.

    Esta clase define el modelo de salida de la API incluyendo una descripción.
    """

    prediction_bool: bool = Field(
        ..., description="Predicción de lluvia para el día próximo."
    )
    prediction_str: Literal[
        "Toma un paraguas. Mañana puede que llueva.",
        "Más seco que el Sahara. Mañana posiblemente no llueva...",
    ]

    model_config = {
        "json_schema_extra": {
            "ejemplos": [
                {
                    "prediction_bool": True,
                    "prediction_str": "Toma un paraguas. Mañana puede que llueva...",
                }
            ]
        }
    }


def load_model(model_name: str, alias: str = "prod_best"):
    """
    Función para cargar el modelo de predicción de lluvia.
    """

    try:
        # Se obtiene la ubicación del modelo guardado en MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        client_mlflow = mlflow.MlflowClient()

        # Se carga el modelo guardado en MLflow
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # # If there is no registry in MLflow, open the default model
        # file_ml = open('/app/files/model.pkl', 'rb')
        # model_ml = pickle.load(file_ml)
        # file_ml.close()
        # version_model_ml = 0

        # If an error occurs during the process, pass silently
        model_ml = None
        version_model_ml = None
        pass

    return model_ml, version_model_ml


def check_model():
    """
    Función para verificar si el modelo ha cambiado en el registro de modelos de MLflow
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "rain_dataset_model_prod"
        alias = "prod_best"

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass


def load_pipelines():
    client = boto3.client(BOTO3_CLIENT)
    obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_input_pipeline_path)
    inputs_pipeline: Pipeline = pickle.load(obj["Body"])
    obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_target_pipeline_path)
    target_pipeline: Pipeline = pickle.load(obj["Body"])
    print(inputs_pipeline)

load_dotenv()
# Load the model before start
model, version_model = load_model("rain_dataset_model_prod", "prod_best")
print(f"VARIABLES_ENTORNO={os.environ}")
# load_pipelines()

app = FastAPI()


@app.get("/")
async def get_root():
    """
    Endpoint de bienvenida.
    """
    return JSONResponse(
        content=jsonable_encoder(
            {"message": "Bienvenido a la API default de Rain Prediction"}
        )
    )


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks,
):
    """
    Endpoint para predecir si lloverá mañana o no, en base a las características del día actual.

    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(
        np.array(features_list).reshape([1, -1]), columns=features_key
    )

    # TODO: usar variables de entorno
    # s3_input_pipeline_path = PIPES_DATA_FOLDER + INPUTS_PIPELINE_NAME
    # PIPES_DATA_FOLDER = Variable.get("PIPES_DATA_FOLDER")
    # INPUTS_PIPELINE_NAME = "inputs_pipeline.pkl"

    s3_input_pipeline_path = "pipes/inputs_pipeline.pkl"

    client = boto3.client(BOTO3_CLIENT)
    obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_input_pipeline_path)
    inputs_pipeline = pickle.load(obj["Body"])

    features_df = inputs_pipeline.transform(features_df)

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = "Más seco que el Sahara... Mañana posiblemente no llueva."
    if prediction[0] > 0:
        str_pred = "Toma un paraguas. Mañana puede que llueva."

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)


@app.post(
    "/test/",
)
def test():
    JSONResponse(
        content=jsonable_encoder(
            {"message": "Bienvenidos a la API default de Rain Prediction"}
        )
    )
