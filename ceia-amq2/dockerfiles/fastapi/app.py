import json
import pickle
import mlflow
import awswrangler
import airflow
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
from datetime import date

import utils.rain_dataset.rain_dataset_tasks.tasks_utils
from utils.rain_dataset.rain_dataset_configs.config_loader import RainDatasetConfigs
config = RainDatasetConfigs()


import os

class ModelInput(BaseModel):
    """
    Esta clase define la estructura de datos de entrada para el modelo de predicción de lluvia.
    """

    Date: date = Field(
        description="Fecha de los datos.",
    )
    Location: Literal[
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
    MinTemp: float = Field(
        description="Temperatura mínima de hoy.",
        ge=-10,
        le=50,
    )
    MaxTemp: float = Field(
        description="Temperatura máxima de hoy.",
        ge=-20,
        le=70,
    )
    Rainfall: float = Field(
        description="Cantidad de lluvia caída hoy.",
    )
    Evaporation: float = Field(
        description="Evaporación hoy.",
    )
    Sunshine: float = Field(
        description="Horas de sol hoy.",
        ge=0,
        le=24,
    )
    WindGustDir: Literal[
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
    WindGustSpeed: float = Field(
        description="Velocidad máxima de las ráfagas hoy.",
    )
    WindDir9am: Literal[
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
    WindDir3pm: Literal[
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
    WindSpeed9am: float = Field(
        description="Velocidad del viento a las 9am.",
    )
    WindSpeed3pm: float = Field(
        description="Velocidad del viento a las 3pm.",
    )
    Humidity9am: float = Field(
        description="Humedad a las 9am.",
    )
    Humidity3pm: float = Field(
        description="Humedad a las 3pm.",
    )
    Pressure9am: float = Field(
        description="Presión a las 9am.",
    )
    Pressure3pm: float = Field(
        description="Presión a las 3pm.",
    )
    Cloud9am: float = Field(
        description="Cobertura de nubes a las 9am.",
    )
    Cloud3pm: float = Field(
        description="Cobertura de nubes a las 3pm.",
    )
    Temp9am: float = Field(
        description="Temperatura a las 9am.",
    )
    Temp3pm: float = Field(
        description="Temperatura a las 3pm.",
    )
    RainToday: float = Field(
        description="Llovio hoy? 1: si llovió, 0: no llovió.",
        ge=0,
    )

    # TODO: Change inputs to Sin + Cos for winddir9am, winddir3pm, location, date

    model_config = {
        "json_schema_extra": {
            "ejemplos": [
                {
                    "Date": "2021-01-01",
                    "Location": "Sydney",
                    "MinTemp": 15.0,
                    "MaxTemp": 25.0,
                    "Rainfall": 0.0,
                    "Evaporation": 5.0,
                    "Sunshine": 10.0,
                    "windgustdir": "N",
                    "WindGustSpeed": 30.0,
                    "WindDir9am": "N",
                    "WindDir3pm": "N",
                    "WindSpeed9am": 10.0,
                    "WindSpeed3pm": 15.0,
                    "Humidity9am": 50.0,
                    "Humidity3pm": 60.0,
                    "Pressure9am": 1010.0,
                    "Pressure3pm": 1005.0,
                    "Cloud9am": 5.0,
                    "Cloud3pm": 5.0,
                    "Temp9am": 20.0,
                    "Temp3pm": 23.0,
                    "RainToday": 0,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Modelo de salida de la API.

    Esta clase define el modelo de salida de la API incluyendo una descripción.
    """

    prediction_bool: int = Field(
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
                    "prediction_bool": 1,
                    "prediction_str": "Toma un paraguas. Mañana puede que llueva...",
                }
            ]
        }
    }


# def load_model(model_name: str, alias: str = "prod_best"):
#     """
#     Función para cargar el modelo de predicción de lluvia.
#     """

#     try:
#         # Se obtiene la ubicación del modelo guardado en MLflow
#         mlflow.set_tracking_uri("http://mlflow:5000")
#         client_mlflow = mlflow.MlflowClient()

#         print("step 1 completed")

#         # Se carga el modelo guardado en MLflow
#         model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
#         model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
#         version_model_ml = int(model_data_mlflow.version)

#         print("step 2 completed")

#         # Cargar los pipelines de entrada y objetivo desde MLFlow
#         input_pipeline_uri = client_mlflow.get_model_version_by_alias("Rain_dataset_etl_inputs_pipeline", alias).source
#         print(input_pipeline_uri)
#         input_pipeline = mlflow.sklearn.load_model(input_pipeline_uri)

#         print("step 3, input pipeline load completed")

#         target_pipeline_uri = client_mlflow.get_model_version_by_alias("Rain_dataset_etl_target_pipeline", alias).source
#         target_pipeline = mlflow.sklearn.load_model(target_pipeline_uri)

#         print("step 4, target pipeline load completed")

#     except Exception as e:
#         print("Exception!!!")
#         print(e)
#         print("Error loading model")
#         # If there is no registry in MLflow, open the default model
#         file_ml = open('/app/files/model.pkl', 'rb')
#         model_ml = pickle.load(file_ml)
#         file_ml.close()
#         version_model_ml = 0

#         input_pipeline_file = open('/app/files/inputs_pipeline.pkl', 'rb')
#         input_pipeline = pickle.load(input_pipeline_file)
#         input_pipeline_file.close()

#         target_pipeline_file = open('/app/files/inputs_pipeline.pkl', 'rb')
#         target_pipeline = pickle.load(target_pipeline_file)
#         target_pipeline_file.close()

#         # If an error occurs during the process, pass silently
#         # model_ml = None
#         # version_model_ml = None
#         # input_pipeline = None
#         # target_pipeline = None
#         pass

#     return model_ml, version_model_ml, input_pipeline, target_pipeline

def load_model(model_name: str, alias: str = "prod_best"):
    """
    Función para cargar el modelo de predicción de lluvia.
    """

    try:
        # Se obtiene la ubicación del modelo guardado en MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        client_mlflow = mlflow.MlflowClient()

        print("step 1 completed")

        # Se carga el modelo guardado en MLflow
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)

        print("step 2 completed")

        # Cargar los pipelines de entrada y objetivo desde MLFlow
        input_pipeline_uri = client_mlflow.get_model_version_by_alias("Rain_dataset_etl_inputs_pipeline", alias).source
        print(input_pipeline_uri)
        input_pipeline = mlflow.sklearn.load_model(input_pipeline_uri)

        print("step 3, input pipeline load completed")

        target_pipeline_uri = client_mlflow.get_model_version_by_alias("Rain_dataset_etl_target_pipeline", alias).source
        target_pipeline = mlflow.sklearn.load_model(target_pipeline_uri)

        print("step 4, target pipeline load completed")

    except Exception as e:
        print("Exception!!!")
        print(e)
        print("Error loading model")
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

        input_pipeline_file = open('/app/files/inputs_pipeline.pkl', 'rb')
        input_pipeline = pickle.load(input_pipeline_file)
        input_pipeline_file.close()

        target_pipeline_file = open('/app/files/inputs_pipeline.pkl', 'rb')
        target_pipeline = pickle.load(target_pipeline_file)
        target_pipeline_file.close()

        # If an error occurs during the process, pass silently
        # model_ml = None
        # version_model_ml = None
        # input_pipeline = None
        # target_pipeline = None
        pass

    return model_ml, version_model_ml, input_pipeline, target_pipeline


def check_model():
    """
    Función para verificar si el modelo ha cambiado en el registro de modelos de MLflow
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "Rain_dataset_model_prod"
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

# Load the model before start
model, version_model, inputs_pipeline, target_pieline = load_model("Rain_dataset_model_prod", "prod_best")

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

    print("Features before transform=")
    print(features_df)
    features_df = inputs_pipeline.transform(features_df)
    print("Features after transform=")
    print(features_df)

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