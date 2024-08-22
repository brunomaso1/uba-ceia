import pickle
from pprint import pprint
import boto3
import json

import boto3.s3
import boto3.session
import awswrangler as wr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
import pandas as pd

DATASET_NAME = 'rain.csv'
COLUMNS_TYPE_FILE_NAME = 'columnsTypes.json'
TARGET_PIPELINE_NAME = 'target_pipeline.pkl'
INPUTS_PIPELINE_NAME = 'inputs_pipeline.pkl'
BUCKET_DATA = 'data'
BOTO3_CLIENT = 's3'
X_TRAIN_NAME = 'X_train.csv'
X_TEST_NAME = 'X_test.csv'
X_TEST_NAME = 'y_train.csv'
Y_TEST_NAME = 'y_test.csv'

S3_RAW_DATA_FOLDER = 's3://data/raw/'
S3_PREPROCESED_DATA_FOLDER = 's3://data/preprocesed/'
S3_INFO_DATA_FOLDER = 's3://data/info/'
S3_PIPES_DATA_FOLDER = 's3://data/pipes/'
S3_FINAL_DATA_FOLDER = 's3://data/final/'

INFO_DATA_FOLDER = 'info/'
PIPES_DATA_FOLDER = 'pipes/'

TEST_SIZE = 0.2

s3_columns_path = INFO_DATA_FOLDER + COLUMNS_TYPE_FILE_NAME

def map_bool(x):
    if isinstance(x, pd.DataFrame):
        return x.applymap(lambda y: {"Yes": 1, "No": 0}.get(y, y))
    else:
        return x.map({"Yes": 1, "No": 0})


def to_category(x):
    return x.astype('category')


def to_datetime(x):
    return x.astype('datetime64[ns]')


def save_to_csv(df, path):
    wr.s3.to_csv(df=df, path=path, index=False)


def process_column_types():
    columns_types = {
        'cat_columns': ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],
        'bool_columns': ['RainToday'],
        'date_columns': ['Date'],
        'cont_columns': ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                         'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                         'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                         'Cloud3pm', 'Temp9am', 'Temp3pm'],
        'target_columns': ['RainTomorrow']
    }

    s3_columns_path = S3_PREPROCESED_DATA_FOLDER + COLUMNS_TYPE_FILE_NAME

    # Cargar la configuración de Minio
    session = boto3.session.Session()
    client = session.client('s3',
                            endpoint_url='http://localhost:9000',
                            aws_access_key_id='minio',
                            aws_secret_access_key='minio123')

    # client = boto3.client('s3')
    # Convertir el diccionario a JSON y subirlo a S3
    json_data = json.dumps(columns_types)
    client.put_object(
        Bucket='data', Key='preprocesed/columnsTypes.json', Body=json_data)

    # Leer
    obj = client.get_object(Bucket='data', Key=s3_columns_path)
    columns_types = json.load(obj['Body'])

    return s3_columns_path


def download_dataset_from_minio():
    import boto3
    import awswrangler as wr
    import pandas as pd

    # Configuración de la conexión a Minio
    endpoint_url = "http://localhost:9000"
    access_key = "minio"
    secret_key = "minio123"

    # Crear una sesión de boto3
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # Configurar awswrangler para usar Minio
    wr.config.s3_endpoint_url = endpoint_url

    df = wr.s3.read_csv(path=S3_RAW_DATA_FOLDER +
                        DATASET_NAME, boto3_session=session)

    if df is not None:
        print(df.head())
        print(f"Tamaño del dataset: {df.shape}")
    else:
        print("No se pudo descargar el dataset.")


def process_geolocations():
    import boto3
    # import geopandas as gpd
    # from geopandas.datasets import get_path
    import pandas as pd
    # import osmnx as ox
    # import re
    # from shapely.geometry import Point

    # country = "Australia"

    # world = gpd.read_file(get_path('naturalearth_lowres'))
    # gdf_australia = world[world.name == country]

    # # Solve manually some mistaken names
    # mapping_dict = {"Dartmoor": "DartmoorVillage", "Richmond": "RichmondSydney"}

    # # Una vez aplicado map, quedan valores NaN, por lo que se completa con los valores que ya tenía,
    # # o sea para cualquier valor que sea NaN después de aplicar map(),
    # # usa el valor original que estaba en esa posición en la columna 'Location
    # df["Location"] = df["Location"].map(mapping_dict).fillna(df["Location"])

    # locations = df["Location"].unique()

    # # Separa las ubicaciones en camelCase con un espacio. Ej: NorthRyde -> North Ryde
    # locations = [re.sub(r'([a-z])([A-Z])', r'\1 \2', l) for l in locations]

    # locs = []
    # lats = []
    # lons = []
    # for location in locations:
    #     try:
    #         lat, lon = ox.geocode(location + f", {country}")

    #         locs.append(location.replace(" ", ""))
    #         lats.append(lat)
    #         lons.append(lon)
    #     except Exception as e:
    #         print(f"Error retrieving coordinates for {location}: {e}")

    # df_locations = pd.DataFrame({
    #     'Location': locs,
    #     'Lat': lats,
    #     'Lon': lons
    # })
    # geometry = [Point(lon, lat) for lon, lat in zip(
    #     df_locations['Lon'], df_locations['Lat'])]
    # gdf_locations = gpd.GeoDataFrame(
    #     df_locations, geometry=geometry, crs="EPSG:4326")
    # gdf_locations.to_file('./data/gdf_locations.geojson', driver='GeoJSON')

def create_target_pipe():
    # Configuración de la conexión a Minio
    endpoint_url = "http://localhost:9000"
    access_key = "minio"
    secret_key = "minio123"

    # Crear una sesión de boto3
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # Configurar awswrangler para usar Minio
    wr.config.s3_endpoint_url = endpoint_url

    # Cargar la configuración de Minio
    client = session.client(BOTO3_CLIENT,
                            endpoint_url=endpoint_url,
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key)
    
    target_pipeline = Pipeline(steps=[])
    target_pipeline.steps.append(
        ('mapping', FunctionTransformer(map_bool)))

    s3_target_pipeline_path = PIPES_DATA_FOLDER + TARGET_PIPELINE_NAME
    
    client.put_object(Bucket=BUCKET_DATA, Key=s3_target_pipeline_path,
                        Body=pickle.dumps(target_pipeline))

    return s3_target_pipeline_path

def create_inputs_pipe(s3_columns_path):
    # Configuración de la conexión a Minio
    endpoint_url = "http://localhost:9000"
    access_key = "minio"
    secret_key = "minio123"

    # Crear una sesión de boto3
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # Configurar awswrangler para usar Minio
    wr.config.s3_endpoint_url = endpoint_url

    # Cargar la configuración de Minio
    client = session.client(BOTO3_CLIENT,
                            endpoint_url=endpoint_url,
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key)
    
    inputs_pipeline = Pipeline(steps=[])

    obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_columns_path)
    columns_types = json.load(obj['Body'])
    cat_columns = columns_types['cat_columns']
    bool_columns = columns_types['bool_columns']
    date_columns = columns_types['date_columns']
    cont_columns = columns_types['cont_columns']
    target_columns = columns_types['target_columns']

    col_types_transf = ColumnTransformer(
        [('categories', FunctionTransformer(to_category), cat_columns),
            ('date', FunctionTransformer(to_datetime), date_columns),
            ('bool', FunctionTransformer(map_bool), bool_columns)],
        remainder='passthrough',
        verbose_feature_names_out=False).set_output(transform='pandas')

    inputs_pipeline.steps.append(('feature_transf', col_types_transf))

    # TODO: Agregar todas las tranformaciones faltantes.

    s3_input_pipeline_path = PIPES_DATA_FOLDER + INPUTS_PIPELINE_NAME


    client.put_object(Bucket=BUCKET_DATA, Key=s3_input_pipeline_path,
                        Body=pickle.dumps(inputs_pipeline))

    return s3_input_pipeline_path

def test_pipelines():
    # Configuración de la conexión a Minio
    endpoint_url = "http://localhost:9000"
    access_key = "minio"
    secret_key = "minio123"

    # Crear una sesión de boto3
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # Configurar awswrangler para usar Minio
    wr.config.s3_endpoint_url = endpoint_url

    # Cargar la configuración de Minio
    client = session.client(BOTO3_CLIENT,
                            endpoint_url=endpoint_url,
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key)

    s3_input_pipeline_path = PIPES_DATA_FOLDER + INPUTS_PIPELINE_NAME
    s3_target_pipeline_path = PIPES_DATA_FOLDER + TARGET_PIPELINE_NAME

    train_test_split_preprocesed_path = {
        'X_train': S3_PREPROCESED_DATA_FOLDER + X_TRAIN_NAME,
        'X_test': S3_PREPROCESED_DATA_FOLDER + X_TEST_NAME,
        'y_train': S3_PREPROCESED_DATA_FOLDER + X_TEST_NAME,
        'y_test': S3_PREPROCESED_DATA_FOLDER + Y_TEST_NAME
    }

    def save_to_csv(df, path):
        wr.s3.to_csv(df=df, path=path, index=False, boto3_session=session)    

    X_train = wr.s3.read_csv(train_test_split_preprocesed_path['X_train'], boto3_session=session)
    X_test = wr.s3.read_csv(train_test_split_preprocesed_path['X_test'], boto3_session=session)
    y_train = wr.s3.read_csv(train_test_split_preprocesed_path['y_train'], boto3_session=session)
    y_test = wr.s3.read_csv(train_test_split_preprocesed_path['y_test'], boto3_session=session)

    obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_input_pipeline_path)
    inputs_pipeline = pickle.load(obj['Body'])
    obj = client.get_object(Bucket=BUCKET_DATA, Key=s3_target_pipeline_path)
    target_pipeline = pickle.load(obj['Body'])

    X_train = inputs_pipeline.fit_transform(X_train)
    # X_test = inputs_pipeline.transform(X_test)
    y_train = target_pipeline.fit_transform(y_train)
    y_test = target_pipeline.transform(y_test)

    # train_test_split_final_path = {
    #     'X_train': S3_FINAL_DATA_FOLDER + X_TRAIN_NAME,
    #     'X_test': S3_FINAL_DATA_FOLDER + X_TEST_NAME,
    #     'y_train': S3_FINAL_DATA_FOLDER + X_TEST_NAME,
    #     'y_test': S3_FINAL_DATA_FOLDER + Y_TEST_NAME
    # }

    # save_to_csv(X_train, train_test_split_final_path['X_train'])
    # save_to_csv(X_test, train_test_split_final_path['X_test'])
    # save_to_csv(y_train, train_test_split_final_path['y_train'])
    # save_to_csv(y_test, train_test_split_final_path['y_test'])

# process_column_types()
# download_dataset_from_minio()
# process_geolocations()
create_inputs_pipe(s3_columns_path)
create_target_pipe()
test_pipelines()
