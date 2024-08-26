<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="resources/images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Aprendizaje de Máquinas II - MLOps</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Índice</summary>
  <ol>
    <li>
      <a href="#el-proyecto">El Proyecto</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#requisitos">Requisitos</a></li>
        <li><a href="#instalacion">Instalación</a></li>
      </ul>
    </li>
    <li><a href="#como-utilizar">Cómo Utilizar</a></li>
    <ul>
        <li><a href="#servicios">Servicios</a></li>
        <li><a href="#test-obtener-prediccion">Test (Obtener Predicción)</a></li>
        <ul>
          <li><a href="#utilizando-el-backend">Utilizando el Backend</a></li>
          <li><a href="#utilizando-el-frontend-con-gradio">Utilizando el Frontend con Gradio</a></li>
      </ul>
    </ul>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# El Proyecto

Este proyecto se realizó como Trabajo Práctico final para la materia **Aprendizaje de Máquinas II** de la **Carrera de Especialización en Inteligencia Artificial** de la Universidad de Buenos Aires.

## Contexto del proyecto
Se trabaja para una empresa llamada **ML Models and something more Inc.**, la cual ofrece un servicio que proporciona modelos mediante una REST API. Internamente, tanto para realizar tareas de DataOps como de MLOps, la empresa cuenta con Apache Airflow y MLflow. También dispone de un Data Lake en S3.

## La implementación y su arquitectura

![Arquitectura de la Implementación](resources\images\arquitectura_mlops.png)

La implementación incluye:
- MinIO como servicio de almacenamiento de objetos en *buckets* con dos rutas de almacenamiento: `s3://data` para almacenar los datasets a utilizar (en crudo, transformado, sets de train y test) y `s3://mlflow` para almacenar los artefactos y objetos relacionados a MLFlow. 
- En Apache Airflow se tienen 3 DAGs:
  1. `etl_process_rain_dataset`: este DAG obtiene el dataset (Extract); lo procesa utilizando pipelines para transformar los datos de entrada y prepararlos para el modelo, divide el dataset en *train* y *test* [Transform]; y por último, guarda los datasets procesados y divididos en un Bucket en MinIO (en `s3://data`). Este DAG también realiza un registro en MLflow de las tareas realizadas.
  2. `model_optimization`: este DAG se encarga de crear, entrenar y optimizar el primer modelo. Las tareas del DAG incluyen la carga de los datasets de *train* y *test* desde el bucket en MinIo, crea y registra el experimento en MLflow, crea un modelo de XGBoost y encuentra los mejores parámetros haciendo un ajuste de hiperparámetros, registra los hiperparámetros y el modelo en MLflow, prueba el modelo (cargando el modelo desde MLflow), y por último, registra el modelo con el alias `prod_best`.
  3. `retrain_model_rain_dataset`: este DAG reentrena el modelo con un nuevo dataset (en caso de estar disponible); compara el nuevo modelo, entrenado con el dataset actualizado, con el modelo entrenado anteriormente (sin hacer una búsqueda o ajuste de hiperparámetros); y elige el mejor modelo de los dos. En caso de que el modelo entrenado con el nuevo dataset resulte mejor (comparando el F1 Score), se le asigna el alias `prod_best`.
- Un servicio de API utilizando `FastAPI` para exponer el modelo almacenado en MLflow con el alias `prod_best` y permite predecir si lloverá al día siguiente.
- Un servicio de Gradio, como Front-End, que se conecta al endpoint de predicción de la API del sistema. En dicho sitio se introducen los datos del tiempo del día de hoy y se realiza solicita una predicción para el día siguiente. Una vez enviados los datos, se observa la respuesta: llueve o no llueve al día siguiente.
- Una base de datos Postgres para almacenar los datos de las aplicaciones MLFlow y Apache Airflow.



<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Built With

Para este proyecto se utilizó:

* [![MLflow][MLflow]][mlflow-url]
* [![Apache Airflow][Apache-Airflow]][airflow-url]
* [![MinIO][MinIO]][minio-url]
* [![PostgreSQL][PostgreSQL]][postgresql-url]
* [![FastAPI][FastAPI]][fastapi-url]
* [![Gradio][Gradio]][gradio-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
# Getting Started

Este proyecto está implementado utilizando contenedores en Docker para facilitar su deployment. Las librerías necesarias se instalan al momento de levantar los contenedores del proyecto.

## Requisitos

El proyecto está contenido en Docker, por lo que el requisito principal es tenerlo instalado localmente.
Para más información se puede visitar el sitio de [Docker](https://docs.docker.com/get-started/get-docker/).


## Instalación

1. Clonar este repositorio localmente.
2. Si estás en Linux o MacOS, en el archivo `.env`, reemplaza `AIRFLOW_UID` por el de tu usuario o alguno que consideres oportuno (para encontrar el UID, usa el comando `id -u <username>`). De lo contrario, Airflow dejará sus carpetas internas como root y no podrás subir DAGs (en airflow/dags) o plugins, etc.
3. En el directorio raíz de este repositorio, levantar el ambiente con:
```sh
docker compose --profile all up
```
4. Una vez que todos los servicios estén funcionando (se verifica con el comando `docker ps -a` que todos los servicios estén *healthy* o revisando en Docker Desktop), podrás acceder a los diferentes [servicios](#servicios). 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
# Cómo utilizar

A continuación se explica cómo utilizar este repositorio y comandos útiles para su ejecución e implementación local.

## Servicios

Una vez que todos los servicios estén funcionando (verifica con el comando docker ps -a que todos los servicios estén healthy o revisa en Docker Desktop), podrás acceder a los diferentes servicios mediante:

* Apache Airflow: [http://localhost:8080](http://localhost:8080)
* MLflow: [http://localhost:5000](http://localhost:5000)
* MinIO: [http://localhost:9001](http://localhost:9001) (ventana de administración de Buckets)
* API: [http://localhost:8800/](http://localhost:8800/)
* Docume*ntación de la API: [http://localhost:8800/docs](http://localhost:8800/docs)
* Gradio: [http://localhost:7860](http://localhost:7860)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Test (Obtener Predicción)

### Utilizando el Backend
Podemos realizar predicciones utilizando la API, accediendo a `http://localhost:8800/`.

Para hacer una predicción, debemos enviar una solicitud al endpoint `Predict` con un 
cuerpo de tipo JSON que contenga un campo de características (`features`) con cada 
entrada para el modelo.

Un ejemplo utilizando `curl` sería:

```bash
curl -X 'POST' \
  'http://localhost:8800/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": {
    "Date": "2024-08-26",
    "Location": "Adelaide",
    "MinTemp": 9.8,
    "MaxTemp": 18.2,
    "Rainfall": 0.2,
    "Evaporation": 1.4,
    "Sunshine": 0,
    "WindGustDir": "ESE",
    "WindGustSpeed": 39,
    "WindDir9am": "NW",
    "WindDir3pm": "W",
    "WindSpeed9am": 17,
    "WindSpeed3pm": 11,
    "Humidity9am": 91,
    "Humidity3pm": 95,
    "Pressure9am": 1020.3,
    "Pressure3pm": 1016.4,
    "Cloud9am": 7,
    "Cloud3pm": 8,
    "Temp9am": 12.7,
    "Temp3pm": 14.6,
    "RainToday": 0
  }
}'
```

La respuesta del modelo será un valor entero (1: lloverá, 0: no lloverá) y un mensaje en forma de cadena de texto que 
indicará si lloverá al día siguiente.

```json
{
  "prediction_bool": 0,
  "prediction_str": "Toma un paraguas. Mañana puede que llueva."
}
```

Para obtener más detalles sobre la API, ingresa a `http://localhost:8800/docs`.

Nota: Recuerda que si esto se ejecuta en un servidor diferente a tu computadora, debes reemplazar 
`localhost` por la IP correspondiente o el dominio DNS, si corresponde.

Nota: Recordar que si esto se ejecuta en un servidor aparte de tu computadora, reemplazar a 
localhost por la IP correspondiente o DNS domain si corresponde.

La forma en que se implementó tiene la desventaja de que solo se puede hacer una predicción a 
la vez, pero tiene la ventaja de que FastAPI y Pydantic nos permiten tener un fuerte control 
sobre los datos sin necesidad de agregar una línea de código adicional. FastAPI maneja toda 
la validación.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Utilizando el Frontend con Gradio

Otra forma disponible para realizar una predicción es ingresando al frontend de Gradio en `http://localhost:7860/`.

El sitio es una página única con un formulario que permite ingresar los valores del tiempo para un día específico y obtener una predicción de si lloverá o no al día siguiente.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Otros Comandos

### Docker compose
<!-- TODO: Hacer una tabla en vez de esto -->

Levantar el ambiente:
```sh
docker compose --profile all up
```

Eliminar el ambiente:
```sh
docker compose --profile all down
```

Eliminar el ambiente y volúmenes:
```sh
docker compose --profile all down -v
```

Eliminar todo (incluído imágenes):
```sh
docker compose down --rmi all --volumes
```

Meterte dentro de un contenedor:
```sh
# Utilizando el nombre del servicio (ej: airflow-webserver).
docker compose exec airflow-webserver /bin/bash
```

```sh
# Utilizando el id del contenedor.
docker compose exec <docker_id> /bin/bash
```

Reconstruir un contenedor y su imagen (cuando se agrega una dependencia global):
```sh
# --no-deps: No levanta servicios linkeados (supuestamente ya están levantados)
# --build: Construye la imagen antes de levantar el contenedor
docker compose up -d --no-deps --build <service_name>
```
Ejemplo, cuando se agrega un modulo en el requirements:
Nota: Le podes dar en otra terminal si no arrancaste docker compose de forma detachada.
```sh
docker compose up -d --no-deps --build airflow-webserver
```

Attachearte a un servicio:
```sh
docker compose --profile all up --no-deps --no-recreate --attach fastapi
```

Logs de FastAPI:
```sh
docker-compose logs -f fastapi
```

Reiniciar FastAPI (cargar otro modelo de entrada):
```sh
docker compose restart --no-deps fastapi
```

### Airflow

Una vez dentro de la instancia web de airflow, se puede utilizar el cli de airflow. Para no esperar el tiempo de 
refresh de los DAGs (una vez que actualizamos algún archivo de python):

```sh
airflow dags reserialize
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
# License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & images -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->


[MLflow]: https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=MLflow&logoColor=white
[mlflow-url]: https://mlflow.org/
[Apache-Airflow]: https://img.shields.io/badge/Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white
[airflow-url]: https://airflow.apache.org/
[MinIO]: https://img.shields.io/badge/MinIO-C72E49?style=for-the-badge&logo=MinIO&logoColor=white
[minio-url]: https://min.io
[PostgreSQL]: https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white
[postgresql-url]: https://www.postgresql.org/
[FastAPI]: https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white
[fastapi-url]: https://fastapi.tiangolo.com/
[Gradio]: https://img.shields.io/badge/Gradio-F08705?style=for-the-badge&logo=Gradio&logoColor=white
[gradio-url]: https://www.gradio.app/