from dataclasses import dataclass
import io
import json
from time import sleep
import zipfile

from fastapi.responses import StreamingResponse
import geopandas

from fastapi_backend.config import MODEL_PATH, RESOURCES_DIR, TARGET_IMG_SIZE_WH
from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore
from fastapi_backend.schemas.data_types.store_data_type import StoreDataType

from loguru import logger

import cv2

from modulo_ia.modeling.predict import DetectionModelPredictor

import modulo_apps.labeling.procesador_geojson_kml as ProcesadorGeoJSONKML

from fastapi_backend.utils import fetch_store_entry_with_checks


@dataclass
class PredictionService:
    store_api: InMemoryStore

    def generate_sync_mock_predictions(self, image_id: int) -> bool:
        store_entry = fetch_store_entry_with_checks(self.store_api, image_id)

        sleep(10)

        uri_annotated_image = RESOURCES_DIR / "mock_data" / "mock_annotated_image.jpg"
        uri_kml = RESOURCES_DIR / "mock_data" / "mock_layer.kml"
        uri_predictions = RESOURCES_DIR / "mock_data" / "mock_annotations.geojson"
        # Check if the URIs are valid and read the images
        if not uri_annotated_image.exists():
            raise FileNotFoundError(f"Annotated image not found at {uri_annotated_image}")
        if not uri_kml.exists():
            raise FileNotFoundError(f"KML file not found at {uri_kml}")
        if not uri_predictions.exists():
            raise FileNotFoundError(f"Predictions file not found at {uri_predictions}")

        store_entry.encoded_annotated_image_buffer = cv2.imread(str(uri_annotated_image), cv2.IMREAD_COLOR)

        with open(uri_kml, "r") as kml_file:
            content = kml_file.read()
        store_entry.kml = content

        store_entry.predictions = geopandas.read_file(uri_predictions)
        self.store_api.update(store_entry)

        return True

    def generate_sync_predictions(self, image_id: int) -> bool:
        store_entry = fetch_store_entry_with_checks(self.store_api, image_id)

        logger.debug(f"Model path: {MODEL_PATH}")
        model_predictor = DetectionModelPredictor(MODEL_PATH, TARGET_IMG_SIZE_WH)
        predictions = model_predictor.predict(store_entry.image).filter_by_confidence()
        if predictions.detections.is_empty():
            raise ValueError("No se encontraron detecciones en la imagen proporcionada.")
        coco_annotations = predictions.as_coco_annotations(pic_name=store_entry.name)

        store_entry.predictions = ProcesadorGeoJSONKML.create_geojson_from_annotations(
            pic_name=store_entry.name,
            coco_annotations=coco_annotations,
            jgw_data=store_entry.jgw.__dict__,
        )
        store_entry.kml = ProcesadorGeoJSONKML.generate_kml_from_geojson(gdf=store_entry.predictions).to_string()
        raw_annotated_image = predictions.get_annotated_image(store_entry.image)
        success, encoded_image = cv2.imencode(".jpg", raw_annotated_image)
        if not success:
            raise ValueError("Failed to encode the annotated image.")
        encoded_image_bytes = encoded_image.tobytes()
        store_entry.encoded_annotated_image_buffer = io.BytesIO(encoded_image_bytes)
        self.store_api.update(store_entry)

        return True

    async def generate_async_predictions(self, image_id: int) -> str:
        raise NotImplementedError("Asynchronous prediction generation is not implemented yet.")