from dataclasses import dataclass
import io
import json
import zipfile

from fastapi.responses import StreamingResponse
from loguru import logger

from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore
from fastapi_backend.schemas.data_types.store_data_type import StoreDataType
from fastapi_backend.utils import fetch_store_entry_with_checks


@dataclass
class ZipService:
    store_api: InMemoryStore

    def download_zip(self, image_id: int) -> StreamingResponse:
        store_entry = fetch_store_entry_with_checks(self.store_api, image_id)
        if store_entry.predictions is None:
            raise ValueError("No se encontraron predicciones para la imagen con el ID proporcionado. Ya las generó?")

        zip_buffer = self._create_zip_with_annotations(store_entry)
        logger.debug(f"Zip buffer size: {len(zip_buffer.getvalue())} bytes")
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=predictions_{image_id}.zip"},
        )

    def _create_zip_with_annotations(self, store_entry: StoreDataType) -> io.BytesIO:
        annotated_image_buffer_bytes = None
        if store_entry.encoded_annotated_image_buffer:
            store_entry.encoded_annotated_image_buffer.seek(0)
            annotated_image_buffer_bytes = store_entry.encoded_annotated_image_buffer.read()
            logger.debug(f"Annotated image bytes size from buffer: {len(annotated_image_buffer_bytes)} bytes")
        else:
            logger.warning(f"No annotated image buffer found for image ID: {store_entry.id}. Skipping addition to zip.")

        geojson = store_entry.predictions.__geo_interface__
        kml = store_entry.kml

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("annotated_image.jpg", annotated_image_buffer_bytes)
            zip_file.writestr("predictions.geojson", json.dumps(geojson).encode("utf-8"))
            zip_file.writestr("layer.kml", kml.encode("utf-8"))

        zip_buffer.seek(0)
        return zip_buffer
