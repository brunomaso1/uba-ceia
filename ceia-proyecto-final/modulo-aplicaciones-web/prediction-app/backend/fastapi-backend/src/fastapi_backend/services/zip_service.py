from dataclasses import dataclass
import io, json, zipfile
from fastapi.responses import StreamingResponse
from loguru import logger
from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore
from fastapi_backend.schemas.data_types.store_data_type import StoreDataType
from fastapi_backend.utils import fetch_store_entry_with_checks


@dataclass
class ZipService:
    """Servicio para crear y descargar un archivo ZIP con las predicciones y anotaciones de una imagen procesada."""
    store_api: InMemoryStore

    def download_zip(self, image_id: int) -> StreamingResponse:
        """
        Descarga un archivo ZIP que contiene las predicciones de una imagen específica.
        Esta función recupera las predicciones de una imagen desde el store API y crea un archivo ZIP
        que contiene las anotaciones correspondientes. El archivo se devuelve como una respuesta de
        streaming para su descarga directa.
        Args:
            image_id (int): El identificador único de la imagen para la cual se desean descargar
                           las predicciones.
        Returns:
            StreamingResponse: Una respuesta de streaming que contiene el archivo ZIP con las
                              predicciones. El archivo incluye headers apropiados para la descarga
                              con el nombre "predictions_{image_id}.zip".
        Raises:
            ValueError: Se lanza cuando no se encuentran predicciones para la imagen con el ID
                       proporcionado, sugiriendo que las predicciones aún no han sido generadas.
            HTTPException: Puede ser lanzada por fetch_store_entry_with_checks si:
                          - La imagen con el ID especificado no existe
                          - Hay problemas de conectividad con el store API
                          - El usuario no tiene permisos para acceder a la imagen
        Example:
            >>> zip_service = ZipService(store_api)
            >>> response = zip_service.download_zip(12345)
            >>> # Retorna StreamingResponse con archivo predictions_12345.zip
        Notes:
            - La función registra el tamaño del buffer ZIP en los logs de debug
            - El archivo ZIP se crea en memoria usando un buffer temporal
            - Es responsabilidad del cliente manejar la respuesta de streaming apropiadamente
            - Se recomienda verificar que existan predicciones antes de llamar esta función
        """

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
        """
        Crea un archivo ZIP que contiene una imagen anotada y datos de predicciones en múltiples formatos.
        Esta función toma una entrada de almacén que contiene una imagen anotada y predicciones,
        y genera un archivo ZIP con tres archivos: la imagen anotada en formato JPG, las
        predicciones en formato GeoJSON y una capa en formato KML.
        Args:
            store_entry (StoreDataType): Entrada del almacén que contiene:
                - encoded_annotated_image_buffer: Buffer de imagen anotada codificada
                - predictions: Objeto con predicciones que implementa __geo_interface__
                - kml: Cadena de texto con datos KML
                - id: Identificador único de la imagen
        Returns:
            io.BytesIO: Buffer de bytes que contiene el archivo ZIP con los siguientes archivos:
                - annotated_image.jpg: Imagen anotada en formato JPEG
                - predictions.geojson: Predicciones en formato GeoJSON
                - layer.kml: Capa de datos en formato KML
        Raises:
            AttributeError: Si store_entry.predictions no tiene el atributo __geo_interface__
            UnicodeEncodeError: Si hay problemas de codificación al convertir KML a bytes
            zipfile.BadZipFile: Si hay errores al crear el archivo ZIP
            IOError: Si hay problemas de entrada/salida al manipular los buffers
        Example:
            >>> store_data = StoreDataType(
            ...     id="img_001",
            ...     encoded_annotated_image_buffer=BytesIO(image_bytes),
            ...     predictions=geospatial_predictions,
            ...     kml="<kml>...</kml>"
            ... )
            >>> zip_buffer = service._create_zip_with_annotations(store_data)
            >>> with open("output.zip", "wb") as f:
            ...     f.write(zip_buffer.getvalue())
        Notes:
            - Si no existe un buffer de imagen anotada, se registra una advertencia y se 
              incluye None en el ZIP para annotated_image.jpg
            - La función asume que store_entry.predictions implementa la interfaz 
              __geo_interface__ para la conversión a GeoJSON
            - El buffer ZIP resultante tiene el puntero posicionado al inicio (seek(0))
            - Los datos KML y GeoJSON se codifican en UTF-8 antes de ser añadidos al ZIP
        """

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
