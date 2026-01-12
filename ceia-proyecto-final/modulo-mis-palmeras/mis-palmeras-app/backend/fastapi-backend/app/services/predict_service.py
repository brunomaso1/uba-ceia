# Dependencias del sistema
from dataclasses import dataclass
import io
import time

# Dependencias locales
from ..config import settings
from ..schemas.data_types.models_types import ModelType
from ..utils import fetch_store_entry_with_checks
from ..dependencies.in_memory_store_api import InMemoryStore

# Dependencias propias
from modulo_ia.core.modeling.predict import DetectionModelPredictor
from modulo_utilidades.core.labeling import procesador_geojson_kml_core

# Dependencias de terceros
from loguru import logger
import cv2

RESOURCES_DIR = settings.folders.resources_dir

# Common model parameters
TARGET_IMG_SIZE_WH = settings.target_img_size_wh
OVERLAP_RATIO_WH = settings.overlap_ratio_wh

PALM_MODEL_PATH = settings.palm_model.model_path
PALM_MODEL_CLASS_NAMES = settings.palm_model.class_names
PALM_MIN_RATIO = settings.palm_model.min_ratio
PALM_NMS_IOU_THRESHOLD = settings.palm_model.nms_iou_threshold
PALM_CONTAINERMENT_THRESHOLD = settings.palm_model.containerment_threshold
PALM_CONFIDENCE = settings.palm_model.confidence

RPW_MODEL_PATH = settings.rpw_model.model_path
RPW_MODEL_CLASS_NAMES = settings.rpw_model.class_names
RPW_MIN_RATIO = settings.rpw_model.min_ratio
RPW_NMS_IOU_THRESHOLD = settings.rpw_model.nms_iou_threshold
RPW_CONTAINERMENT_THRESHOLD = settings.rpw_model.containerment_threshold
RPW_CONFIDENCE = settings.rpw_model.confidence


@dataclass
class PredictionService:
    """Servicio para manejar predicciones de modelos de detección de objetos."""

    store_api: InMemoryStore

    def generate_sync_predictions(self, image_id: int, model_type: ModelType = ModelType.PALM_DETECTION) -> bool:
        """
        Genera predicciones síncronas para una imagen específica utilizando un modelo de detección.
        Este método procesa una imagen del almacén, ejecuta el modelo de predicción especificado,
        genera anotaciones en formato COCO, convierte las predicciones a GeoJSON y KML, y
        almacena la imagen anotada junto con todos los metadatos generados.
        Args:
            image_id (int): Identificador único de la imagen en el almacén de datos.
            model_type (ModelType, optional): Tipo de modelo a utilizar para la predicción.
                Por defecto es ModelType.PALM_DETECTION.
        Returns:
            bool: True si la predicción se generó y almacenó exitosamente, False en caso contrario.
        Raises:
            ValueError: Si falla la codificación de la imagen anotada o si la imagen no puede
                ser procesada por el modelo.
            StoreEntryNotFoundError: Si el image_id especificado no existe en el almacén.
            ModelProcessingError: Si el modelo no puede procesar la imagen correctamente.
            GeospatialProcessingError: Si falla la conversión a GeoJSON o KML.
        Example:
            >>> predict_service = PredictService(store_api)
            >>> success = predict_service.generate_sync_predictions(
            ...     image_id=123,
            ...     model_type=ModelType.PALM_DETECTION
            ... )
            >>> print(success)
            True
        Notes:
            - La función modifica directamente el store_entry en el almacén de datos.
            - La imagen anotada se codifica como JPEG y se almacena en un buffer de memoria.
            - Las coordenadas geoespaciales se obtienen de los datos JGW asociados a la imagen.
            - El proceso es síncrono y puede tardar varios segundos dependiendo del tamaño
              de la imagen y la complejidad del modelo.
        """
        # Measure time
        start_time = time.time()
        store_entry = fetch_store_entry_with_checks(self.store_api, image_id)
        predictions, coco_annotations = self._generate_sync_predictions(store_entry.image, store_entry.name, model_type)
        store_entry.predictions = procesador_geojson_kml_core.create_geojson_from_annotations(
            pic_name=store_entry.name,
            coco_annotations=coco_annotations,
            jgw_data=store_entry.jgw.__dict__,
        )
        store_entry.kml = procesador_geojson_kml_core.generate_kml_from_geojson(gdf=store_entry.predictions).to_string()
        raw_annotated_image = predictions.get_annotated_image(store_entry.image)
        success, encoded_image = cv2.imencode(".jpg", raw_annotated_image)
        if not success:
            raise ValueError("Failed to encode the annotated image.")
        encoded_image_bytes = encoded_image.tobytes()
        store_entry.encoded_annotated_image_buffer = io.BytesIO(encoded_image_bytes)
        self.store_api.update(store_entry)
        end_time = time.time()
        logger.debug(
            f"Synchronous prediction generated for image ID {image_id} using model {model_type.name} "
            f"in {end_time - start_time:.2f} seconds."
        )

        return True

    async def generate_async_predictions(self, image_id: int, model_type: ModelType = ModelType.PALM_DETECTION) -> str:
        """
        Genera predicciones de manera asíncrona para una imagen específica utilizando el modelo especificado.
        Esta función crea un trabajo en segundo plano (background job) de FastAPI para procesar
        las predicciones de manera asíncrona, permitiendo que el cliente reciba una respuesta
        inmediata mientras el procesamiento continúa en el background.
        Args:
            image_id (int): Identificador único de la imagen sobre la cual se realizarán las predicciones.
            model_type (ModelType, optional): Tipo de modelo a utilizar para las predicciones.
                Por defecto es ModelType.PALM_DETECTION.
        Returns:
            str: Identificador único del job/tarea creada que puede ser utilizado para
                consultar el estado y obtener los resultados de la predicción asíncrona.
        Raises:
            NotImplementedError: La función aún no ha sido implementada.
            ValueError: Si el image_id proporcionado no es válido o no existe.
            ModelNotAvailableError: Si el tipo de modelo especificado no está disponible.
            InsufficientResourcesError: Si no hay recursos suficientes para procesar la tarea.
        Example:
            >>> service = PredictService()
            >>> job_id = await service.generate_async_predictions(
            ...     image_id=123,
            ...     model_type=ModelType.PALM_DETECTION
            ... )
            >>> print(f"Job creado con ID: {job_id}")
            Job creado con ID: job_abc123xyz
        Notes:
            - Esta función está diseñada para manejar predicciones que pueden tomar tiempo considerable.
            - El job_id retornado puede ser usado con otros endpoints para consultar el estado del procesamiento.
            - La implementación futura utilizará el sistema de background tasks de FastAPI.
            - Se recomienda implementar un sistema de polling o webhooks para notificar cuando
              la predicción esté completa.
        """

        raise NotImplementedError("Asynchronous prediction generation is not implemented yet.")

    def _generate_sync_predictions(self, image, name, model_type: ModelType):
        """
        Genera predicciones síncronas utilizando modelos de detección de objetos.
        Esta función procesa una imagen utilizando el modelo especificado para detectar
        objetos (palmas o plagas RPW) y retorna tanto las predicciones como las
        anotaciones en formato COCO.
        Args:
            image: Imagen a procesar para la detección de objetos.
            name (str): Nombre del archivo de imagen para incluir en las anotaciones.
            model_type (ModelType): Tipo de modelo a utilizar (PALM_DETECTION o RPW_DETECTION).
        Returns:
            tuple: Una tupla conteniendo:
                - predictions: Objeto con las predicciones filtradas por confianza mínima.
                - coco_annotations: Anotaciones en formato COCO v1 con las detecciones.
        Raises:
            ValueError: Si el tipo de modelo proporcionado no está soportado.
            FileNotFoundError: Si el archivo del modelo no existe en la ruta especificada.
            RuntimeError: Si ocurre un error durante la predicción del modelo.
        Example:
            >>> model_type = ModelType.PALM_DETECTION
            >>> predictions, annotations = self._generate_sync_predictions(
            ...     image=cv2.imread("palm_image.jpg"),
            ...     name="palm_image.jpg",
            ...     model_type=model_type
            ... )
            >>> print(f"Se detectaron {len(predictions)} objetos")
            >>> print(f"Anotaciones COCO: {annotations}")
        Notes:
            - Las predicciones se filtran automáticamente por confianza mínima usando PALM_MODEL_MIN_CONFIDENCE.
            - La función utiliza diferentes configuraciones de modelo según el ModelType especificado.
            - Las categorías se generan automáticamente basadas en los nombres de clase del modelo.
            - El predictor utiliza configuraciones predefinidas para tamaño de imagen y filtros de superposición.
        """

        match model_type:
            case ModelType.PALM_DETECTION:
                logger.debug("Usando modelo de detección de palmas.")
                model_path = PALM_MODEL_PATH
                model_class_names = PALM_MODEL_CLASS_NAMES
                overlap_ratio_wh = OVERLAP_RATIO_WH
                min_ratio = PALM_MIN_RATIO
                nms_iou_threshold = PALM_NMS_IOU_THRESHOLD
                containerment_threshold = PALM_CONTAINERMENT_THRESHOLD
                confidence = PALM_CONFIDENCE
            case ModelType.RPW_DETECTION:
                logger.debug("Usando modelo de detección de palmas con RPW.")
                model_path = RPW_MODEL_PATH
                model_class_names = RPW_MODEL_CLASS_NAMES
                overlap_ratio_wh = OVERLAP_RATIO_WH
                min_ratio = RPW_MIN_RATIO
                nms_iou_threshold = RPW_NMS_IOU_THRESHOLD
                containerment_threshold = RPW_CONTAINERMENT_THRESHOLD
                confidence = RPW_CONFIDENCE
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")

        categories = [{"id": id, "name": name, "supercategory": ""} for id, name in model_class_names.items()]
        model_predictor = DetectionModelPredictor(
            model=model_path, target_img_size_wh=TARGET_IMG_SIZE_WH, overlap_ratio_wh=overlap_ratio_wh
        )
        # TODO: Medir tiempos + dimensiones de la imagen.
        predictions = (
            model_predictor.predict(image)
            .filter_by_square_ratio(min_ratio=min_ratio)
            .filter_by_nms(nms_iou_threshold, class_agnostic=True)
            .filter_by_containment(containerment_threshold, class_agnostic=True, check_confidence=False)
            .filter_by_confidence(confidence)
        )
        coco_annotations = predictions.as_coco_annotations(
            pic_name=name,
            categories=categories,
        )

        return predictions, coco_annotations
