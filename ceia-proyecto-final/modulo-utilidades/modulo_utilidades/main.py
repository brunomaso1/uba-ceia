from loguru import logger as LOGGER
from modulo_utilidades.config import settings as CONFIG

ENVIRONMENT = CONFIG.environment


def main():
    if ENVIRONMENT == "dev":
        LOGGER.debug("Modo de desarrollo activado.")
        try:
            LOGGER.debug("Iniciando la importación de módulos...")
            # Core modules (módulos utilizables por otros proyectos).
            from modulo_utilidades.core.labeling import (
                convertor_cordenadas_core,
                procesador_anotaciones_coco_dataset_core,
                procesador_geojson_kml_core,
            )

            # Modulos de desarrollo (módulos específicos de este proyecto).
            from modulo_utilidades.database_comunication.mongodb_client import mongodb
            from modulo_utilidades.s3_comunication.s3_client import s3client
            from modulo_utilidades.s3_comunication import procesador_s3
            from modulo_utilidades.labeling import (
                procesador_anotaciones_coco_dataset,
                procesador_anotaciones_cvat,
                procesador_anotaciones_mongodb,
                procesador_geojson_kml,
                procesador_recortes,
                visualizador_coco_dataset,
            )

            LOGGER.debug("Todos los módulos se importaron correctamente.")
        except ImportError as e:
            LOGGER.error(f"Error al importar un módulo: {e}")
        except Exception as e:
            LOGGER.error(f"Ocurrió un error inesperado: {e}")
    elif ENVIRONMENT == "prod":
        LOGGER.success("Modo de producción activado.")
        try:
            LOGGER.debug("Iniciando la importación de módulos...")
            # Core modules (módulos utilizables por otros proyectos).
            from modulo_utilidades.core.labeling import (
                convertor_cordenadas_core,
                procesador_anotaciones_coco_dataset_core,
                procesador_geojson_kml_core,
            )

            LOGGER.debug("Todos los módulos se importaron correctamente.")
        except ImportError as e:
            LOGGER.error(f"Error al importar un módulo: {e}")
        except Exception as e:
            LOGGER.error(f"Ocurrió un error inesperado: {e}")
    else:
        LOGGER.error(f"Entorno desconocido: {ENVIRONMENT}")


if __name__ == "__main__":
    main()
