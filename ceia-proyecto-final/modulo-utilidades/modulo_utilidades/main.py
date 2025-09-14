from loguru import logger
from modulo_utilidades.config import config as CONFIG


def main():
    try:
        logger.debug("Iniciando la importación de módulos...")
        logger.debug(f"Ambiente actual: {CONFIG.environment}")
        from modulo_utilidades.database_comunication.mongodb_client import mongodb

        from modulo_utilidades.s3_comunication.s3_client import s3client
        from modulo_utilidades.s3_comunication import procesador_s3

        from modulo_utilidades.labeling import (
            convertor_cordenadas,
            procesador_anotaciones_coco_dataset,
            procesador_anotaciones_cvat,
            procesador_anotaciones_mongodb,
            procesador_geojson_kml,
            procesador_recortes,
            visualizador_coco_dataset
        )

        logger.debug("Todos los módulos se importaron correctamente.")
    except ImportError as e:
        logger.error(f"Error al importar un módulo: {e}")
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    main()
