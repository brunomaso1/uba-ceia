from loguru import logger
from modulo_apps.config import config as MODULO_APPS_CONFIG
from modulo_ia.config import config as MODULO_IA_CONFIG

def main():
    try:
        logger.debug("Iniciando la importación de módulos...")
        logger.debug(f"Ambiente actual: {MODULO_IA_CONFIG.environment}")
        from modulo_ia import dataset
        from modulo_ia import features

        from modulo_ia.utils import gpu
        from modulo_ia.utils import metrics
        from modulo_ia.utils import types
        from modulo_ia.utils import yolo_utils

        from modulo_ia.modeling import predict
        logger.debug("Todos los módulos se importaron correctamente.")
    except ImportError as e:
        logger.error(f"Error al importar un módulo: {e}")
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    main()
