from loguru import logger as LOGGER
from modulo_ia.config import settings as CONFIG

ENVIRONMENT = CONFIG.environment

def main():
    if ENVIRONMENT == "dev":
        LOGGER.debug("Modo de desarrollo activado")
        try:
            LOGGER.debug("Iniciando la importación de módulos...")
            # Core modules (módulos utilizables por otros proyectos).
            from modulo_ia.core.modeling import predict

            # Modulos de desarrollo (módulos específicos de este proyecto).
            from modulo_ia import dataset
            from modulo_ia import features
            from modulo_ia.utils import gpu
            from modulo_ia.utils import types
            from modulo_ia.utils import yolo_utils

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
            from modulo_ia.core.modeling import predict

            LOGGER.debug("Todos los módulos se importaron correctamente.")
        except ImportError as e:
            LOGGER.error(f"Error al importar un módulo: {e}")
        except Exception as e:
            LOGGER.error(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()
