import sys, os
from typing import Any, Dict, Optional

sys.path.append(os.path.abspath("../"))

from apps_utils.apps_utils import Singleton
from apps_config.settings import Config

import logging
import datetime

CONFIG = Config().config_data


class Logging(metaclass=Singleton):
    """Clase para manejar el logging de la aplicación."""

    logger = None

    def __init__(self, config: Optional[Dict[str, Any]] = CONFIG["logging"]) -> None:
        """Inicializa el logger y lo configura para escribir en un archivo."""
        # Configuración del logger
        logging.basicConfig(
            format=config["format"],
            level=logging.INFO if config["level"] == "INFO" else logging.DEBUG,
        )
        self.logger = logging.getLogger()

    def set_log_to_file(file_name: str) -> None:
        """Configura el logger para que escriba en un archivo con el nombre especificado."""
        logger = logging.getLogger()
        today = datetime.date.today()

        log_file = f"{file_name}_{today}.log"
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
