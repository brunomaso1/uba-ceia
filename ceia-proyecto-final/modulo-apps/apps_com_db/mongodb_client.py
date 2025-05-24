import sys, os
from typing import Any, Dict, Optional

sys.path.append(os.path.abspath("../"))

from apps_utils.apps_utils import Singleton
from pymongo import MongoClient
from apps_config.settings import Config

CONFIG = Config().config_data


class MongoDB(metaclass=Singleton):
    connection_string = None
    database_name = None
    client = None
    db = None

    def __init__(self, config: Optional[Dict[str, Any]] = CONFIG["mongodb"]) -> None:
        """Inicializa la conexión a MongoDB."""
        try:
            self.connection_string = config["connection_string"]
            self.database_name = config["database"]
            self.client = MongoClient(self.connection_string)
            self.db = self.client.get_database(self.database_name)
        except Exception as e:
            self.client = None
            self.db = None
            raise Exception(f"Error al conectar a MongoDB: {e}")
