# Dependencias del sistema
from dataclasses import dataclass, field
from typing import Optional

# Dependencias locales
from modulo_utilidades.config import settings as CONFIG
from modulo_utilidades.config import MongoDBConfig

# Dependencias de terceros
from pymongo import MongoClient
from pymongo.database import Database
from loguru import logger as LOGGER

# Configuraciones
MONGODB_CONFIG: MongoDBConfig = CONFIG.mongodb


@dataclass
class MongoDB:
    """Clase para manejar la conexión a MongoDB usando dataclass."""

    mongodb_config: Optional[MongoDBConfig] = field(default_factory=lambda: MONGODB_CONFIG)
    db: Optional[Database] = field(init=False)

    def __post_init__(self) -> None:
        """Inicializa la conexión a MongoDB después de la creación del objeto."""
        try:
            self.connection_string = self.mongodb_config.connection_string
            self.database_name = self.mongodb_config.database
            self.client = MongoClient(self.connection_string)
            self.db = self.client.get_database(self.database_name)
        except Exception as e:
            self.client = None
            self.db = None
            raise Exception(f"Error al conectar a MongoDB: {e}")

    def close_connection(self) -> None:
        """Cierra la conexión a MongoDB."""
        if self.client:
            LOGGER.debug("Cerrando conexión a MongoDB.")
            self.client.close()
            self.client = None  # Resetear el cliente después de cerrar


# Instancia única (patrón Singleton a nivel de módulo)
mongo_instance = MongoDB()
mongodb = mongo_instance.db
