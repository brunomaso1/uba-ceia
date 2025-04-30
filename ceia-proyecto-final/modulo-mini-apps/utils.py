import logging
import datetime
from pymongo import MongoClient


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MongoDB(metaclass=Singleton):
    connection_string = None
    database_name = None
    client = None
    db = None

    def __init__(
        self, mongodb_conection_string: str, mongodb_initdb_database: str
    ) -> None:
        """Inicializa la conexión a MongoDB."""
        try:
            self.connection_string = mongodb_conection_string
            self.database_name = mongodb_initdb_database
            self.client = MongoClient(self.connection_string)
            self.db = self.client.get_database(self.database_name)
        except Exception as e:
            self.client = None
            self.db = None
            raise Exception(f"Error al conectar a MongoDB: {e}")


def set_log_to_file(file_name: str) -> None:
    """Configura el logger para que escriba en un archivo con el nombre especificado."""
    logger = logging.getLogger()
    today = datetime.date.today()

    log_file = f"{file_name}_{today}.log"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
