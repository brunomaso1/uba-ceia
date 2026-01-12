# Dependencias del sistema
import os
from pathlib import Path

# Dependencias de terceros
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Configuration.
PROJECT_DIR = Path(__file__).parent.parent.resolve()
ROOT_DIR = PROJECT_DIR / "modulo_ia"
OPENCV_IO_MAX_IMAGE_PIXELS = 50000 * 50000  # Para imágenes grandes, ej: barrio3Ombues_20180801_dji_pc_3cm.jpg
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(OPENCV_IO_MAX_IMAGE_PIXELS)


class FoldersCoreConfig(BaseModel):
    """Configuración de carpetas del núcleo de la aplicación."""

    project_dir: Path = PROJECT_DIR
    root_dir: Path = ROOT_DIR
    download_folder: Path = PROJECT_DIR / "downloads"


class CoreSettings(BaseSettings):

    folders: FoldersCoreConfig = FoldersCoreConfig()


core_settings = CoreSettings()
