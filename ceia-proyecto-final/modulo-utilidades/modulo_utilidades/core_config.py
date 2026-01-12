# Dependencias del sistema
import os
from pathlib import Path
from typing import Any

# Dependencias de terceros
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings

# Configuration.
PROJECT_DIR = Path(__file__).parent.parent.resolve()
ROOT_DIR = PROJECT_DIR / "modulo_utilidades"
OPENCV_IO_MAX_IMAGE_PIXELS = 50000 * 50000  # Para imágenes grandes, ej: barrio3Ombues_20180801_dji_pc_3cm.jpg
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(OPENCV_IO_MAX_IMAGE_PIXELS)


class COCOInfo(BaseModel):
    description: str = "Conjunto de imágenes para la detección del picudo rojo"
    url: str = "https://picudo-rojo.org"
    version: str = "1.0"
    year: int = 2025
    contributor: str = "Intendencia de Montevideo"
    date_created: str = "2025/01/01"


class COCOLicense(BaseModel):
    id: int
    name: str
    url: str


class COCOCategory(BaseModel):
    id: int
    name: str
    supercategory: str = ""


class COCODatasetConfig(BaseModel):
    info: COCOInfo = Field(default_factory=COCOInfo)
    licenses: list[COCOLicense] = Field(
        default_factory=lambda: [
            COCOLicense(id=1, name="CC BY-NC-SA 4.0", url="https://creativecommons.org/licenses/by-nc-sa/4.0/")
        ]
    )
    categories: list[COCOCategory] = Field(
        default_factory=lambda: [
            COCOCategory(id=0, name="palmera-sana"),
            COCOCategory(id=1, name="palmera-infectada"),
            COCOCategory(id=2, name="palmera-muerta"),
            COCOCategory(id=3, name="palmera-exterminada"),
        ]
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return {
            "info": self.info.model_dump(),
            "licenses": [lic.model_dump() for lic in self.licenses],
            "categories": [cat.model_dump() for cat in self.categories],
        }


class GeoreferencingConfig(BaseModel):
    sistema_referencia: str = "WGS84"
    proyeccion: str = "UTM 21S"
    codigo_epsg: str = "EPSG:32721"

class FoldersCoreConfig(BaseModel):
    """Configuración de carpetas del núcleo de la aplicación."""

    project_dir: Path = PROJECT_DIR
    root_dir: Path = ROOT_DIR
    download_folder: Path = PROJECT_DIR / "downloads"

    @computed_field
    def download_coco_annotations_folder(self) -> Path:
        return self.download_folder / "coco_annotations"
    
    @computed_field
    def download_kmls_folder(self) -> Path:
        return self.download_folder / "kmls"
    
    @computed_field
    def download_geojson_folder(self) -> Path:
        return self.download_folder / "geojson"


class CoreSettings(BaseSettings):
    """Configuración base del núcleo de la aplicación."""

    folders: FoldersCoreConfig = FoldersCoreConfig()
    coco_dataset: COCODatasetConfig = Field(default_factory=COCODatasetConfig)
    georeferenciacion: GeoreferencingConfig = Field(default_factory=GeoreferencingConfig)


core_settings = CoreSettings()
