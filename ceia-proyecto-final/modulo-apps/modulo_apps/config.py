import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Any, Tuple, Optional

OPENCV_IO_MAX_IMAGE_PIXELS = 50000 * 50000  # Para imágenes grandes, ej: barrio3Ombues_20180801_dji_pc_3cm.jpg
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(OPENCV_IO_MAX_IMAGE_PIXELS)

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent

# Construir las rutas con respecto al directorio raíz.
env_dev_path = PROJECT_DIR / ".env.dev"
env_prod_path = PROJECT_DIR / ".env.prod"

logger.info(f"Directorio de configuración raíz: {ROOT_DIR}")

if env_dev_path.exists():
    load_dotenv(env_dev_path)
    logger.info("Variables de entorno cargadas desde .env.dev.")
elif env_prod_path.exists():
    load_dotenv(env_prod_path)
    logger.info("Variables de entorno cargadas desde .env.prod.")
else:
    logger.info("No se encontraron archivos .env. Cargando variables desde el entorno del sistema.")


@dataclass
class MongoDBConfig:
    database: str
    host: str
    port: str
    user: str
    password: str

    @property
    def connection_string(self) -> str:
        return f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class CVATConfig:
    url: str
    user: str
    password: str
    export_format: str = "COCO 1.0"
    task_export_path: str = "annotations\\instances_default.json"
    job_export_path: str = "annotations\\instances_default.json"


@dataclass
class MinioPaths:
    images: str = "imagenes"
    metadata: str = "imagenes_metadatos"
    patches: str = "patches"
    cutouts: str = "recortes"
    cutouts_metadata: str = "recortes_metadatos"


@dataclass
class MinioConfig:
    bucket: str
    endpoint_url: str
    access_key: str
    secret_key: str
    region: str = "nl-ams"
    paths: MinioPaths = field(default_factory=MinioPaths)


@dataclass
class FoldersConfig:
    download_folder: Path = Path("downloads")

    def __post_init__(self):
        self.download_images_folder: Path = self.download_folder / "images"
        self.download_patches_folder: Path = self.download_folder / "patches"
        self.download_temp_folder: Path = self.download_folder / "temp"
        self.download_jobs_folder: Path = self.download_folder / "jobs"
        self.download_tasks_folder: Path = self.download_folder / "tasks"
        self.download_coco_annotations_folder: Path = self.download_folder / "coco_annotations"
        self.download_google_maps_folder: Path = self.download_folder / "google_maps"
        self.download_kmls_folder: Path = self.download_folder / "kmls"
        self.download_cutouts_folder: Path = self.download_folder / "cutouts"
        self.download_cutouts_metadata_folder: Path = self.download_folder / "cutouts_metadata"
        self.download_geojson_folder: Path = self.download_folder / "geojson"
        self.download_jgw_folder: Path = self.download_folder / "jgw"


@dataclass
class URLsConfig:
    main_page: str = "https://gis.montevideo.gub.uy/pmapper/map.phtml?&config=default&me=548000,6130000,596000,6162000"
    toc: str = "https://intgis.montevideo.gub.uy/pmapper/incphp/xajax/x_toc.php?"
    generate_zip: str = (
        "https://intgis.montevideo.gub.uy/sit/php/common/datos/generar_zip2.php?nom_jpg=/inetpub/wwwroot/sit/mapserv/data/fotos_dron/{id}&tipo=jpg"
    )
    download_zip: str = "https://intgis.montevideo.gub.uy/sit/tmp/{id}.zip"
    js: str = "https://intgis.montevideo.gub.uy/pmapper/config/default/custom.js"


@dataclass
class CommonHeaders:
    user_agent: str = "Mozilla/5.0"


@dataclass
class TOCHeaders:
    user_agent: str = "Mozilla/5.0"
    referer: str = "https://gis.montevideo.gub.uy/pmapper/map.phtml?&config=default&me=548000,6130000,596000,6162000"
    x_requested_with: str = "XMLHttpRequest"
    content_type: str = "application/x-www-form-urlencoded"


@dataclass
class HeadersConfig:
    common: CommonHeaders = field(default_factory=CommonHeaders)
    toc: TOCHeaders = field(default_factory=TOCHeaders)


@dataclass
class TOCRequestBody:
    dummy: str = "dummy"


@dataclass
class RequestBodyConfig:
    toc: TOCRequestBody = field(default_factory=TOCRequestBody)


@dataclass
class PatchesConfig:
    tile_size: Tuple[int, int] = (4096, 4096)
    over_lap: int = 400
    purge_white_images: bool = True


@dataclass
class COCOInfo:
    description: str = "Conjunto de imágenes para la detección del picudo rojo"
    url: str = "https://picudo-rojo.org"
    version: str = "1.0"
    year: int = 2025
    contributor: str = "Intendencia de Montevideo"
    date_created: str = "2025/01/01"


@dataclass
class COCOLicense:
    id: int
    name: str
    url: str


@dataclass
class COCOCategory:
    id: int
    name: str
    supercategory: str = ""


@dataclass
class COCODatasetConfig:
    info: COCOInfo = field(default_factory=COCOInfo)
    licenses: List[COCOLicense] = field(
        default_factory=lambda: [
            COCOLicense(id=1, name="CC BY-NC-SA 4.0", url="https://creativecommons.org/licenses/by-nc-sa/4.0/")
        ]
    )
    # TODO: Cambiar el nombre a categorias de CVAT...
    categories: List[COCOCategory] = field(
        default_factory=lambda: [
            COCOCategory(id=0, name="palmera-sana"),
            COCOCategory(id=1, name="palmera-infectada"),
            COCOCategory(id=3, name="palmera-muerta"),
            COCOCategory(id=5, name="palmera-exterminada"),
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración del dataset COCO a un diccionario"""
        return {
            "info": asdict(self.info),
            "licenses": [asdict(lic) for lic in self.licenses],
            "categories": [asdict(cat) for cat in self.categories],
        }


@dataclass
class LayoutParserDrawBox:
    box_width: int = 10
    box_alpha: int = 0
    color_map: Dict[str, str] = field(
        default_factory=lambda: {
            "palmera-sana": "green",
            "palmera-inf-leve": "yellow",
            "palmera-inf-grave": "orange",
            "palmera-muerta": "red",
            "palmera": "blue",
        }
    )
    show_element_id: bool = True


@dataclass
class LayoutParserConfig:
    draw_box: LayoutParserDrawBox = field(default_factory=LayoutParserDrawBox)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración de LayoutParser a un diccionario"""
        return {
            "draw_box": asdict(self.draw_box),
        }


@dataclass
class OpenCVDrawBox:
    box_width: int = 5
    box_alpha: float = 0.5
    color_map: Dict[int, List[int]] = field(
        default_factory=lambda: {
            1: [0, 255, 0],
            2: [255, 255, 0],
            3: [255, 165, 0],
            4: [255, 0, 0],
            5: [0, 0, 255],
        }
    )
    font_scale: float = 0.5
    font_thickness: int = 2


@dataclass
class OpenCVDrawConfig:
    draw_box: OpenCVDrawBox = field(default_factory=OpenCVDrawBox)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración de OpenCV Draw a un diccionario"""
        return {
            "draw_box": asdict(self.draw_box),
        }


@dataclass
class GoogleMapsCategory:
    id: int
    name: str
    supercategory: str = ""


@dataclass
class GoogleMapsConfig:
    mid: str = "1yCQ986yfEy6SmUXEREmedCEptZEA_h0"
    base_url: str = "https://www.google.com/maps/d/u/0/kml"
    categories: List[GoogleMapsCategory] = field(
        default_factory=lambda: [GoogleMapsCategory(id=1, name="palmera-google-maps")]
    )


@dataclass
class GeoreferencingConfig:
    sistema_referencia: str = "WGS84"
    proyeccion: str = "UTM 21S"
    codigo_epsg: str = "EPSG:32721"


@dataclass
class BBoxSizeConfig:
    width: int = 10
    height: int = 10


class Config:
    """Clase principal de configuración del sistema"""

    def __init__(self):
        # Configuración general
        self.environment = os.getenv("ENVIRONMENT", "dev")
        self.seed = 42

        self.mongodb = self._get_mongodb_config()
        self.cvat = self._get_cvat_config()
        self.minio = self._get_minio_config()
        self.folders = FoldersConfig()
        self.urls = URLsConfig()
        self.headers = HeadersConfig()
        self.request_body = RequestBodyConfig()
        self.patches = PatchesConfig()
        self.coco_dataset = COCODatasetConfig()
        self.layoutparser = LayoutParserConfig()
        self.opencv_draw = OpenCVDrawConfig()
        self.google_maps = GoogleMapsConfig()
        self.georeferenciacion = GeoreferencingConfig()
        self.bbox_size = BBoxSizeConfig()

    def _get_mongodb_config(self) -> MongoDBConfig:
        """Obtiene la configuración de MongoDB desde variables de entorno"""
        return MongoDBConfig(
            database=os.getenv("MONGODB_INITDB_DATABASE", ""),
            host=os.getenv("MONGODB_SERVER_HOST", ""),
            port=os.getenv("MONGODB_SERVER_PORT", ""),
            user=os.getenv("MONGODB_USER", ""),
            password=os.getenv("MONGODB_PASSWORD", ""),
        )

    def _get_cvat_config(self) -> CVATConfig:
        """Obtiene la configuración de CVAT desde variables de entorno"""
        return CVATConfig(
            url=os.getenv("CVAT_URL", ""), user=os.getenv("CVAT_USER", ""), password=os.getenv("CVAT_PASSWORD", "")
        )

    def _get_minio_config(self) -> MinioConfig:
        """Obtiene la configuración de MinIO desde variables de entorno"""
        return MinioConfig(
            bucket=os.getenv("MINIO_BUCKET", ""),
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL", ""),
            access_key=os.getenv("MINIO_ACCESS_KEY", ""),
            secret_key=os.getenv("MINIO_SECRET_KEY", ""),
        )

    def get_headers_dict(self) -> Dict[str, Dict[str, str]]:
        """Convierte las configuraciones de headers a diccionarios para uso con requests"""
        return {
            "common": {"User-Agent": self.headers.common.user_agent},
            "toc": {
                "User-Agent": self.headers.toc.user_agent,
                "Referer": self.headers.toc.referer,
                "X-Requested-With": self.headers.toc.x_requested_with,
                "Content-Type": self.headers.toc.content_type,
            },
        }

    def get_request_body_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convierte las configuraciones de request body a diccionarios"""
        return {"toc": {"dummy": self.request_body.toc.dummy}}


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Instancia global de configuración
config = Config()
