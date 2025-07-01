from pathlib import Path
import shutil
from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import typer

import fiftyone as fo

from loguru import logger as LOGGER
import yaml
from modulo_ia.config import config as CONFIG
from modulo_apps.database_comunication.mongodb_client import mongodb as DB

import modulo_apps.s3_comunication.procesador_s3 as ProcesadorS3
import modulo_apps.labeling.procesador_anotaciones_mongodb as ProcesadorCocoDataset
import modulo_apps.labeling.procesador_recortes as ProcesadorRecortes
from modulo_ia.utils.types import DatasetFormat

RAW_DATA_FOLDER = CONFIG.folders.raw_data_folder
EXTERNAL_DATA_FOLDER = CONFIG.folders.external_data_folder
INTERIM_DATA_FOLDER = CONFIG.folders.interim_data_folder
PROCESSED_DATA_FOLDER = CONFIG.folders.processed_data_folder

FULL_DATASET_NAME = CONFIG.names.detection_dataset_name
FULL_DATASET_VERSION = CONFIG.versions.detection_dataset_version

PARTIAL_DATASET_NAME = CONFIG.names.partial_dataset_name
PARTIAL_DATASET_VERSION = CONFIG.versions.detection_dataset_version

CUTOUTS_DATASET_NAME = CONFIG.names.cutouts_dataset_name
CUTOUTS_DATASET_VERSION = CONFIG.versions.cutouts_dataset_version

RANDOM_SEED = CONFIG.seed

app = typer.Typer()


@app.command()
def download_full_raw_dataset(
    for_patches: bool = False,
    output_folder: Path = RAW_DATA_FOLDER,
    with_annotations: bool = True,
    annotations_field_name: str = "cvat",
    annotations_output_filename: str = None,
) -> List[str]:
    """
    Descarga el conjunto de datos bruto completo desde la base de datos y el almacenamiento S3.
    La descarga se realiza en formato COCO, ya sea de imágenes o parches.

    Args:
        for_patches (bool, optional): Indica si se deben descargar parches en lugar de imágenes.
            Por defecto es False.
        folder_path (Optional[Path], optional): Ruta de la carpeta donde se almacenará el conjunto
            de datos descargado. Si no se proporciona, se utiliza la carpeta predeterminada
            configurada en RAW_DATA_FOLDER.
        with_annotations (bool, optional): Indica si se deben descargar las anotaciones junto con
            las imágenes o parches. Por defecto es True.
        annotations_field_name (str, optional): Nombre del campo en la base de datos que contiene
            las anotaciones. Por defecto es "cvat".
        annotations_output_filename (str, optional): Nombre del archivo donde se guardarán las
            anotaciones descargadas. Si no se proporciona, se utiliza "labels.json" en la carpeta
            destino.

    Returns:
        List[str]: Lista de nombres de imágenes o parches descargados, dependiendo del valor de
        for_patches.
    """
    output_folder /= f"{FULL_DATASET_NAME}_{FULL_DATASET_VERSION}"
    output_folder.mkdir(parents=True, exist_ok=True)
    annotations_output_filename = (
        output_folder / "labels.json" if annotations_output_filename is None else annotations_output_filename
    )

    data_folder_path = output_folder / "data"
    data_folder_path.mkdir(parents=True, exist_ok=True)
    if not for_patches:
        images_names = ProcesadorCocoDataset.list_images_w_ann_from_mongodb()
        ProcesadorS3.download_images_from_minio(images_names, data_folder_path)
    else:
        patch_names = ProcesadorCocoDataset.list_patches_w_ann_from_mongodb()
        ProcesadorS3.download_patches_from_minio(patch_names, data_folder_path)

    if with_annotations:
        if for_patches:
            ProcesadorCocoDataset.download_annotations_as_coco_from_mongodb(
                patches_names=patch_names,
                field_name=annotations_field_name,
                output_filename=annotations_output_filename,
            )
        else:
            ProcesadorCocoDataset.download_annotations_as_coco_from_mongodb(
                field_name=annotations_field_name,
                images_names=images_names,
                output_filename=annotations_output_filename,
            )

    return patch_names if for_patches else images_names


@app.command()
def download_partial_raw_dataset(
    patches_names: List[str] = typer.Option(None, help="Lista de nombres de parches"),
    images_names: List[str] = typer.Option(None, help="Lista de nombres de imágenes"),
    output_folder: Path = RAW_DATA_FOLDER,
    with_annotations: bool = True,
    annotations_field_name: str = "cvat",
    annotations_output_filename: str = None,
) -> List[str]:
    """
    Descarga un conjunto de datos parcial desde la base de datos y el almacenamiento S3.
    La descarga se realiza en formato COCO, ya sea de imágenes o parches.

    Args:
        images_names (List[str], optional): Lista de nombres de imágenes a descargar. Por defecto es None.
        patches_names (List[str], optional): Lista de nombres de parches a descargar. Por defecto es None.
        folder_path (Optional[Path], optional): Ruta de la carpeta donde se almacenará el conjunto
            de datos descargado. Si no se proporciona, se utiliza la carpeta predeterminada
            configurada en RAW_DATA_FOLDER.
        with_annotations (bool, optional): Indica si se deben descargar las anotaciones junto con
            las imágenes o parches. Por defecto es True.
        annotations_field_name (str, optional): Nombre del campo en la base de datos que contiene
            las anotaciones. Por defecto es "cvat".
        annotations_output_filename (str, optional): Nombre del archivo donde se guardarán las
            anotaciones descargadas. Si no se proporciona, se utiliza "labels.json" en la carpeta
            destino.

    Raises:
        ValueError: Si no se proporciona una lista de nombres de imágenes o parches, o si se
            proporcionan ambas listas al mismo tiempo.

    Returns:
        List[str]: Lista de nombres de imágenes o parches descargados, dependiendo de los argumentos
        proporcionados.
    """
    if bool(patches_names is None) == bool(images_names is None):  # xor
        raise ValueError("Se debe proporcionar una lista de nombres de parches o imágenes.")
    output_folder /= f"{FULL_DATASET_NAME}_{FULL_DATASET_VERSION}"
    output_folder.mkdir(parents=True, exist_ok=True)
    annotations_output_filename = (
        output_folder / "labels.json" if annotations_output_filename is None else annotations_output_filename
    )

    data_folder_path = output_folder / "data"
    data_folder_path.mkdir(parents=True, exist_ok=True)
    if not patches_names:
        ProcesadorS3.download_images_from_minio(images_names, data_folder_path)
    else:
        ProcesadorS3.download_patches_from_minio(patches_names, data_folder_path)

    if with_annotations:
        if patches_names:
            ProcesadorCocoDataset.download_annotations_as_coco_from_mongodb(
                patches_names=patches_names,
                field_name=annotations_field_name,
                output_filename=annotations_output_filename,
            )
        else:
            ProcesadorCocoDataset.download_annotations_as_coco_from_mongodb(
                field_name=annotations_field_name,
                images_names=images_names,
                output_filename=annotations_output_filename,
            )

    return patches_names if patches_names else images_names


@app.command()
def convert_dataset_to_model_format(
    dataset_path: Path = RAW_DATA_FOLDER / f"{FULL_DATASET_NAME}_{FULL_DATASET_VERSION}",
    dataset_name: str = FULL_DATASET_NAME,
    output_format: DatasetFormat = DatasetFormat.YOLO,
    output_dir: Optional[Path] = None,
    delete_previous_data: bool = True,
    clean: bool = False,
):
    # Validar que exista el directorio de entrada
    if not dataset_path.exists():
        LOGGER.error(f"El directorio de entrada no existe: {dataset_path}")
        raise typer.Exit(1)

    if not output_dir:
        LOGGER.debug(
            f"No se especificó un directorio de salida, se usará el predeterminado para el fomato {output_format.value.upper()}"
        )
        output_dir = INTERIM_DATA_FOLDER / f"{FULL_DATASET_NAME}_{FULL_DATASET_VERSION}_{output_format.value}"

    # Eliminar datos anteriores si se especifica
    if delete_previous_data and output_dir.exists():
        LOGGER.info(f"Eliminando datos anteriores en el directorio de salida: {output_dir}")
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            LOGGER.info(f"Datos anteriores eliminados: {output_dir}")
        except Exception as e:
            LOGGER.warning(f"Error al eliminar los datos anteriores: {e}")
            raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Convirtiendo dataset a formato {output_format.value.upper()}")
    LOGGER.info(f"Entrada: {dataset_path}")
    LOGGER.info(f"Salida: {output_dir}")

    if output_format == DatasetFormat.YOLO:
        _convert_to_yolo(dataset_path, dataset_name, output_dir)
    elif output_format == DatasetFormat.HUGGINGFACE:
        _convert_to_huggingface(dataset_path, output_dir)
    else:
        LOGGER.error(f"Formato {output_format} no implementado aún")
        raise typer.Exit(1)
    LOGGER.success(f"Conversión completada a formato {output_format.value.upper()}")

    if clean:
        LOGGER.info(f"Limpiando directorio de entrada: {dataset_path}")
        try:
            shutil.rmtree(dataset_path, ignore_errors=True)
            LOGGER.info(f"Directorio de entrada limpiado: {dataset_path}")
        except Exception as e:
            LOGGER.warning(f"Error al limpiar el directorio de entrada: {e}")
            raise typer.Exit(1)


def _convert_to_yolo(input_path: Path, dataset_name: str, output_path: Path) -> None:
    """Convierte el dataset al formato YOLO"""
    LOGGER.info("Implementando conversión a YOLO...")
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        dataset_dir=input_path,
        overwrite=True,
        name=dataset_name,
    )
    dataset.export(
        export_dir=str(output_path),
        dataset_type=fo.types.YOLOv5Dataset,
        overwrite=True,
        split="full",
    )
    LOGGER.info(f"Dataset exportado a {output_path} en formato YOLO")


def _convert_to_huggingface(input_path: Path, output_path: Path):
    raise NotImplementedError("Conversión a HuggingFace pendiente de implementar")


@app.command()
def split_dataset(
    dataset_path: Path = INTERIM_DATA_FOLDER / f"{FULL_DATASET_NAME}_{FULL_DATASET_VERSION}_{DatasetFormat.YOLO.value}",
    input_format: DatasetFormat = DatasetFormat.YOLO,
    output_dir: Optional[Path] = None,
    delete_previous_data: bool = True,
    clean: bool = False,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> None:
    """
    Divide un dataset en formato YOLO en conjuntos de entrenamiento, validación y prueba,
    moviendo los archivos y eliminando la carpeta 'full'.

    Args:
        dataset_path (Path): Ruta al directorio principal del dataset YOLO.
        ratios (tuple): Tupla con los ratios para (entrenamiento, validación, prueba).
                       Por defecto es (0.7, 0.2, 0.1).
    """
    # Validar que exista el directorio de entrada
    if not dataset_path.exists():
        LOGGER.error(f"El directorio de entrada no existe: {dataset_path}")
        raise typer.Exit(1)

    if not output_dir:
        LOGGER.debug(
            f"No se especificó un directorio de salida, se usará el predeterminado para el fomato {input_format.value.upper()}"
        )
        output_dir = PROCESSED_DATA_FOLDER / f"{FULL_DATASET_NAME}_{FULL_DATASET_VERSION}_{input_format.value}"

    # Eliminar datos anteriores si se especifica
    if delete_previous_data and output_dir.exists():
        LOGGER.info(f"Eliminando datos anteriores en el directorio de salida: {output_dir}")
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
            LOGGER.info(f"Datos anteriores eliminados: {output_dir}")
        except Exception as e:
            LOGGER.warning(f"Error al eliminar los datos anteriores: {e}")
            raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_split, val_split, test_split = ratios
    if not (0 < train_split < 1 and 0 <= val_split < 1 and 0 <= test_split < 1):
        raise ValueError("Los ratios deben estar entre 0 y 1 y sumar 1.")
    ruta_imagenes_full = dataset_path / "images" / "full"
    ruta_etiquetas_full = dataset_path / "labels" / "full"

    if not ruta_imagenes_full.is_dir() or not ruta_etiquetas_full.is_dir():
        raise FileNotFoundError(
            f"No se encontraron las carpetas 'full' en {dataset_path / 'images'} o {dataset_path / 'labels'}. "
            "Asegúrate de que el dataset esté en el formato correcto."
        )

    nombres_imagenes = [f.name for f in ruta_imagenes_full.iterdir() if f.is_file()]
    nombres_etiquetas = [f.name for f in ruta_etiquetas_full.iterdir() if f.is_file()]

    # Asegurarse de que haya correspondencia entre imágenes y etiquetas (mismo nombre base)
    nombres_imagenes_base = {f.stem for f in ruta_imagenes_full.iterdir() if f.is_file()}
    nombres_etiquetas_base = {f.stem for f in ruta_etiquetas_full.iterdir() if f.is_file()}
    nombres_comunes = list(nombres_imagenes_base.intersection(nombres_etiquetas_base))

    if not nombres_comunes:
        LOGGER.warning(
            "No se encontraron pares de imágenes y etiquetas con nombres coincidentes en las carpetas 'full'."
        )
        return None

    # Dividir los nombres de los archivos en conjuntos de entrenamiento, validación y prueba
    train_nombres, temp_nombres = train_test_split(
        nombres_comunes, test_size=(val_split + test_split), random_state=RANDOM_SEED
    )
    if test_split > 0:
        val_nombres, test_nombres = train_test_split(
            temp_nombres, test_size=test_split / (val_split + test_split), random_state=RANDOM_SEED
        )
    else:
        val_nombres = temp_nombres
        test_nombres = []

    conjuntos = (
        {"train": train_nombres, "val": val_nombres, "test": test_nombres}
        if test_split > 0
        else {"train": train_nombres, "val": val_nombres}
    )
    subcarpetas = ["images", "labels"]

    # Crear las carpetas para los conjuntos divididos si no existen
    for subcarpeta in subcarpetas:
        for conjunto in conjuntos:
            (output_dir / subcarpeta / conjunto).mkdir(parents=True, exist_ok=True)

    # Copiar los archivos a sus respectivas carpetas
    total_files = sum(len(nombres) for nombres in conjuntos.values())
    with tqdm(total=total_files, desc="Moviendo archivos", unit="archivo") as pbar:
        for conjunto, nombres in conjuntos.items():
            for nombre_base in nombres:
                # Obtener el nombre de la imagen con su extensión
                resultados_glob_img = list(ruta_imagenes_full.glob(f"{nombre_base}.*"))
                nombre_imagen = ""
                if resultados_glob_img:
                    nombre_imagen = resultados_glob_img[0].name

                nombre_etiqueta = nombre_base + ".txt"

                ruta_imagen_origen = ruta_imagenes_full / nombre_imagen
                ruta_etiqueta_origen = ruta_etiquetas_full / nombre_etiqueta

                ruta_imagen_destino = output_dir / "images" / conjunto / nombre_imagen
                ruta_etiqueta_destino = output_dir / "labels" / conjunto / nombre_etiqueta

                if ruta_imagen_origen.exists():
                    shutil.copy2(str(ruta_imagen_origen), str(ruta_imagen_destino))
                if ruta_etiqueta_origen.exists():
                    shutil.copy2(str(ruta_etiqueta_origen), str(ruta_etiqueta_destino))
                pbar.update(1)

    # Eliminar las carpetas "full"
    if clean:
        try:
            shutil.rmtree(ruta_imagenes_full)
            LOGGER.debug(f"Carpeta eliminada: {ruta_imagenes_full}")
        except OSError as e:
            LOGGER.warning(f"Error al eliminar {ruta_imagenes_full}: {e}")

        try:
            shutil.rmtree(ruta_etiquetas_full)
            LOGGER.debug(f"Carpeta eliminada: {ruta_etiquetas_full}")
        except OSError as e:
            LOGGER.warning(f"Error al eliminar {ruta_etiquetas_full}: {e}")

    # Actualizar el archivo dataset.yaml
    ruta_yaml = dataset_path / "dataset.yaml"
    if ruta_yaml.exists():
        with open(ruta_yaml, "r") as f:
            data = yaml.safe_load(f)
        data["train"] = str(Path("images") / "train")
        data["val"] = str(Path("images") / "val")
        if "test" in data and test_split > 0:
            data["test"] = str(Path("images") / "test")
        if "full" in data:
            del data["full"]
        with open(ruta_yaml, "w") as f:
            yaml.dump(data, f)
        LOGGER.debug(
            f"Archivo {ruta_yaml} actualizado con las rutas de train, val y test, y se eliminó la clave 'path' si era necesario."
        )
    else:
        LOGGER.warning(f"Advertencia: No se encontró el archivo {ruta_yaml}. Deberás actualizar las rutas manualmente.")

    LOGGER.info("División del dataset YOLO completada (archivos copiados).")
    return None


@app.command()
def download_cutouts_raw_dataset(
    output_folder: Path = RAW_DATA_FOLDER,
    with_annotations: bool = True,
    annotations_field_name: str = "cvat",
    annotations_output_filename: str = None,
) -> List[str]:
    raise NotImplementedError()


if __name__ == "__main__":
    app()
