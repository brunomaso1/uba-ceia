# Dependencias del sistema
from torch import device
import datetime, yaml
from pathlib import Path
from typing import Any

# Dependencias locales
from modulo_ia.config import settings as CONFIG

# Dependencias de terceros
from loguru import logger as LOGGER
import pandas as pd
from matplotlib import pyplot as plt
from ultralytics.engine.results import Results
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils.metrics import DetMetrics
from ultralytics import YOLO
import typer

app = typer.Typer()


def filter_results_by_confidence(
    results: list[dict[str, Any]],
    min_confidence: float = 0.5,
) -> Results:
    """
    Filtra los resultados de detección de objetos para eliminar aquellas detecciones
    con una confianza inferior al umbral especificado.

    Args:
        results (list[dict[str, Any]]): Lista de resultados de detección, donde cada resultado
                                        es un diccionario que contiene información sobre las
                                        cajas delimitadoras, categorías y confianza.
        min_confidence (float): Umbral mínimo de confianza para filtrar las detecciones.
                                Por defecto es 0.5.

    Returns:
        Results: Resultados filtrados que cumplen con el umbral de confianza.
    """
    if len(results) > 1:
        raise ValueError(
            "Se detectó un resultado para varias imágenes. Asegúrate de que solo haya el resultado de una imagen."
        )
    result = results[0]

    # Filtrar los índices según el umbral
    filtered_indices = [i for i, conf in enumerate(result.boxes.conf) if conf.item() >= min_confidence]

    # Filtrar las cajas y otros atributos
    filtered_boxes = result.boxes[filtered_indices]

    # Crear un nuevo objeto Results con las mismas propiedades pero solo con las detecciones filtradas
    results_filtered = Results(
        orig_img=result.orig_img,
        path=result.path,
        names=result.names,
    )
    results_filtered.boxes = filtered_boxes
    results_filtered.orig_shape = result.orig_shape
    results_filtered.speed = result.speed
    results_filtered.save_dir = result.save_dir

    return [results_filtered]


def get_yolo_training_dataloader(
    cfg_path: Path,
    cfg_data: Path,
    cfg_imgsz: int,
    cfg_batch: int,
    dataset_path: Path,
    shuffle: bool = True,
) -> InfiniteDataLoader:
    with open(cfg_path, "r") as f:
        cfg_overrides = yaml.safe_load(f)

    cfg_overrides.update(
        {
            "data": str(cfg_data),
            "imgsz": cfg_imgsz,
            "batch": cfg_batch,
            "mode": "train",
        }
    )

    cfg_dict = get_cfg(overrides=cfg_overrides)

    # Cargar el YAML como diccionario
    with open(cfg_dict.data, "r") as f:
        data_dict = yaml.safe_load(f)
        if "channels" not in data_dict:
            data_dict["channels"] = 3  # RGB

    yolo_training_dataset = build_yolo_dataset(
        cfg_dict, rect=False, img_path=dataset_path, batch=cfg_dict.batch, data=data_dict
    )
    yolo_training_dataloader = build_dataloader(
        yolo_training_dataset, shuffle=shuffle, batch=cfg_dict.batch, workers=cfg_dict.workers
    )
    return yolo_training_dataloader


def plot_yolo_augmentations(dataloader, class_names, color_map=None, max_batches=1, ncols=3):
    """
    Visualiza imágenes de un DataLoader junto con sus cajas delimitadoras (bounding boxes) y etiquetas de clase.
        dataloader (torch.utils.data.DataLoader): Dataloader que produce lotes con las claves:
            - "bboxes": Tensor de forma (n_boxes, 4) con las coordenadas de las cajas en formato
              (x_center, y_center, width, height) normalizadas en [0, 1].
            - "cls": Tensor de forma (n_boxes, 1) con los índices de clase de cada caja.
            - "img": Tensor de forma (batch_size, 3, H, W) con las imágenes del lote.
            - "batch_idx": Tensor de forma (n_boxes, 1) que indica a qué imagen pertenece cada caja.
        class_names (list[str]): Lista con los nombres de clases; el índice corresponde al identificador de clase.
        color_map (dict[int | str, Any] | list[Any] | None, opcional): Mapa de colores para las clases. Puede ser:
            - dict que mapea id de clase (int) o nombre de clase (str) a un color aceptado por matplotlib,
            - list/tuple donde la posición i corresponde al color de la clase i,
            - Los colores pueden ser strings ('red', '#FF0000') o tuplas RGB(A). Si son enteros 0-255 se normalizan a 0-1.
            - None para usar el color por defecto (rojo).
        max_batches (int, opcional): Número máximo de lotes a visualizar. Por defecto 1.
        ncols (int, opcional): Número de columnas en la cuadrícula de subplots. Por defecto 3.
        should_shuffle (bool, opcional): Si es True, se intenta recrear el DataLoader con shuffle=True
            manteniendo el resto de parámetros (batch_size, num_workers, etc.). Si falla, se continúa sin barajar.
        None: Muestra las imágenes con sus cajas y etiquetas utilizando matplotlib.
        - Las imágenes se normalizan automáticamente al rango [0, 1] para su visualización.
        - Las cajas delimitadoras usan el color provisto en color_map cuando corresponde.
        - Si el número de imágenes del lote es menor que el número de subplots, los subplots restantes se desactivan.
    """

    def _to_mpl_color(c):
        """
            Normaliza una especificación de color al formato compatible con Matplotlib (RGB(A) con componentes en el rango [0, 1]).
            - Si c es una tupla o lista de longitud 3 o 4, se interpreta como RGB(A).
                - Si alguno de los tres primeros valores es > 1, se asume que están en el rango 0–255 y se escalan a 0–1.
                - En caso contrario, se dejan tal como están (se asume que ya están en 0–1).
                - Si hay un cuarto valor (alfa) y es numérico > 1, también se escala de 0–255 a 0–1; en otros casos se deja sin cambios.
            - Para cualquier otro tipo de entrada (por ejemplo, nombre de color, cadena hex, otros tipos), el valor se devuelve sin modificar.
            - Solo se normalizan secuencias tipo tupla/lista; otros iterables (p. ej., arrays de NumPy) no se transforman.
        Args:
                c (tuple | list | Any): Especificación de color: secuencia RGB o RGBA (longitud 3 o 4)
                        de enteros/flotantes, o cualquier color aceptado por Matplotlib (p. ej., 'red', '#FF007F').
                        Solo se normalizan las secuencias tipo tupla/lista.
        Returns:
                tuple | Any: Una tupla RGB o RGBA con componentes float en el rango [0, 1],
                        o la entrada original sin cambios si no es una tupla/lista de longitud 3 o 4.
        Examples:
            >>> _to_mpl_color((255, 0, 127))
            (1.0, 0.0, 0.4980392156862745)
            >>> _to_mpl_color((10, 20, 30, 128))
            (0.0392156862745098, 0.0784313725490196, 0.11764705882352941, 0.5019607843137255)
            >>> _to_mpl_color((0.1, 0.2, 0.3, 0.4))
            (0.1, 0.2, 0.3, 0.4)
            >>> _to_mpl_color("#ff007f")
            '#ff007f'
        """
        if isinstance(c, (tuple, list)):
            if len(c) in (3, 4):
                vals = list(c[:3])
                if any(v > 1 for v in vals):
                    vals = [v / 255.0 for v in vals]
                if len(c) == 4:
                    a = c[3]
                    a = a / 255.0 if isinstance(a, (int, float)) and a > 1 else a
                    return (*vals, a)
                return tuple(vals)
        return c

    def _get_color_for_class(cls_idx: int, cls_name: str):
        default_color = "red"
        if color_map is None:
            return default_color
        try:
            if isinstance(color_map, dict):
                if cls_idx in color_map:
                    return _to_mpl_color(color_map[cls_idx])
                if cls_name in color_map:
                    return _to_mpl_color(color_map[cls_name])
            elif isinstance(color_map, (list, tuple)):
                if 0 <= cls_idx < len(color_map):
                    return _to_mpl_color(color_map[cls_idx])
        except Exception:
            pass
        return default_color

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        bboxes = batch["bboxes"]  # (n_boxes, 4)
        cls = batch["cls"]  # (n_boxes, 1)
        imgs = batch["img"]  # (batch_size, 3, H, W)
        batch_idx = batch["batch_idx"]  # (n_boxes, 1)

        cant_imgs = len(imgs)
        nrows = (cant_imgs + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

        # Asegurarse de que axs sea siempre una lista para la iteración
        if nrows * ncols > 1:
            axs = axs.flatten()
        else:
            axs = [axs]

        for j, img_tensor in enumerate(imgs):
            ax = axs[j]

            # 1. Preparar la imagen para mostrarla con matplotlib
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            denom = img.max() - img.min()
            if denom > 0:
                img = (img - img.min()) / denom

            ax.imshow(img)
            ax.axis("off")

            h, w, _ = img.shape

            # 2. Filtrar las cajas delimitadoras y clases para la imagen actual
            current_img_bboxes = bboxes[batch_idx.flatten() == j]
            current_img_cls = cls[batch_idx.flatten() == j]

            # 3. Dibujar las cajas delimitadoras
            for bbox, cls_id in zip(current_img_bboxes, current_img_cls):
                # Convertir de formato (x_center, y_center, width, height) a (x1, y1, width, height)
                x_center, y_center, bbox_w, bbox_h = bbox.cpu().numpy()
                x1 = int((x_center - bbox_w / 2) * w)
                y1 = int((y_center - bbox_h / 2) * h)
                width = int(bbox_w * w)
                height = int(bbox_h * h)

                cls_idx = int(cls_id.item())
                class_name = class_names[cls_idx]
                color = _get_color_for_class(cls_idx, class_name)

                # Crear y añadir el rectángulo
                rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none")
                ax.add_patch(rect)

                # Añadir el texto con fondo del mismo color
                ax.text(
                    x1,
                    max(0, y1 - 10),
                    class_name,
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor=color, alpha=0.5, edgecolor="none"),
                )

        # Eliminar los ejes no utilizados si el número de imágenes es menor que el número de subplots
        for j in range(cant_imgs, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.show()


def ultralytics_confusion_to_df(metrics: DetMetrics, include_background: bool = True) -> pd.DataFrame:
    """
    Convierte la matriz de confusión de un objeto DetMetrics de Ultralytics en un DataFrame de Pandas.
    Agrega automáticamente la clase 'background' si corresponde.

    Args:
        metrics (DetMetrics): Resultado del método model.val(), que contiene metrics.confusion_matrix.
        include_background (bool): Si True, incluye la clase 'background' al final si hay una fila/col extra.

    Returns:
        pd.DataFrame: Matriz de confusión con índices y columnas nombradas según las clases.
    """
    cm = getattr(metrics, "confusion_matrix", None)
    if cm is None or getattr(cm, "matrix", None) is None:
        raise ValueError("El objeto metrics no contiene una matriz de confusión válida.")

    matrix = cm.matrix
    n_rows, n_cols = matrix.shape

    # Obtener nombres de clases del objeto Ultralytics
    names = list(getattr(cm, "names", {}).values()) if getattr(cm, "names", None) else []

    # Si la matriz incluye fondo (una fila/col extra)
    if include_background and len(names) + 1 == n_rows == n_cols:
        names.append("fondo")

    # Validar tamaño
    if len(names) != n_rows:
        raise ValueError(
            f"Cantidad de nombres ({len(names)}) no coincide con la forma de la matriz ({n_rows}, {n_cols})."
            f" Verificá si el modelo y el dataset tienen la misma cantidad de clases."
        )

    # Crear DataFrame
    df_cm = pd.DataFrame(matrix, index=names, columns=names)
    return df_cm


@app.command()
def convert_model_to_onnx(model_path):
    """
    Convierte un modelo de PyTorch a ONNX y lo guarda en la ruta especificada.

    Args:
        model_path (str | Path): La ruta del modelo de PyTorch a convertir.
        output_path (str | Path): La ruta donde se guardará el archivo ONNX resultante.

    Returns:
        None: Guarda el modelo convertido en la ruta especificada.
    """
    model = YOLO(model_path)

    # Originalmente, sin half=True. Esto implica que el modelo ocupe el doble de tamaño que el original, dado que por defecto
    # se exporta con precisión fp32, o sea: YOLO11x summary: 56,828,179 parameters -> 56.8M params × 4 bytes (fp32) ≈ 227 MB
    # Al poner half=True, se exporta con precisión fp16, lo que reduce el tamaño a la mitad: 56.8M params × 2 bytes (fp16) ≈ 113 MB, el tamaño original
    # del modelo. Para que funcione tiene que ser en GPU, o sea, device=0 (o el número del dispositivo CUDA que corresponda).
    model.export(format="onnx", half=True, device=0)


@app.command()
def test_converted_model(onnx_model_path):
    """
    Prueba un modelo ONNX convertido utilizando una imagen de prueba.

    Args:
        onnx_model_path (str | Path): La ruta del modelo ONNX a probar.
        test_image_path (str | Path): La ruta de la imagen de prueba.

    Returns:
        None: Muestra los resultados de la inferencia en la imagen de prueba.
    """
    onnx_model = YOLO(onnx_model_path, task="detect")
    results = onnx_model("https://ultralytics.com/images/bus.jpg")
    print(results)


def convert_and_test_model():
    LOGGER.info("Iniciando la conversión del modelo YOLO a ONNX...")
    models_folder = CONFIG.folders.models_folder
    model_name = "palm_detection_yolo11x_640_a3a50bd4646e4044bed83f02f8bb03f4.pt"

    model_path = models_folder / model_name
    assert model_path.exists(), f"No se encontró el modelo en la ruta especificada: {model_path}"

    output_path = model_path.with_suffix(".onnx")

    LOGGER.info(f"Convirtiendo el modelo '{model_path.name}' a ONNX...")
    convert_model_to_onnx(model_path)
    assert output_path.exists(), f"No se generó el archivo ONNX en la ruta esperada: {output_path}"

    LOGGER.info(f"Modelo convertido exitosamente y guardado en: {output_path}")
    test_converted_model(output_path)
    LOGGER.info("Prueba del modelo ONNX completada.")
