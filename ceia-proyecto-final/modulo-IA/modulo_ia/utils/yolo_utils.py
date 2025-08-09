import datetime, yaml

from pathlib import Path
from typing import Any

from loguru import logger as LOGGER
from matplotlib import pyplot as plt
from modulo_ia.config import config as CONFIG

from ultralytics.engine.results import Results
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg

from deprecated import deprecated


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


@deprecated(
    version="1.0.0",
    reason="Esta función está obsoleta y será eliminada en futuras versiones. Usa create_coco_annotations_from_detections en su lugar.",
)
def create_coco_annotations_from_yolo_result(
    results: list[dict[str, Any]],
    pic_name: str,
) -> dict[str, Any]:
    """
    Convierte los resultados de detección de objetos en formato YOLO a anotaciones en formato COCO.

    Esta función toma una imagen y sus resultados de detección, y genera un diccionario
    que sigue la estructura del formato COCO. Se espera que los resultados contengan
    información sobre las cajas delimitadoras, categorías y confianza de las detecciones.
    Solamente se procesa una imagen a la vez.

    Args:
        image (np.ndarray): Imagen en formato numpy array sobre la que se realizó la detección.
        results (dict[str, Any]): Resultados de la detección de objetos, típicamente una lista de predicciones YOLO.
        pic_name (str): Nombre base del archivo de la imagen (sin extensión).

    Raises:
        ValueError: Si los resultados contienen detecciones para más de una imagen.

    Returns:
        dict[str, Any]: Diccionario con las anotaciones en formato COCO, incluyendo info, licenses, categories, images y annotations.
    """
    bboxes_result = results.boxes.cpu()
    names = results.names
    coco_annotations = {
        "info": CONFIG.coco_dataset.to_dict()["info"],
        "licenses": CONFIG.coco_dataset.to_dict()["licenses"],
        "categories": CONFIG.coco_dataset.to_dict()["categories"],
        "images": [],
        "annotations": [],
    }

    category_map = {cat["name"]: cat["id"] for cat in coco_annotations["categories"]}
    image_height, image_width = results.orig_shape

    image = {
        "id": 1,
        "width": image_width,
        "height": image_height,
        "file_name": f"{pic_name}.jpg",
        "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    coco_annotations["images"] = [image]

    if not results.boxes:
        LOGGER.warning("No se encontraron resultados de detección de objetos.")
        return coco_annotations

    annotations = []
    for index, bbox_result in enumerate(bboxes_result):
        id = index + 1
        category_name = names[bbox_result.cls.int().item()]
        category_id = category_map.get(category_name, None)
        if category_id is None:
            LOGGER.warning(f"Categoría '{category_name}' no encontrada en el mapa de categorías.")
            continue

        x_min, y_min, x_max, y_max = bbox_result.xyxy[0].numpy()
        ancho = x_max - x_min
        alto = y_max - y_min
        area = ancho * alto

        conf = bbox_result.conf[0].numpy()

        # Casteamos a float para evitar problemas de serialización
        conf = float(conf)
        x_min, y_min, ancho, alto = map(float, [x_min, y_min, ancho, alto])
        area = float(area)

        annotation = {
            "id": id,
            "image_id": image["id"],
            "category_id": category_id,
            "bbox": [x_min, y_min, ancho, alto],
            "area": area,
            "iscrowd": 0,
            "attributes": {
                "occluded": False,
                "rotation": 0.0,
            },
            "confidence": conf,
        }
        annotations.append(annotation)

    coco_annotations["annotations"] = annotations

    return coco_annotations


def get_yolo_training_dataloader(
    cfg_path: Path,
    cfg_data: Path,
    cfg_imgsz: int,
    cfg_batch: int,
    dataset_path: Path,
) -> InfiniteDataLoader:
    cfg_dict = get_cfg(overrides={"cfg": str(cfg_path)})
    cfg_dict.data = str(cfg_data)
    cfg_dict.imgsz = cfg_imgsz
    cfg_dict.batch = cfg_batch
    cfg_dict.mode = "train"

    # Cargar el YAML como diccionario
    with open(cfg_dict.data, "r") as f:
        data_dict = yaml.safe_load(f)
        if "channels" not in data_dict:
            data_dict["channels"] = 3  # RGB

    yolo_training_dataset = build_yolo_dataset(cfg_dict, img_path=dataset_path, batch=cfg_dict.batch, data=data_dict)
    yolo_training_dataloader = build_dataloader(yolo_training_dataset, batch=cfg_dict.batch, workers=cfg_dict.workers)
    return yolo_training_dataloader


def plot_yolo_augmentations(dataloader, class_names, max_batches=1, ncols=3):
    """
    Visualiza las imágenes y sus correspondientes cajas delimitadoras (bounding boxes)
    y clases a partir de un dataloader.
    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader que proporciona lotes de datos
            con las claves "bboxes", "cls", "img" y "batch_idx".
            - "bboxes": Tensor de forma (n_boxes, 4) que contiene las coordenadas de las
              cajas delimitadoras en formato (x_center, y_center, width, height).
            - "cls": Tensor de forma (n_boxes, 1) que contiene los índices de las clases
              correspondientes a cada caja delimitadora.
            - "img": Tensor de forma (batch_size, 3, H, W) que contiene las imágenes del lote.
            - "batch_idx": Tensor de forma (n_boxes, 1) que indica a qué imagen pertenece
              cada caja delimitadora.
        class_names (list): Lista de nombres de las clases, donde el índice corresponde
            al identificador de la clase.
        max_batches (int, opcional): Número máximo de lotes a visualizar. Por defecto es 1.
        ncols (int, opcional): Número de columnas en la cuadrícula de subplots. Por defecto es 3.
    Returns:
        None: La función no retorna ningún valor, pero muestra las imágenes con sus
        correspondientes cajas delimitadoras y nombres de clases utilizando matplotlib.
    Notas:
        - Las imágenes se normalizan automáticamente al rango [0, 1] si no están ya en ese rango.
        - Las cajas delimitadoras se dibujan en rojo, y los nombres de las clases se muestran
          como texto sobre las cajas.
        - Si el número de imágenes en un lote es menor que el número de subplots, los subplots
          restantes se desactivan.
    """
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
            # Cambiar el formato de (C, H, W) a (H, W, C)
            # Convertir el tensor a NumPy y normalizar a [0, 1] si no lo está
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalizar si es necesario

            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Image {j + 1}")

            h, w, _ = img.shape

            # 2. Filtrar las cajas delimitadoras y clases para la imagen actual
            current_img_bboxes = bboxes[batch_idx.flatten() == j]
            current_img_cls = cls[batch_idx.flatten() == j]

            # 3. Dibujar las cajas delimitadoras
            for bbox, cls_id in zip(current_img_bboxes, current_img_cls):
                # Convertir de formato (x_center, y_center, width, height) a (x1, y1, x2, y2)
                # y escalar a las dimensiones de la imagen
                x_center, y_center, bbox_w, bbox_h = bbox.cpu().numpy()
                x1 = int((x_center - bbox_w / 2) * w)
                y1 = int((y_center - bbox_h / 2) * h)
                width = int(bbox_w * w)
                height = int(bbox_h * h)

                # Crear y añadir el rectángulo
                rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor="red", facecolor="none")
                ax.add_patch(rect)

                # Obtener el nombre de la clase y añadir el texto
                class_name = class_names[int(cls_id.item())]
                ax.text(
                    x1,
                    y1 - 5,
                    class_name,
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="red", alpha=0.5, edgecolor="none"),
                )

        # Eliminar los ejes no utilizados si el número de imágenes es menor que el número de subplots
        for j in range(cant_imgs, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.show()
