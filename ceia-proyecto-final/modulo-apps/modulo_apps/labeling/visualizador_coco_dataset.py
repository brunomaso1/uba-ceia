import os

from matplotlib import pyplot as plt
import numpy as np

from pathlib import Path
from typing import Any, Optional
import layoutparser as lp

import cv2 as cv
from pycocotools.coco import COCO

from modulo_apps.config import config as CONFIG

import modulo_apps.labeling.procesador_anotaciones_coco_dataset as CocoDatasetUtils


def _load_coco_annotations(annotations: list[dict[str, Any]], coco: Any = None):
    """Carga las anotaciones en formato COCO y las convierte en un objeto Layout de LayoutParser.

    Args:
        annotations (list): Lista de anotaciones en formato COCO.
        coco (COCO, opcional): Objeto COCO que contiene información adicional sobre las categorías. Por defecto es None.

    Returns:
        Layout: Objeto Layout de LayoutParser que contiene las anotaciones procesadas.
    """
    layout = lp.Layout()

    for ele in annotations:

        x, y, w, h = ele["bbox"]

        layout.append(
            lp.TextBlock(
                block=lp.Rectangle(x, y, w + x, h + y),
                type=(ele["category_id"] if coco is None else coco.cats[ele["category_id"]]["name"]),
                id=ele["id"],
            )
        )

    return layout


def show_anotated_image(
    image_path: Optional[Path] = None,
    image: Optional[np.ndarray] = None,
    image_name: Optional[str] = None,
    coco_annotations: Optional[dict[str, Any]] = None,
    annotation_path: Optional[Path] = None,
    fig_size: Optional[tuple[int, int]] = None,
    use_layoutparser: bool = False,
) -> None:
    """Muestra una imagen anotada con las anotaciones proporcionadas en formato COCO.

    Args:
        image_path (str): Ruta al archivo de imagen.
        coco_annotations (dict, optional): Diccionario con las anotaciones en formato COCO. Defaults to None.
        annotation_path (str, optional): Ruta al archivo JSON con las anotaciones en formato COCO. Defaults to None.
        fig_size (tuple, optional): Tamaño de la figura para la visualización. Defaults to None.
        use_layoutparser (bool, optional): Si se debe usar layoutparser para dibujar las anotaciones. Defaults to False.

    Raises:
        FileNotFoundError: Si el archivo de imagen no existe.
        ValueError: Si se proporcionan tanto `coco_annotations` como `annotation_path`.
        ValueError: Si no se encuentra el archivo de anotaciones especificado.
        ValueError: Si no se encuentra el ID de la imagen en las anotaciones.

    Example:
        >>> anottation_path = download_image_annotations_as_coco("8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm", "cvat")
        >>> print(f"Archivo generado en: {anottation_path}")
        >>> image_path = download_image_from_minio("8deOctubreyCentenario-EspLibreLarranaga_20190828_dji_pc_5cm")
        >>> print(f"Imagen descargada en: {image_path}")
        >>> show_anotated_image(image_path, annotation_path=anottation_path, fig_size=(20, 20))
        >>> show_anotated_image(
        ...     image_path=image_path,
        ...     annotation_path=anottation_path,
        ...     fig_size=(10, 10),
        ...     use_layoutparser=False
        ... )
    """
    if bool(image_path is None) == bool(image is None):  # xor
        raise ValueError("Se debe proporcionar image_path o image.")
    if bool(coco_annotations is None) == bool(annotation_path is None):  # xor
        raise ValueError("Se debe proporcionar coco_annotations o annotation_path.")
    if image and not image_name:
        raise ValueError("Se debe proporcionar image_name si se proporciona image.")
    if annotation_path and not annotation_path.exists():
        raise FileNotFoundError(f"El archivo de anotaciones {annotation_path} no existe.")
    if image_path and not image_path.exists():
        raise FileNotFoundError(f"El archivo de imagen {image_path} no existe.")

    if annotation_path:
        coco_annotations = CocoDatasetUtils.load_annotations_from_path(annotation_path)
    if image_path:
        image = cv.imread(str(image_path))

    # Buscamos el id de la imagen en las anotaciones
    image_name = image_path.name if image_path else image_name
    image_id = CocoDatasetUtils.get_image_id_from_annotations(image_name, coco_annotations)
    if use_layoutparser:
        coco = COCO(annotation_path)
        annotations = coco.loadAnns(coco.getAnnIds([image_id]))

        layout = _load_coco_annotations(annotations, coco)

        if fig_size:
            plt.figure(figsize=fig_size)

        layoutparser_draw_box_config = CONFIG.layoutparser.draw_box
        viz = lp.draw_box(
            image,
            layout,
            box_width=layoutparser_draw_box_config["box_width"],
            box_alpha=layoutparser_draw_box_config["box_alpha"],
            color_map=layoutparser_draw_box_config["color_map"],
        )
        display(viz)
    else:
        if fig_size:
            plt.figure(figsize=fig_size)
        else:
            plt.figure()
        
        drawbox_config = CONFIG.opencv_draw.to_dict()['draw_box']
        color_map = {k: tuple(v) for k, v in drawbox_config["color_map"].items()}
        category_map = {cat["id"]: cat["name"] for cat in coco_annotations["categories"]}

        # Dibujar las anotaciones en la imagen
        for annotation in coco_annotations["annotations"]:
            if annotation["image_id"] == image_id:
                x, y, w, h = annotation["bbox"]
                color = color_map.get(annotation["category_id"], (0, 255, 0))
                cv.rectangle(
                    image, (int(x), int(y)), (int(x + w), int(y + h)), color, CONFIG.opencv_draw.draw_box.box_width
                )
                cv.putText(
                    image,
                    str(category_map[annotation["category_id"]]),
                    (int(x), int(y) - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    CONFIG.opencv_draw.draw_box.font_scale,
                    color,
                    CONFIG.opencv_draw.draw_box.font_thickness,
                )

        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Imagen: {os.path.basename(image_path)}")
        plt.show()