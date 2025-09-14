import os

from matplotlib import pyplot as plt
import numpy as np

from pathlib import Path
from typing import Any, Optional
import layoutparser as lp

import cv2 as cv
from pycocotools.coco import COCO

from modulo_apps.config import config as CONFIG

DOWNLOAD_FOLDER = CONFIG.folders.download_folder
DOWNLOAD_IMAGES_FOLDER = CONFIG.folders.download_images_folder
DOWNLOAD_PATCHES_FOLDER = CONFIG.folders.download_patches_folder
DOWNLOAD_CUTOUTS_FOLDER = CONFIG.folders.download_cutouts_folder
DOWNLOAD_CUTOUTS_METADATA_FOLDER = CONFIG.folders.download_cutouts_metadata_folder

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


def show_annotated_image_path(
    image_path: Path,
    annotation_path: Path,
    image_name: str,
    fig_size: Optional[tuple[int, int]] = None,
    use_layoutparser: bool = False,
    should_download_annotated_image: bool = False,
) -> None:
    coco_annotations = CocoDatasetUtils.load_annotations_from_path(annotation_path)
    image = cv.imread(str(image_path), cv.IMREAD_COLOR_RGB)

    if image is None:
        raise FileNotFoundError(f"El archivo de imagen {image_path} no existe o no se puede leer.")

    show_annotated_image(
        image=image,
        coco_annotations=coco_annotations,
        image_name=image_name,
        fig_size=fig_size,
        use_layoutparser=use_layoutparser,
        should_download_annotated_image=should_download_annotated_image,
    )


def show_annotated_image(
    image: np.ndarray,
    coco_annotations: dict[str, Any],
    image_name: str,
    fig_size: Optional[tuple[int, int]] = None,
    use_layoutparser: bool = False,
    should_download_annotated_image: bool = False,
) -> None:
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

        DOWNLOAD_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
        if should_download_annotated_image:
            viz.save(DOWNLOAD_IMAGES_FOLDER / f"{image_name}_annotated.png")
    else:
        if fig_size:
            plt.figure(figsize=fig_size)
        else:
            plt.figure()

        drawbox_config = CONFIG.opencv_draw.to_dict()["draw_box"]
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

                # Obtener la confianza (si existe) y formatearla
                confidence = annotation.get("confidence")
                text = str(category_map[annotation["category_id"]])
                if confidence is not None:
                    # Formatea la confianza para mostrar solo dos decimales
                    text += f": {confidence:.2f}"

                cv.putText(
                    image,
                    text,
                    (int(x), int(y) - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    CONFIG.opencv_draw.draw_box.font_scale,
                    color,
                    CONFIG.opencv_draw.draw_box.font_thickness,
                )

        DOWNLOAD_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
        if should_download_annotated_image:
            cv.imwrite(DOWNLOAD_IMAGES_FOLDER / f"{image_name}_annotated.png", image)

        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Imagen: {image_name}")
        plt.show()