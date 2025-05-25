from typing import Any, Dict, Optional, Tuple

import numpy as np

    
def convert_bbox_patch_to_image(patch_bbox: list, x_start: int, y_start: int) -> list:
    """Convierte las coordenadas de un bounding box (bbox) de un parche a coordenadas de la imagen
    (bbox_local -> bbox_image).

    En el formato COCO, los bounding boxes se representan como [x_min, y_min, width, height].
    Este método transforma las coordenadas locales de un parche a las coordenadas globales
    de la imagen original aplicando una traslación basada en los valores de desplazamiento
    proporcionados (x_start, y_start).

    Precondición: El parche no debe tener ninguna transformación afín.

    Args:
        patch_bbox (list): Bounding box del parche en formato [x_min, y_min, width, height].
        x_start (int): Coordenada x inicial del parche en la imagen original.
        y_start (int): Coordenada y inicial del parche en la imagen original.

    Returns:
        list: Bounding box transformado a coordenadas de la imagen en formato [x_global, y_global, width, height].
    """
    x_min, y_min, width, height = patch_bbox
    x_image = x_min + x_start
    y_image = y_min + y_start
    return [x_image, y_image, width, height]


def convert_bbox_image_to_patch(
    bbox_image: list, x_start: int, y_start: int, patch_width: int, patch_height: int
) -> Optional[list]:
    """Convierte las coordenadas de un bounding box (bbox) de una imagen a coordenadas locales de un parche.

    Este método transforma las coordenadas globales de un bounding box en una imagen
    a coordenadas locales dentro de un parche específico, siempre y cuando el bounding box
    esté completamente contenido dentro del parche.

    Args:
    bbox_global (list): Bounding box global en formato [x_min, y_min, width, height].
    x_start (int): Coordenada x inicial del parche en la imagen original.
    y_start (int): Coordenada y inicial del parche en la imagen original.
    patch_width (int): Ancho del parche.
    patch_height (int): Alto del parche.

    Returns:
    list: Bounding box transformado a coordenadas locales del parche en formato [x_local, y_local, width, height].
        Devuelve None si el bounding box no está completamente contenido dentro del parche.
    """
    x_min, y_min, w, h = bbox_image

    # Verificar que el bbox global esté dentro del parche
    if (
        x_min < x_start
        or y_min < y_start
        or x_min + w > x_start + patch_width
        or y_min + h > y_start + patch_height
    ):
        return None  # Fuera de los límites del parche

    # Transformar a coordenadas locales
    xl = x_min - x_start
    yl = y_min - y_start
    return [xl, yl, w, h]


def get_bbox_center(bbox: list) -> Tuple[float, float]:
    """Calcula el centro de un bounding box (bbox) en formato COCO.

    Args:
        bbox (list): Bounding box en formato [x_min, y_min, width, height].

    Returns:
        tuple: Coordenadas del centro del bounding box (x_center, y_center).
    """
    x_min, y_min, width, height = bbox
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return (x_center, y_center)


def convert_bbox_image_to_world(bbox: list, jgw_data: Dict[str, Any]) -> Dict[str, tuple]:
    # TODO: Definir el tipo jgw_data
    """Convierte un bounding box de coordenadas de imagen a coordenadas del mundo.

    Este método transforma las coordenadas de un bounding box definido en el sistema
    de coordenadas de la imagen a coordenadas en el sistema del mundo, utilizando
    los parámetros de transformación proporcionados.

    Args:
    bbox (list): Bounding box en coordenadas de la imagen en formato [x_min, y_min, width, height].
    jgw_data (Dict[str, Any]): Diccionario con los parámetros de transformación del archivo JGW.
                    Debe contener las claves:
                    - "x_pixel_size": Tamaño del píxel en X.
                    - "y_rotation": Rotación en Y.
                    - "x_rotation": Rotación en X.
                    - "y_pixel_size": Tamaño del píxel en Y.
                    - "x_origin": Origen en X.
                    - "y_origin": Origen en Y.

    Returns:
    Dict[str, tuple]: Bounding box en coordenadas del mundo. Cada clave representa un vértice
                del bounding box ("tl", "tr", "br", "bl") y su valor es una tupla (X, Y).
    """
    A = jgw_data["x_pixel_size"]
    D = jgw_data["y_rotation"]
    B = jgw_data["x_rotation"]
    E = jgw_data["y_pixel_size"]
    C = jgw_data["x_origin"]
    F = jgw_data["y_origin"]

    x0, y0, w, h = bbox
    # Definir los 4 vértices en píxeles
    corners = {
        "tl": (x0, y0),
        "tr": (x0 + w, y0),
        "br": (x0 + w, y0 + h),
        "bl": (x0, y0 + h),
    }

    world_bbox = {}
    for name, (px, py) in corners.items():
        X = A * px + B * py + C
        Y = D * px + E * py + F
        world_bbox[name] = (X, Y)

    return world_bbox


def convert_bbox_world_to_image(world_bbox: list, jgw_data: Dict[str, Any]) -> list:
    """Convierte un bounding box en coordenadas del mundo a coordenadas de la imagen.

    Este método transforma las coordenadas de un bounding box definido en el sistema de coordenadas del mundo
    a coordenadas en píxeles dentro de la imagen, utilizando los parámetros de transformación proporcionados.

    Args:
        world_bbox (list): Bounding box en coordenadas del mundo. Debe ser un diccionario con las claves:
                        "tl" (top-left), "tr" (top-right), "br" (bottom-right), "bl" (bottom-left),
                        donde cada clave tiene un valor de tupla (X, Y).
        jgw_data (dict): Diccionario con los parámetros de transformación del archivo JGW. Debe contener las claves:
                        - "x_pixel_size": Tamaño del píxel en X.
                        - "y_rotation": Rotación en Y.
                        - "x_rotation": Rotación en X.
                        - "y_pixel_size": Tamaño del píxel en Y.
                        - "x_origin": Origen en X.
                        - "y_origin": Origen en Y.

    Raises:
        ValueError: Si la transformación no es invertible (determinante ≈ 0).

    Returns:
        list: Bounding box en coordenadas de la imagen en formato [x_min, y_min, width, height].

    Ejemplo:
        world_bbox = {
            "tl": (100.0, 200.0),
            "tr": (150.0, 200.0),
            "br": (150.0, 250.0),
            "bl": (100.0, 250.0),
        }
        jgw_data = {
            "x_pixel_size": 0.5,
            "y_rotation": 0.0,
            "x_rotation": 0.0,
            "y_pixel_size": -0.5,
            "x_origin": 50.0,
            "y_origin": 300.0,
        }
        bbox_image = convert_bbox_world_to_image(world_bbox, jgw_data)
        # bbox_image = [100.0, 100.0, 50.0, 50.0]
    """
    # Leer parámetros
    A = jgw_data["x_pixel_size"]
    B = jgw_data["x_rotation"]
    D = jgw_data["y_rotation"]
    E = jgw_data["y_pixel_size"]
    C = jgw_data["x_origin"]
    F = jgw_data["y_origin"]

    # Montar matriz y calcular su inversa
    M = np.array([[A, B], [D, E]])
    det = A * E - B * D
    if abs(det) < 1e-12:
        raise ValueError("Transformación no invertible (det≈0).")
    M_inv = (1.0 / det) * np.array([[E, -B], [-D, A]])

    # Transformar cada vértice
    pixels = []
    for corner in ("tl", "tr", "br", "bl"):
        X, Y = world_bbox[corner]
        vec = np.dot(M_inv, np.array([X - C, Y - F]))
        pixels.append(vec)

    # Extraer coordenadas
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def convert_point_patch_to_image(point: Tuple[float, float], x_start: int, y_start: int) -> Tuple[float, float]:
    """Convierte las coordenadas de un punto de un parche a coordenadas de la imagen.

    Este método transforma las coordenadas locales de un punto dentro de un parche
    a coordenadas en la imagen original aplicando una traslación basada
    en los valores de desplazamiento proporcionados (x_start, y_start).

    Args:
        point (tuple): Coordenadas del punto en el parche en formato (x, y).
        x_start (int): Coordenada x inicial del parche en la imagen original.
        y_start (int): Coordenada y inicial del parche en la imagen original.

    Returns:
        tuple: Coordenadas del punto en la imagen en formato (x_image, y_image).
    """
    x_p, y_p = point
    return x_p + x_start, y_p + y_start


def convert_point_image_to_patch(
    point: Tuple[float, float], x_start: int, y_start: int, patch_width: int, patch_height: int
) -> tuple:
    """Convierte las coordenadas de un punto de la imagen a coordenadas locales de un parche.

    Este método transforma las coordenadas globales de un punto en la imagen
    a coordenadas locales dentro de un parche específico, siempre y cuando el punto
    esté contenido dentro del parche.

    Args:
        point (tuple): Coordenadas del punto en la imagen en formato (x, y).
        x_start (int): Coordenada x inicial del parche en la imagen original.
        y_start (int): Coordenada y inicial del parche en la imagen original.
        patch_width (int): Ancho del parche.
        patch_height (int): Alto del parche.

    Returns:
        tuple: Coordenadas locales del punto en el parche en formato (x_local, y_local).
            Devuelve None si el punto no está contenido dentro del parche.
    """
    x_p, y_p = point

    # Verificar que el punto esté dentro del parche
    if x_p < x_start or y_p < y_start or x_p > x_start + patch_width or y_p > y_start + patch_height:
        return None  # Fuera de los límites del parche

    # Transformar a coordenadas locales
    xl = x_p - x_start
    yl = y_p - y_start
    return xl, yl


def convert_point_image_to_world(point: Tuple[float, float], jgw_data: Dict[str, Any]) -> tuple:
    """
    Convierte un punto de coordenadas de la imagen a coordenadas del mundo.

    Este método transforma las coordenadas de un punto definido en el sistema
    de coordenadas de la imagen a coordenadas en el sistema del mundo, utilizando
    los parámetros de transformación proporcionados.

    Args:
        point (tuple): Coordenadas del punto en la imagen en formato (x, y).
        jgw_data (Dict[str, Any]): Diccionario con los parámetros de transformación del archivo JGW.
                                Debe contener las claves:
                                - "x_pixel_size": Tamaño del píxel en X.
                                - "y_rotation": Rotación en Y.
                                - "x_rotation": Rotación en X.
                                - "y_pixel_size": Tamaño del píxel en Y.
                                - "x_origin": Origen en X.
                                - "y_origin": Origen en Y.

    Returns:
        tuple: Coordenadas del punto en el sistema del mundo en formato (X, Y).
    """
    x_p, y_p = point
    A = jgw_data["x_pixel_size"]
    B = jgw_data["x_rotation"]
    D = jgw_data["y_rotation"]
    E = jgw_data["y_pixel_size"]
    C = jgw_data["x_origin"]
    F = jgw_data["y_origin"]

    X = A * x_p + B * y_p + C
    Y = D * x_p + E * y_p + F
    return X, Y


def convert_point_world_to_image(point: Tuple[float, float], jgw_data: Dict[str, Any]) -> Tuple[float, float]:
    """
    Convierte un punto de coordenadas del mundo a coordenadas de la imagen.

    Este método transforma las coordenadas de un punto definido en el sistema
    de coordenadas del mundo a coordenadas en píxeles dentro de la imagen,
    utilizando los parámetros de transformación proporcionados.

    Args:
        point (tuple): Coordenadas del punto en el sistema del mundo en formato (X, Y).
        jgw_data (Dict[str, Any]): Diccionario con los parámetros de transformación del archivo JGW.
                                Debe contener las claves:
                                - "x_pixel_size": Tamaño del píxel en X.
                                - "y_rotation": Rotación en Y.
                                - "x_rotation": Rotación en X.
                                - "y_pixel_size": Tamaño del píxel en Y.
                                - "x_origin": Origen en X.
                                - "y_origin": Origen en Y.

    Raises:
        ValueError: Si la transformación no es invertible (determinante ≈ 0).

    Returns:
        tuple: Coordenadas del punto en el sistema de la imagen en formato (x, y).
    """
    X, Y = point
    A, B = jgw_data["x_pixel_size"], jgw_data["x_rotation"]
    D, E = jgw_data["y_rotation"], jgw_data["y_pixel_size"]
    C, F = jgw_data["x_origin"], jgw_data["y_origin"]
    det = A * E - B * D
    if abs(det) < 1e-12:
        raise ValueError("Transformación no invertible (det≈0).")
    M_inv = (1.0 / det) * np.array([[E, -B], [-D, A]])
    vec = np.dot(M_inv, np.array([X - C, Y - F]))
    return float(vec[0]), float(vec[1])
