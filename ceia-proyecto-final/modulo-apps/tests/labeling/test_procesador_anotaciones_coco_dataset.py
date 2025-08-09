from typing import Any
from modulo_apps.labeling.procesador_anotaciones_coco_dataset import convert_label, merge_labels, delete_label
import pytest
import copy


@pytest.fixture
def sample_coco_data() -> dict[str, Any]:
    """Fixture que proporciona un conjunto de datos COCO de ejemplo."""
    return {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "clase_a", "supercategory": "animal"},
            {"id": 2, "name": "clase_b", "supercategory": "animal"},
            {"id": 3, "name": "clase_c", "supercategory": "vehiculo"},
        ],
        "images": [{"id": 1, "width": 640, "height": 480, "file_name": "image1.jpg"}],
        "annotations": [{"id": 101, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}],
    }


# --- Casos de Prueba ---
# convert_label ->
def test_convert_label_successful(sample_coco_data: dict[str, Any]) -> None:
    """
    Prueba unitaria para la función `convert_label`.
    Esta prueba verifica que la función `convert_label` actualice correctamente
    las etiquetas en un conjunto de datos COCO. Se realizan las siguientes verificaciones:
    1. La etiqueta de origen (`origin_label`) ya no debe existir en los datos actualizados.
    2. La etiqueta de destino (`target_label`) debe estar presente en los datos actualizados.
    3. El resto de las categorías deben permanecer sin cambios.
    Args:
        sample_coco_data (dict[str, Any]): Un diccionario que representa un conjunto de datos
        COCO de ejemplo, utilizado como entrada para la prueba.
    Raises:
        AssertionError: Si alguna de las verificaciones falla.
    """
    data_to_test = copy.deepcopy(sample_coco_data)

    # Valores de entrada para la función
    origin_label = "clase_a"
    target_label = "nueva_clase"

    updated_data = convert_label(data_to_test, origin_label, target_label)

    # Se espera que la etiqueta "clase_a" ya no exista
    category_names = [cat["name"] for cat in updated_data["categories"]]
    assert origin_label not in category_names

    # Se espera que la etiqueta "nueva_clase" exista
    assert target_label in category_names

    # Se espera que el resto de las categorías permanezcan sin cambios
    assert {"id": 2, "name": "clase_b", "supercategory": "animal"} in updated_data["categories"]


def test_convert_label_origin_label_not_found(sample_coco_data: dict[str, Any]):
    """
    Prueba unitaria para la función `convert_label` que verifica el comportamiento
    cuando la etiqueta de origen no se encuentra en las anotaciones del dataset COCO.

    Args:
        sample_coco_data (dict[str, Any]): Diccionario que representa un conjunto de datos COCO de ejemplo.

    Escenario de prueba:
    - Se utiliza una etiqueta de origen inexistente (`etiqueta_inexistente`).
    - Se espera que la función `convert_label` lance una excepción `ValueError`
      con un mensaje indicando que la etiqueta de origen no se encuentra en las anotaciones.

    Resultado esperado:
    - La excepción `ValueError` es lanzada con el mensaje esperado.
    """

    data_to_test = copy.deepcopy(sample_coco_data)

    origin_label = "etiqueta_inexistente"
    target_label = "nueva_clase"

    with pytest.raises(
        ValueError, match=f"La etiqueta de origen '{origin_label}' no se encuentra en las anotaciones COCO."
    ):
        convert_label(data_to_test, origin_label, target_label)


def test_convert_label_no_annotations(sample_coco_data: dict[str, Any]):
    """
    Prueba unitaria para la función `convert_label` cuando no hay anotaciones en los datos de entrada.
    Este test verifica que la función `convert_label` maneje correctamente un caso en el que
    la lista de anotaciones está vacía. Se asegura de que:
    - La lista de anotaciones en los datos actualizados permanezca vacía.

    Args:
        sample_coco_data (dict[str, Any]): Datos de entrada en formato COCO que se utilizarán
        como base para la prueba. Este diccionario debe contener las claves "annotations" y "categories".

    Exceptions:
        AssertionError: Si alguna de las condiciones de la prueba no se cumple.
    """

    data_to_test = copy.deepcopy(sample_coco_data)
    data_to_test["annotations"] = []

    updated_data = convert_label(data_to_test, "clase_a", "nueva_clase")

    assert updated_data["annotations"] == []


# merge_labels ->
def test_merge_labels_successful(sample_coco_data: dict[str, Any]) -> None:
    """
    Prueba unitaria para la función `merge_labels`.
    Esta prueba verifica que la función `merge_labels` actualice correctamente
    las etiquetas en un conjunto de datos COCO. Se realizan las siguientes verificaciones:
    1. La etiqueta de origen (`origin_label`) ya no debe existir en los datos actualizados.
    2. La etiqueta de destino (`target_label`) debe estar presente en los datos actualizados.
    3. El resto de las categorías deben permanecer sin cambios.

    Args:
        sample_coco_data (dict[str, Any]): Un diccionario que representa un conjunto de datos
        COCO de ejemplo, utilizado como entrada para la prueba.

    Raises:
        AssertionError: Si alguna de las verificaciones falla.
    """
    data_to_test = copy.deepcopy(sample_coco_data)

    # Valores de entrada para la función
    origin_labels = ["clase_a", "clase_c"]
    target_label = "nueva_clase"

    updated_data = merge_labels(data_to_test, origin_labels, target_label)

    # Se espera que las etiquetas "clase_a" y "clase_c" ya no existan
    category_names = [cat["name"] for cat in updated_data["categories"]]
    assert origin_labels not in category_names

    # Se espera que la etiqueta "nueva_clase" exista
    assert target_label in category_names

    # Se espera que cambie el ID de la nueva etiqueta
    assert {"id": 0, "name": "clase_b", "supercategory": "animal"} in updated_data["categories"]


def test_merge_labels_origin_label_not_found(sample_coco_data: dict[str, Any]):
    """
    Prueba unitaria para la función `merge_labels` que verifica el comportamiento
    cuando la etiqueta de origen no se encuentra en las anotaciones del dataset COCO.

    Args:
        sample_coco_data (dict[str, Any]): Diccionario que representa un conjunto de datos COCO de ejemplo.

    Escenario de prueba:
    - Se utiliza una etiqueta de origen inexistente (`etiqueta_inexistente`).
    - Se espera que la función `merge_labels` lance una excepción `ValueError`
      con un mensaje indicando que la etiqueta de origen no se encuentra en las anotaciones.

    Resultado esperado:
    - La excepción `ValueError` es lanzada con el mensaje esperado.
    """

    data_to_test = copy.deepcopy(sample_coco_data)

    origin_label = ["clase_a", "etiqueta_inexistente"]
    target_label = "nueva_clase"

    with pytest.raises(
        ValueError, match=f"La etiqueta \'etiqueta_inexistente\' no se encuentra en las anotaciones COCO."
    ):
        merge_labels(data_to_test, origin_label, target_label)


def test_merge_labels_no_annotations(sample_coco_data: dict[str, Any]) -> None:
    """
    Prueba unitaria para la función `merge_labels` cuando no hay anotaciones en los datos de entrada.
    Este test verifica que la función `merge_labels` maneje correctamente un caso en el que
    la lista de anotaciones está vacía. Se asegura de que:
    - La lista de anotaciones en los datos actualizados permanezca vacía.

    Args:
        sample_coco_data (dict[str, Any]): Datos de entrada en formato COCO que se utilizarán
        como base para la prueba. Este diccionario debe contener las claves "annotations" y "categories".

    Exceptions:
        AssertionError: Si alguna de las condiciones de la prueba no se cumple.
    """

    data_to_test = copy.deepcopy(sample_coco_data)
    data_to_test["annotations"] = []

    updated_data = merge_labels(data_to_test, "clase_a", "nueva_clase")

    assert updated_data["annotations"] == []


# delete_label ->
def test_delete_label_successful(sample_coco_data: dict[str, Any]) -> None:
    """
    Prueba unitaria para la función `delete_label`.
    Esta prueba verifica que la función `delete_label` elimine correctamente
    una etiqueta de un conjunto de datos COCO. Se realizan las siguientes verificaciones:
    1. La etiqueta especificada debe ser eliminada de las categorías.
    2. Las anotaciones que contenían la etiqueta eliminada deben ser removidas.

    Args:
        sample_coco_data (dict[str, Any]): Un diccionario que representa un conjunto de datos
        COCO de ejemplo, utilizado como entrada para la prueba.

    Raises:
        AssertionError: Si alguna de las verificaciones falla.
    """
    data_to_test = copy.deepcopy(sample_coco_data)

    # Valores de entrada para la función
    label_to_delete = "clase_a"

    updated_data = delete_label(data_to_test, label_to_delete)

    # Se espera que la etiqueta "clase_a" ya no exista
    category_names = [cat["name"] for cat in updated_data["categories"]]
    assert label_to_delete not in category_names

    # Se espera que las anotaciones con la etiqueta eliminada ya no existan
    for annotation in updated_data["annotations"]:
        assert annotation["category_id"] != label_to_delete


def test_delete_label_not_found(sample_coco_data: dict[str, Any]) -> None:
    """
    Prueba unitaria para la función `delete_label` que verifica el comportamiento
    cuando la etiqueta a eliminar no se encuentra en las anotaciones del dataset COCO.

    Args:
        sample_coco_data (dict[str, Any]): Diccionario que representa un conjunto de datos COCO de ejemplo.

    Escenario de prueba:
    - Se utiliza una etiqueta inexistente (`etiqueta_inexistente`).
    - Se espera que la función `delete_label` lance una excepción `ValueError`
      con un mensaje indicando que la etiqueta no se encuentra en las anotaciones.

    Resultado esperado:
    - La excepción `ValueError` es lanzada con el mensaje esperado.
    """

    data_to_test = copy.deepcopy(sample_coco_data)

    label_to_delete = "etiqueta_inexistente"

    with pytest.raises(
        ValueError, match=f"La etiqueta 'etiqueta_inexistente' no se encuentra en las anotaciones COCO."
    ):
        delete_label(data_to_test, label_to_delete)


def test_delete_label_no_annotations(sample_coco_data: dict[str, Any]) -> None:
    """
    Prueba unitaria para la función `delete_label` cuando no hay anotaciones en los datos de entrada.
    Este test verifica que la función `delete_label` maneje correctamente un caso en el que
    la lista de anotaciones está vacía. Se asegura de que:
    - La lista de anotaciones en los datos actualizados permanezca vacía.

    Args:
        sample_coco_data (dict[str, Any]): Datos de entrada en formato COCO que se utilizarán
        como base para la prueba. Este diccionario debe contener las claves "annotations" y "categories".

    Exceptions:
        AssertionError: Si alguna de las condiciones de la prueba no se cumple.
    """

    data_to_test = copy.deepcopy(sample_coco_data)
    data_to_test["annotations"] = []

    updated_data = delete_label(data_to_test, "clase_a")

    assert updated_data["annotations"] == []


def test_delete_label_one_annotation(sample_coco_data: dict[str, Any]) -> None:
    """
    Prueba unitaria para la función `delete_label` cuando hay una sola anotación.
    Este test verifica que la función `delete_label` maneje correctamente un caso en el que
    hay una sola anotación que contiene la etiqueta a eliminar. Se asegura de que:
    - La lista de anotaciones en los datos actualizados permanezca vacía.

    Args:
        sample_coco_data (dict[str, Any]): Datos de entrada en formato COCO que se utilizarán
        como base para la prueba. Este diccionario debe contener las claves "annotations" y "categories".

    Exceptions:
        AssertionError: Si alguna de las condiciones de la prueba no se cumple.
    """

    data_to_test = copy.deepcopy(sample_coco_data)
    data_to_test["annotations"] = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}]

    updated_data = delete_label(data_to_test, "clase_a")

    assert updated_data["annotations"] == []
