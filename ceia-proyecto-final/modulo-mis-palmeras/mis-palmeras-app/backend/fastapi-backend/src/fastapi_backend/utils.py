from fastapi_backend.dependencies.in_memory_store_api import InMemoryStore
from fastapi_backend.schemas.data_types.store_data_type import StoreDataType


def fetch_store_entry_with_checks(store_api: InMemoryStore, image_id: int) -> StoreDataType:
    store_entry = store_api.get(image_id)
    if store_entry is None:
        raise ValueError("No se encontró la imagen con el ID proporcionado.")
    if store_entry.image is None:
        raise ValueError("No se encontró la imagen para el ID proporcionado. Ya la subió?")
    if store_entry.jgw is None:
        raise ValueError("No se encontró el JGW para la imagen con el ID proporcionado. Ya lo subió?")

    return store_entry
