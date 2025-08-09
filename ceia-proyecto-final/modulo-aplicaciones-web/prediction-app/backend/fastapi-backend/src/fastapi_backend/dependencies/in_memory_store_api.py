from dataclasses import dataclass, field

from fastapi_backend.schemas.data_types.store_data_type import StoreDataType


@dataclass
class InMemoryStore:
    data: list[StoreDataType] = field(default_factory=list)

    def get(self, item_id: int) -> StoreDataType | None:
        """Get an item by its ID from the in-memory store."""
        if 0 < item_id <= len(self.data):
            return self.data[item_id - 1]  # Adjust for 0-based index
        return None

    def add(self, item: StoreDataType) -> int:
        """Add a new item to the in-memory store."""
        item.id = len(self.data) + 1  # Assign a new ID based on the current length
        self.data.append(item)
        return item.id

    def get_length(self) -> int:
        """Get the current length of the in-memory store."""
        return len(self.data)

    def update(self, item: StoreDataType) -> None:
        """Update an existing item in the in-memory store."""
        if 0 < item.id <= len(self.data):
            self.data[item.id - 1] = item  # Adjust for 0-based index
        else:
            raise ValueError("Item ID out of range for update.")


store_api = InMemoryStore()


# Singleton dependency to provide the in-memory store API
def get_store_api() -> InMemoryStore:
    """Dependency to get the in-memory store API."""
    return store_api
