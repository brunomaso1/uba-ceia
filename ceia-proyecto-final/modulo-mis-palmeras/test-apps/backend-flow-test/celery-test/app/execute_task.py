# System imports

# Local imports
from .tasks import add

# Third-party imports


if __name__ == "__main__":
    t = add.delay(4, 6)
    print("Task submitted, waiting for result...")

    # Get id of the task
    print(f"Result id: {t.id}")

    # Check if the task is ready
    print(f"Is task ready? {t.ready()}")

    # Check the status of the task
    print(f"Task status: {t.status}")

    # Check the result of the task without blocking
    print(f"Task result (non-blocking): {t.result}")

    # Get the result of the task (blocking)
    print(f"Task result: {t.get()}")

    # Get the result of the task with timeout (blocking)
    try:
        print(f"Task result with timeout: {t.get(timeout=15)}")
    except Exception as e:
        print(f"Error getting task result with timeout: {e}")

    # Siempre hay que llamar a get() o forget() para limpiar el resultado del backend
    t.forget()
