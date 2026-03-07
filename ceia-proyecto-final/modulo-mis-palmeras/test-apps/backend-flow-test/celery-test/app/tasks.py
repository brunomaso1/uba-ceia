# System imports
from time import sleep

# Local imports
from .main import app

# Third-party imports


@app.task(name="app.tasks.add")
def add(x, y):
    # Simulate a time-consuming task
    sleep(5)
    return x + y
