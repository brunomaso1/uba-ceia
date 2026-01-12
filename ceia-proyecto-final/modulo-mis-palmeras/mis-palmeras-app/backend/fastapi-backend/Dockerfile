# Extraido de: https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
# WARNING: Hay que chequear la versión de cuda compatible con la GPU del servidor donde se desplegará.
# WARNING2: Si se utiliza Pytorch, hay que chequear la versión de PyTorch compatible con la versión de cuda elegida.
# FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Puerto
ENV PORT=8000

# UV setup
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin
# Dejar que uv gestione su propia instalacion de python
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=managed

# Install the project into `/app`
WORKDIR /app

# Install system dependencies
# NOTE: gdal-bin and libgdal-dev are required for GDAL Python bindings to work
# NOTE: libgl1, libglib2.0-0 is needed for opencv-python, consider using opencv-python-headless if you don't need GUI features
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the project's dependencies using the lockfile and settings
# NOTE: Had to mount the wheels directory to be able to install local wheels too.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=wheels,target=wheels \
    uv sync --locked --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# App
EXPOSE ${PORT}
# HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"]