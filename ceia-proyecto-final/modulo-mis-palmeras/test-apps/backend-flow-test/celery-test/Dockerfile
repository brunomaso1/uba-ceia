FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv to manage dependencies from pyproject/lock files.
RUN pip install --no-cache-dir uv

# Copy dependency metadata first for better layer caching.
COPY pyproject.toml uv.lock ./

# Create a local virtual environment and install project dependencies.
RUN uv sync --frozen --no-dev

# Copy source code after dependencies are installed.
COPY app ./app

CMD ["uv", "run", "celery", "-A", "app.main", "worker", "--loglevel=info"]
