FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip (reliable in Docker)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency metadata first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (no venv inside container)
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
