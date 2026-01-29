FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    build-essential \
    curl \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libgtk-3-0 \
    libxss1 \
    libxtst6 \
    xdg-utils \
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
