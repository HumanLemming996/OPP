FROM python:3.11-slim

WORKDIR /app

# Install system deps for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY opp/ opp/

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "opp.server"]
