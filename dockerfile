FROM python:3.11-slim

WORKDIR /app

# Required system packages for OpenCV/ffmpeg and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.docker.txt requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt
# Install PyTorch CPU wheels (use official index)
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
# Copy project
COPY . .

# Default command (can be overridden by docker run)
ENTRYPOINT ["python", "main.py"]