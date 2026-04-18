# Hugging Face Spaces (Docker SDK): full control over pip so we can install
# lingbot-world without pulling flash_attn (we patch SDPA in app.py instead).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir "git+https://github.com/robbyant/lingbot-world.git" --no-deps

COPY . /app

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860

CMD ["python", "app.py"]
