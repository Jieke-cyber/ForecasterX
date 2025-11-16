# backend/Dockerfile
FROM python:3.10-slim

# Toolchain per compilare eventuali wheels (argon2-cffi, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"


# Copia requirements nel container
COPY requirements.txt /app/requirements.txt

# Config pip / env
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

# Aggiorna pip/setuptools/wheel
RUN pip install --upgrade pip setuptools wheel

# 🔹 Installa TORCH CPU dal canale ufficiale PyTorch
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

# 🔹 Installa tutte le altre dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lag_llama /app/lag_llama



# Codice applicazione
COPY app /app/app
COPY FoundationModel /app/FoundationModel


EXPOSE 8000

# Avvio FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
