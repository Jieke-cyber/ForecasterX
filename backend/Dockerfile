FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"


COPY requirements.txt /app/requirements.txt

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

RUN pip install --upgrade pip setuptools wheel

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lag_llama /app/lag_llama



COPY app /app/app
COPY FoundationModel /app/FoundationModel
COPY Modelli /app/Modelli


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
