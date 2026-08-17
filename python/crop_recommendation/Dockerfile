# --- Etapa 1: builder ---
# Instala las dependencias en una imagen que puede tener compiladores y
# archivos temporales. Nada de eso viajara a la imagen final.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Solo requirements.txt, NO requirements_training.txt: mlflow, matplotlib y
# pytest no pintan nada en un servidor de inferencia.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt


# --- Etapa 2: imagen final ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

WORKDIR /app

# Solo las dependencias ya instaladas, sin el bagaje de la etapa anterior.
COPY --from=builder /install /usr/local

# El codigo y lo que necesita para arrancar.
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

# Usuario sin privilegios: si alguien logra ejecutar codigo dentro del
# contenedor, no lo hara como root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run inyecta el puerto en la variable PORT, asi que no lo fijamos a
# mano. La forma shell del CMD permite que ${PORT} se expanda.
CMD exec uvicorn src.server.app:app --host 0.0.0.0 --port ${PORT}
