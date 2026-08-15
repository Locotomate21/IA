"""
API de recomendacion de cultivos.

Levantar en desarrollo desde python/crop_recommendation:
    uvicorn src.server.app:app --reload --port 8000

Documentacion interactiva:
    http://localhost:8000/docs
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

SERVICE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVICE_ROOT))

from src.inference.predictor import CropPredictor  # noqa: E402
from src.server.schemas import CropInput, CropOutput, HealthOutput  # noqa: E402

log = logging.getLogger("crop_recommendation.server")

# Un unico predictor para todo el proceso. Se rellena al arrancar.
predictor: CropPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga los artefactos UNA vez, al arrancar el servicio.

    Hacerlo dentro del endpoint significaria leer tres archivos de disco y
    reconstruir la red en cada peticion: cientos de milisegundos por
    llamada en vez de microsegundos.

    Si los artefactos faltan, el arranque falla aqui y de forma ruidosa. Es
    preferible a arrancar "bien" y devolver error 500 al primer usuario.
    """
    global predictor
    predictor = CropPredictor()
    log.info(f"Modelo cargado: {predictor.model_cfg['model_name']}")
    yield
    predictor = None


app = FastAPI(
    title="Crop Recommendation API",
    description=(
        "Recomienda que cultivo sembrar a partir de las condiciones de suelo "
        "y clima, usando una red neuronal MLP entrenada sobre 22 cultivos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOutput, tags=["infraestructura"])
def health() -> HealthOutput:
    """
    Comprobacion de vida. Cloud Run la usa para decidir si enrutar trafico.

    Devuelve 503 si el modelo no esta cargado, para que el orquestador no
    mande peticiones a una instancia que no puede responderlas.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="El modelo no esta cargado")

    return HealthOutput(
        status="ok",
        model_name=predictor.model_cfg["model_name"],
        num_classes=len(predictor.class_names),
    )


@app.post("/crop_recommend", response_model=CropOutput, tags=["inferencia"])
def crop_recommend(payload: CropInput) -> CropOutput:
    """
    Recomienda un cultivo.

    Pydantic ya valido los rangos antes de llegar aqui, asi que este codigo
    puede confiar en los datos que recibe.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="El modelo no esta cargado")

    resultado = predictor.predict(payload.model_dump())
    return CropOutput(**resultado)


@app.get("/", tags=["infraestructura"])
def root() -> dict:
    return {
        "service": "crop_recommendation",
        "docs": "/docs",
        "endpoints": ["/health", "/crop_recommend"],
    }
