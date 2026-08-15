"""
Esquemas de entrada y salida de la API.

Pydantic valida cada peticion ANTES de que llegue al modelo. Si un campo
falta, no es numerico o esta fuera de rango, FastAPI responde 422 y el
modelo ni se entera.

A diferencia de credit_scoring, aqui no hay Enum: las 7 variables son
numericas, asi que la validacion es por RANGO en vez de por lista de
valores permitidos.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class CropInput(BaseModel):
    """
    Condiciones de suelo y clima de un terreno.

    Los limites son fisicos o de sentido comun, no los del entrenamiento.
    Se permite enviar valores raros pero validos; si caen fuera de lo que el
    modelo vio, la respuesta incluira un aviso en vez de rechazar la
    peticion. Rechazar seria mentir sobre por que no se puede responder.
    """

    N: float = Field(..., ge=0, le=500, description="Nitrogeno en el suelo (kg/ha)")
    P: float = Field(..., ge=0, le=500, description="Fosforo en el suelo (kg/ha)")
    K: float = Field(..., ge=0, le=500, description="Potasio en el suelo (kg/ha)")
    temperature: float = Field(..., ge=-20, le=60, description="Temperatura media (C)")
    humidity: float = Field(..., ge=0, le=100, description="Humedad relativa (%)")
    ph: float = Field(..., ge=0, le=14, description="pH del suelo")
    rainfall: float = Field(..., ge=0, le=1000, description="Precipitacion (mm)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.9,
                "humidity": 82.0,
                "ph": 6.5,
                "rainfall": 202.9,
            }
        }
    }


class CropAlternative(BaseModel):
    """Un cultivo candidato que no gano, con su probabilidad."""

    crop: str
    confidence: float = Field(..., ge=0, le=1)


class CropOutput(BaseModel):
    """
    Respuesta del servicio.

    Devuelve alternativas porque con 22 clases quedarse solo con la ganadora
    desperdicia informacion util: si el primero saca 0.68 y el segundo 0.32,
    el agricultor merece saberlo.
    """

    crop: str = Field(..., description="Cultivo recomendado")
    confidence: float = Field(..., ge=0, le=1, description="Probabilidad del recomendado")
    alternatives: List[CropAlternative] = Field(default_factory=list)
    warnings: Optional[List[str]] = Field(
        default=None, description="Avisos si algun valor cae fuera del rango de entrenamiento"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "crop": "rice",
                "confidence": 0.6833,
                "alternatives": [
                    {"crop": "jute", "confidence": 0.316},
                    {"crop": "coffee", "confidence": 0.0002},
                ],
            }
        }
    }


class HealthOutput(BaseModel):
    """Estado del servicio. Cloud Run lo consulta para saber si esta vivo."""

    status: str
    model_name: str
    num_classes: int
