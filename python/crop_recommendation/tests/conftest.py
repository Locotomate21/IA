"""
Configuración compartida de las pruebas.

pytest ejecuta este archivo antes que los tests, así que es el sitio para
dejar el path listo y definir los datos que varios tests reutilizan.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

NUM_FEATURES = 7
NUM_CLASSES = 22


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    """El CSV real. Si no está, las pruebas que lo usen se saltan."""
    ruta = (
        SERVICE_ROOT.parents[1]
        / "datasets"
        / "ia_crop_recommendation_csv_crop_v1.0.0_training_20260812"
        / "crop_recommendation.csv"
    )
    if not ruta.exists():
        pytest.skip(f"No se encontró el dataset en {ruta}")
    return pd.read_csv(ruta)


@pytest.fixture
def terreno_valido() -> dict:
    """Un arrozal: el caso de ejemplo de la documentación de la API."""
    return {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }
