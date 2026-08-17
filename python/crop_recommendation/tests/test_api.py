"""
Pruebas de la API.

Usan TestClient, que levanta la aplicación en memoria: no hace falta arrancar
uvicorn ni ocupar un puerto.

Si los artefactos del modelo no existen, el arranque falla y estas pruebas se
saltan con un mensaje claro en vez de reventar.
"""

import pytest
from fastapi.testclient import TestClient

from src.server.app import app


@pytest.fixture(scope="module")
def client():
    try:
        with TestClient(app) as c:
            yield c
    except FileNotFoundError as e:
        pytest.skip(f"Faltan artefactos del modelo: {e}")


def test_health_responde_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["num_classes"] == 22


def test_recomienda_un_cultivo(client, terreno_valido):
    r = client.post("/crop_recommend", json=terreno_valido)
    assert r.status_code == 200

    body = r.json()
    assert isinstance(body["crop"], str)
    assert 0 <= body["confidence"] <= 1
    assert len(body["alternatives"]) == 3


def test_el_arrozal_da_arroz(client, terreno_valido):
    """
    Prueba de comportamiento, no solo de formato: un terreno con mucha
    lluvia y humedad debe recomendar arroz.
    """
    r = client.post("/crop_recommend", json=terreno_valido)
    assert r.json()["crop"] == "rice"


def test_ph_imposible_es_rechazado(client, terreno_valido):
    """El pH va de 0 a 14. Pydantic debe frenarlo antes de llegar al modelo."""
    terreno_valido["ph"] = 20
    r = client.post("/crop_recommend", json=terreno_valido)
    assert r.status_code == 422


def test_campo_faltante_es_rechazado(client, terreno_valido):
    del terreno_valido["rainfall"]
    r = client.post("/crop_recommend", json=terreno_valido)
    assert r.status_code == 422


def test_valor_fuera_de_rango_avisa(client, terreno_valido):
    """
    N=300 es físicamente posible pero el modelo nunca vio nada así. Debe
    responder igualmente, con un aviso de que está extrapolando.
    """
    terreno_valido["N"] = 300
    r = client.post("/crop_recommend", json=terreno_valido)

    assert r.status_code == 200
    assert r.json()["warnings"], "Debería avisar de la extrapolación"
