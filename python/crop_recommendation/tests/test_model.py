"""
Pruebas del modelo.

Comprueban el contrato de formas y la propiedad que define una salida
multiclase: que las probabilidades sumen 1.
"""

import torch

from conftest import NUM_CLASSES, NUM_FEATURES
from src.training.model import CropRecommendationModel


def _modelo() -> CropRecommendationModel:
    return CropRecommendationModel(
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        hidden_layers=[64, 32],
        dropout_rate=0.2,
    )


def test_forward_devuelve_una_puntuacion_por_clase():
    """La salida debe tener 22 columnas, una por cultivo."""
    logits = _modelo()(torch.randn(4, NUM_FEATURES))
    assert logits.shape == (4, NUM_CLASSES)


def test_las_probabilidades_suman_uno():
    """
    Propiedad que define al softmax. Si falla, o falta el softmax o se está
    aplicando sobre la dimensión equivocada.
    """
    probs = _modelo().predict_proba(torch.randn(8, NUM_FEATURES))
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)
    assert (probs >= 0).all()


def test_forward_devuelve_logits_no_probabilidades():
    """
    Los logits NO deben estar normalizados.

    Si esta prueba fallara, forward() estaría aplicando softmax y
    CrossEntropyLoss lo aplicaría otra vez: el modelo aprendería mal sin
    lanzar ningún error. Es el fallo silencioso más caro del proyecto.
    """
    logits = _modelo()(torch.randn(16, NUM_FEATURES))
    assert not torch.allclose(logits.sum(dim=1), torch.ones(16), atol=1e-3)


def test_predict_devuelve_indices_validos():
    """La predicción es un índice de clase, entre 0 y 21."""
    pred = _modelo().predict(torch.randn(5, NUM_FEATURES))
    assert pred.shape == (5,)
    assert pred.min() >= 0 and pred.max() < NUM_CLASSES


def test_la_arquitectura_sale_del_parametro():
    """Cambiar hidden_layers debe cambiar el número de parámetros."""
    pequeno = CropRecommendationModel(NUM_FEATURES, NUM_CLASSES, hidden_layers=[16])
    grande = CropRecommendationModel(NUM_FEATURES, NUM_CLASSES, hidden_layers=[128, 64])
    assert (
        grande.get_model_info()["total_params"] > pequeno.get_model_info()["total_params"]
    )


def test_activacion_desconocida_falla_pronto():
    """Un nombre inválido en el YAML debe romper al construir, no al entrenar."""
    try:
        CropRecommendationModel(NUM_FEATURES, NUM_CLASSES, activation_fn="NoExiste")
    except ValueError:
        return
    raise AssertionError("Debería haber lanzado ValueError")
