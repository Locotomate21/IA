"""
Modelo MLP para recomendación de cultivos.

Red neuronal de clasificación MULTICLASE: recibe las 7 variables de suelo y
clima ya escaladas, y devuelve una puntuación para cada uno de los 22 cultivos.

Diferencias frente al modelo binario de credit_scoring:
    última capa      1 neurona          ->  22 neuronas
    predict_proba    sigmoid            ->  softmax
    predict          umbral en 0.5      ->  argmax
    pérdida (train)  BCEWithLogitsLoss  ->  CrossEntropyLoss

Probar con:
    python src/training/model.py
"""

from typing import Dict, List

import torch
import torch.nn as nn


class CropRecommendationModel(nn.Module):
    """
    Perceptrón multicapa para clasificar 22 cultivos.

    Cada capa oculta es una secuencia de cuatro piezas:
        Linear -> BatchNorm -> Activación -> Dropout

    La capa de salida es un Linear pelado: sin activación, a propósito.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_layers: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        activation_fn: str = "ReLU",
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_fn = activation_fn

        # Construimos las capas en un bucle en vez de a mano. Así la
        # arquitectura queda definida por la lista hidden_layers, y cambiarla
        # desde el YAML no requiere tocar este archivo.
        layers: List[nn.Module] = []
        in_features = num_features

        for units in hidden_layers:
            layers.append(nn.Linear(in_features, units))

            # BatchNorm va ANTES de la activación: normaliza la salida lineal
            # para que la activación reciba valores centrados y el
            # entrenamiento sea más estable.
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(units))

            layers.append(self._build_activation(activation_fn))

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            # La salida de esta capa es la entrada de la siguiente.
            in_features = units

        # Capa de salida: una neurona por cultivo, SIN activación.
        # Devuelve logits crudos porque CrossEntropyLoss aplica el softmax
        # internamente. Añadirlo aquí sería normalizar dos veces, y el
        # modelo aprendería mucho peor sin lanzar ningún error.
        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        """Traduce el nombre del YAML a un módulo de PyTorch."""
        disponibles = {
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "ELU": nn.ELU,
            "GELU": nn.GELU,
            "Tanh": nn.Tanh,
        }
        if name not in disponibles:
            raise ValueError(
                f"Activación '{name}' no soportada. Usa una de: {list(disponibles)}"
            )
        # Una instancia nueva por capa: reutilizar el mismo objeto funciona
        # con ReLU (no tiene estado) pero rompe con activaciones que sí lo
        # tienen. Mejor no acostumbrarse.
        return disponibles[name]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante.

        Entrada : (batch, num_features)
        Salida  : (batch, num_classes) -> LOGITS, no probabilidades.

        Un logit puede ser negativo o mayor que 1; no es una probabilidad.
        Es la puntuación bruta que el softmax convertirá después.
        """
        return self.network(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Probabilidades por cultivo. Cada fila suma 1.

        softmax convierte 22 puntuaciones cualesquiera en 22 probabilidades:
        exponencia cada una (todas quedan positivas) y divide por la suma
        (el total queda en 1). dim=1 indica que la suma es por fila, es
        decir por muestra, no por columna.
        """
        self.eval()
        return torch.softmax(self.forward(x), dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cultivo predicho: el índice de la probabilidad más alta.

        En binario se comparaba contra un umbral de 0.5. Con 22 clases eso
        no aplica: se elige la mayor, sea 0.9 o 0.15.
        """
        return torch.argmax(self.predict_proba(x), dim=1)

    def get_model_info(self) -> Dict:
        """Resumen de la arquitectura, para registrar en MLflow."""
        total = sum(p.numel() for p in self.parameters())
        entrenables = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation_fn": self.activation_fn,
            "total_params": total,
            "trainable_params": entrenables,
        }


# ---------------------------------------------------------------------------
# Banco de pruebas
# ---------------------------------------------------------------------------


def main() -> None:
    NUM_FEATURES = 7
    NUM_CLASSES = 22
    BATCH = 4

    model = CropRecommendationModel(
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        hidden_layers=[64, 32],
        dropout_rate=0.2,
        use_batch_norm=True,
        activation_fn="ReLU",
    )

    print("=== ARQUITECTURA ===")
    print(model)
    print(f"\n=== INFO ===\n{model.get_model_info()}")

    # Un lote falso con la forma que tendrán los datos reales.
    x = torch.randn(BATCH, NUM_FEATURES)

    logits = model(x)
    print(f"\n=== FORMAS ===")
    print(f"entrada  : {tuple(x.shape)}       (esperado: (4, 7))")
    print(f"logits   : {tuple(logits.shape)}      (esperado: (4, 22))")

    probs = model.predict_proba(x)
    pred = model.predict(x)
    print(f"probs    : {tuple(probs.shape)}      (esperado: (4, 22))")
    print(f"predicc. : {tuple(pred.shape)}          (esperado: (4,))")

    # La comprobación que de verdad importa: los logits NO suman 1,
    # las probabilidades SI.
    print(f"\n=== LOGITS vs PROBABILIDADES ===")
    print(f"suma de logits por fila : {logits.sum(dim=1).tolist()}")
    print(f"suma de probs por fila  : {probs.sum(dim=1).round(decimals=4).tolist()}")
    print(f"rango de logits         : {logits.min():.2f} .. {logits.max():.2f}")
    print(f"rango de probs          : {probs.min():.4f} .. {probs.max():.4f}")

    print(f"\ncultivos predichos (indices): {pred.tolist()}")


if __name__ == "__main__":
    main()
