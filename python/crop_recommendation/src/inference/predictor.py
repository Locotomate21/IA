"""
Inferencia: convierte las condiciones de un terreno en un cultivo recomendado.

Carga los tres artefactos que dejó el entrenamiento y los mantiene en memoria:

    modelo .pt          la red con sus pesos aprendidos
    preprocesador       la media y el std de cada variable
    label encoder       la traduccion numero <-> nombre de cultivo

Los tres son imprescindibles. Sin el preprocesador, los datos entrantes se
escalarian con numeros distintos a los del entrenamiento. Sin el codificador,
la respuesta seria "7" en vez de "coffee".

Probar con:
    python src/inference/predictor.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

SERVICE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVICE_ROOT))

from src.training.model import CropRecommendationModel  # noqa: E402

# Rangos observados en el entrenamiento (fase 03). Predecir fuera de estos
# valores es extrapolar: el modelo nunca vio nada parecido y su respuesta no
# es fiable, aunque la devuelva con mucha confianza.
TRAINING_RANGES: Dict[str, tuple] = {
    "N": (0.0, 140.0),
    "P": (5.0, 145.0),
    "K": (5.0, 205.0),
    "temperature": (8.83, 43.68),
    "humidity": (14.26, 99.98),
    "ph": (3.50, 9.94),
    "rainfall": (20.21, 298.56),
}

DEFAULT_CONFIG = (
    SERVICE_ROOT
    / "config"
    / "training"
    / "experiments"
    / "04-crop_recommendation-mlp-crop-v100-training.yaml"
)


class CropPredictor:
    """
    Predictor de cultivos.

    Se construye una sola vez al arrancar el servicio, nunca dentro de un
    endpoint: cargar los artefactos en cada peticion multiplicaria por mil
    el tiempo de respuesta.
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model_cfg = self.config["model_config"]
        self.data_cfg = self.config["data_source"]
        self.class_names: List[str] = self.model_cfg["classes"]

        # CPU: el modelo tiene 3.510 parametros, una GPU seria absurda aqui
        # y Cloud Run no ofrece ninguna.
        self.device = torch.device("cpu")

        self._load_artifacts()

    # -- carga --------------------------------------------------------------

    def _load_artifacts(self) -> None:
        models_dir = SERVICE_ROOT / "models"

        model_path = models_dir / self.model_cfg["model_name"]
        prep_path = models_dir / self.data_cfg["preprocessor_filename"]
        enc_path = models_dir / self.data_cfg["label_encoder_filename"]

        for p in (model_path, prep_path, enc_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Falta el artefacto {p.name}. Entrena primero:\n"
                    f"    python src/training/train.py --config {self.config_path}"
                )

        self.preprocessor = joblib.load(prep_path)
        self.label_encoder = joblib.load(enc_path)
        self.feature_names: List[str] = list(self.preprocessor.transformers_[0][2])

        # El .pt guarda solo los PESOS, no la arquitectura. Hay que
        # reconstruir la red con los mismos parametros del YAML antes de
        # volcarlos dentro.
        arch = self.model_cfg["architecture"]
        self.model = CropRecommendationModel(
            num_features=len(self.feature_names),
            num_classes=len(self.class_names),
            hidden_layers=arch["hidden_layers"],
            dropout_rate=arch["dropout_rate"],
            use_batch_norm=arch["use_batch_norm"],
            activation_fn=arch["activation_fn"],
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # apaga dropout y fija BatchNorm

    # -- prediccion ---------------------------------------------------------

    @staticmethod
    def _check_ranges(features: Dict[str, float]) -> List[str]:
        """Avisa si algun valor cae fuera de lo que el modelo vio al entrenar."""
        avisos = []
        for nombre, valor in features.items():
            minimo, maximo = TRAINING_RANGES[nombre]
            if valor < minimo or valor > maximo:
                avisos.append(
                    f"{nombre}={valor} esta fuera del rango de entrenamiento "
                    f"({minimo}-{maximo}); la prediccion es menos fiable"
                )
        return avisos

    def predict(self, features: Dict[str, float], top_k: int = 3) -> Dict[str, Any]:
        """
        Recomienda un cultivo a partir de las 7 variables del terreno.

        Devuelve el ganador, su confianza y las siguientes alternativas. Con
        22 clases, quedarse solo con la ganadora desperdicia informacion que
        el modelo ya calculo.
        """
        avisos = self._check_ranges(features)

        # DataFrame de una fila con las columnas EN EL ORDEN del entrenamiento.
        # Este es el punto donde un orden distinto arruinaria la prediccion
        # sin lanzar ningun error.
        fila = pd.DataFrame([[features[c] for c in self.feature_names]], columns=self.feature_names)

        x = self.preprocessor.transform(fila)
        tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        probs = self.model.predict_proba(tensor)[0].cpu().numpy()

        # np.argsort ordena de menor a mayor; [::-1] le da la vuelta.
        orden = np.argsort(probs)[::-1]

        resultado = {
            "crop": self.label_encoder.inverse_transform([orden[0]])[0],
            "confidence": round(float(probs[orden[0]]), 4),
            "alternatives": [
                {
                    "crop": self.label_encoder.inverse_transform([i])[0],
                    "confidence": round(float(probs[i]), 4),
                }
                for i in orden[1 : top_k + 1]
            ],
        }
        if avisos:
            resultado["warnings"] = avisos
        return resultado


# ---------------------------------------------------------------------------
# Banco de pruebas
# ---------------------------------------------------------------------------


def main() -> None:
    predictor = CropPredictor()
    print(f"Modelo cargado : {predictor.model_cfg['model_name']}")
    print(f"Variables      : {predictor.feature_names}")
    print(f"Cultivos       : {len(predictor.class_names)}")

    casos = {
        "Arrozal (mucha lluvia y humedad)": {
            "N": 90, "P": 42, "K": 43, "temperature": 20.9,
            "humidity": 82.0, "ph": 6.5, "rainfall": 202.9,
        },
        "Cafetal": {
            "N": 101, "P": 28, "K": 30, "temperature": 25.5,
            "humidity": 58.9, "ph": 6.8, "rainfall": 158.0,
        },
        "Manzano (potasio alto)": {
            "N": 21, "P": 134, "K": 200, "temperature": 22.6,
            "humidity": 92.3, "ph": 5.9, "rainfall": 112.7,
        },
        "Terreno fuera de rango": {
            "N": 300, "P": 42, "K": 43, "temperature": 20.9,
            "humidity": 82.0, "ph": 6.5, "rainfall": 202.9,
        },
    }

    for titulo, features in casos.items():
        r = predictor.predict(features)
        alt = ", ".join(f"{a['crop']} {a['confidence']:.3f}" for a in r["alternatives"])
        print(f"\n{titulo}")
        print(f"  -> {r['crop']}  (confianza {r['confidence']:.4f})")
        print(f"  alternativas: {alt}")
        for aviso in r.get("warnings", []):
            print(f"  AVISO: {aviso}")


if __name__ == "__main__":
    main()
