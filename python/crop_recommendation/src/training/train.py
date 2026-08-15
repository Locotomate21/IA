"""
Entrenamiento del MLP de recomendación de cultivos.

Rebanada vertical (fase 06): el objetivo es que el circuito completo funcione
una vez, no que el modelo sea bueno.

    config YAML -> datos -> preprocesador -> modelo -> bucle -> artefactos

MLflow llega en la fase 07; aquí todavía no aparece.

Ejecutar desde python/crop_recommendation:
    python src/training/train.py --config config/training/experiments/01-crop_recommendation-mlp-crop-v100-training.yaml
"""

import argparse
import logging as log
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# La raíz del servicio (python/crop_recommendation) al path, para poder
# importar src.* sin depender de la variable PYTHONPATH del sistema.
SERVICE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVICE_ROOT))

from src.processing.main import CropDataPreprocessor  # noqa: E402
from src.training.model import CropRecommendationModel  # noqa: E402

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> Dict:
    """Lee el YAML. Es la única fuente de verdad del experimento."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """
    Fija las tres fuentes de aleatoriedad del proyecto.

    Sin esto, dos ejecuciones idénticas darían resultados distintos y sería
    imposible saber si una mejora vino de tu cambio o de la suerte.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# ---------------------------------------------------------------------------
# Entrenador
# ---------------------------------------------------------------------------


class CropModelTrainer:
    def __init__(self, config_path: Path) -> None:
        self.config = load_config(config_path)

        env = self.config["environment"]
        self.seed = env["seed"]
        self.device = resolve_device(env["device"])
        set_seed(self.seed)

        self.data_cfg = self.config["data_source"]
        self.model_cfg = self.config["model_config"]
        self.train_cfg = self.config["training_params"]

        # Las rutas del YAML son relativas a la raíz del servicio.
        self.dataset_path = (SERVICE_ROOT / self.data_cfg["dataset_path"]).resolve()
        self.models_dir = SERVICE_ROOT / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor_helper = CropDataPreprocessor()

        log.info("--- Configuración ---")
        log.info(f"Servicio  : {self.config['project_info']['service_name']}")
        log.info(f"Dataset   : {self.dataset_path}")
        log.info(f"Modelos   : {self.models_dir}")
        log.info(f"Dispositivo: {self.device}")
        log.info(f"Semilla   : {self.seed}")

    # -- datos --------------------------------------------------------------

    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga el CSV y lo parte en entrenamiento y validación.

        El corte se hace ANTES de preprocesar. Si escalaras primero, la media
        incluiría las filas de validación y estarías filtrando información
        que el modelo no debería conocer.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"No se encontró el dataset: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)
        log.info(f"Datos cargados: {df.shape}")

        target = self.preprocessor_helper.target_feature

        # stratify reparte cada cultivo proporcionalmente entre los dos
        # conjuntos. Sin él, el azar podría dejar un cultivo fuera de
        # validación y las métricas de esa clase serían indefinidas.
        df_train, df_val = train_test_split(
            df,
            test_size=self.train_cfg["test_size"],
            random_state=self.train_cfg["random_state"],
            stratify=df[target],
        )
        log.info(f"Entrenamiento: {df_train.shape}  |  Validación: {df_val.shape}")
        return df_train, df_val

    def _build_dataloaders(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, int]:
        """Preprocesa y empaqueta los datos en lotes."""
        # El preprocesador se AJUSTA solo con entrenamiento...
        preprocessor = self.preprocessor_helper.fit_preprocessor(df_train)

        # ...y se APLICA a los dos conjuntos.
        x_train, y_train = self.preprocessor_helper.process_data(df_train, preprocessor)
        x_val, y_val = self.preprocessor_helper.process_data(df_val, preprocessor)

        # CrossEntropyLoss espera las etiquetas como enteros long de forma
        # (batch,), no como float de forma (batch, 1) como en el caso binario.
        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train.to_numpy(), dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val.to_numpy(), dtype=torch.long),
        )

        batch_size = self.train_cfg["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        log.info(f"Lotes por época: {len(train_loader)} entrenamiento, {len(val_loader)} validación")

        self._preprocessor = preprocessor
        return train_loader, val_loader, x_train.shape[1]

    # -- bucle --------------------------------------------------------------

    def _train_one_epoch(self, loader, model, criterion, optimizer) -> float:
        model.train()  # activa dropout y el modo lote de BatchNorm
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()          # 1. borra los gradientes anteriores
            logits = model(x_batch)        # 2. hacia adelante
            loss = criterion(logits, y_batch)  # 3. mide el error
            loss.backward()                # 4. hacia atrás: calcula gradientes
            optimizer.step()               # 5. ajusta los pesos

            total_loss += loss.item() * x_batch.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _evaluate(self, loader, model, criterion) -> Tuple[float, Dict[str, float]]:
        model.eval()  # apaga dropout; BatchNorm usa lo aprendido
        total_loss = 0.0
        y_true, y_pred = [], []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = model(x_batch)
            total_loss += criterion(logits, y_batch).item() * x_batch.size(0)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

        # average="macro": promedia la métrica de las 22 clases dándoles el
        # mismo peso. zero_division=0 evita avisos si alguna clase no se
        # predice nunca, cosa habitual en las primeras épocas.
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
        return total_loss / len(loader.dataset), metrics

    # -- orquestación -------------------------------------------------------

    def train(self) -> None:
        df_train, df_val = self._load_and_split_data()
        train_loader, val_loader, num_features = self._build_dataloaders(df_train, df_val)

        arch = self.model_cfg["architecture"]
        num_classes = len(self.model_cfg["classes"])

        model = CropRecommendationModel(
            num_features=num_features,
            num_classes=num_classes,
            hidden_layers=arch["hidden_layers"],
            dropout_rate=arch["dropout_rate"],
            use_batch_norm=arch["use_batch_norm"],
            activation_fn=arch["activation_fn"],
        ).to(self.device)

        log.info(f"Modelo creado: {model.get_model_info()}")

        criterion = nn.CrossEntropyLoss()

        opt_cfg = self.train_cfg["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg["learning_rate"],
            weight_decay=opt_cfg["weight_decay"],
        )

        epochs = self.train_cfg["epochs"]
        log.info(f"--- Entrenando {epochs} épocas ---")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader, model, criterion, optimizer)
            val_loss, metrics = self._evaluate(val_loader, model, criterion)

            log.info(
                f"Época {epoch}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"accuracy={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}"
            )

        self._save_artifacts(model)
        log.info("--- Entrenamiento terminado ---")

    def _save_artifacts(self, model: nn.Module) -> None:
        """
        Guarda las TRES piezas que la API necesitará.

        Sin el preprocesador, los datos entrantes se escalarían con otros
        números. Sin el codificador, la API devolvería 7 en vez de "coffee".
        Un modelo solo no sirve para nada.
        """
        model_path = self.models_dir / self.model_cfg["model_name"]
        prep_path = self.models_dir / self.data_cfg["preprocessor_filename"]
        enc_path = self.models_dir / self.data_cfg["label_encoder_filename"]

        torch.save(model.state_dict(), model_path)
        joblib.dump(self._preprocessor, prep_path)
        joblib.dump(self.preprocessor_helper.label_encoder, enc_path)

        log.info("--- Artefactos guardados ---")
        for p in (model_path, prep_path, enc_path):
            log.info(f"  {p.name}  ({p.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el MLP de recomendación de cultivos.")
    parser.add_argument("--config", type=str, required=True, help="Ruta al YAML del experimento.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (SERVICE_ROOT / config_path).resolve()

    log.info(f"Config: {config_path}")

    trainer = CropModelTrainer(config_path)
    trainer.train()


if __name__ == "__main__":
    main()
