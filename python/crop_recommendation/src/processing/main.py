"""
Preprocesamiento del dataset de recomendación de cultivos.

Convierte el CSV crudo en lo que la red necesita:
  - Las 7 variables escaladas (media 0, std 1)
  - Los 22 cultivos convertidos de texto a enteros 0..21

Decisiones tomadas a partir del EDA (fase 03):
  - Las 7 variables son numéricas  -> no hace falta OneHotEncoder
  - No hay nulos                   -> no hace falta imputador
  - 22 clases de texto             -> hace falta LabelEncoder

Probar con:
    python src/processing/main.py
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


class CropDataPreprocessor:
    """
    Preprocesador del dataset de cultivos.

    Misma interfaz que CreditDataPreprocessor, para que train.py pueda
    llamarla igual: fit_preprocessor() y process_data().

    Uso previsto:
        prep = CropDataPreprocessor()
        preprocessor = prep.fit_preprocessor(df_train)   # aprende
        x_tr, y_tr = prep.process_data(df_train, preprocessor)
        x_va, y_va = prep.process_data(df_val,   preprocessor)   # solo aplica
    """

    def __init__(self) -> None:
        # Las 7 variables de entrada. El ORDEN de esta lista es el contrato
        # con el modelo: StandardScaler aprende una media y un std por
        # posición, así que en inferencia las columnas deben llegar igual.
        self.numerical_features = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ]

        self.target_feature = "label"

        # El traductor cultivo <-> número. Se ajusta UNA sola vez, en
        # fit_preprocessor, y vive como atributo para que train.py pueda
        # recuperarlo y guardarlo en disco junto al modelo.
        self.label_encoder = LabelEncoder()

    # -- API principal ------------------------------------------------------

    def fit_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Aprende las transformaciones a partir de los datos de entrenamiento.

        Aprende dos cosas:
          1. La media y el std de cada variable  -> en el ColumnTransformer
          2. La correspondencia cultivo <-> número -> en self.label_encoder

        Solo debe llamarse con el conjunto de ENTRENAMIENTO. Si lo ajustaras
        con todo el dataset, la media incluiría los datos de validación y
        estarías filtrando información que el modelo no debería conocer.
        """
        # Una sola rama: escalar las numéricas. No hay categóricas que
        # codificar ni nulos que imputar, así que el pipeline es de un paso.
        numeric_tf = Pipeline(steps=[("scaler", StandardScaler())])

        # remainder="drop" (el valor por defecto) descarta cualquier columna
        # no listada. Es la opción estricta: si mañana el CSV trae un ID o un
        # índice, se descarta en vez de colarse al modelo SIN escalar.
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_tf, self.numerical_features)],
            remainder="drop",
        )

        # Seleccionar por lista explícita (y no df.drop(target)) fija el orden
        # de las columnas. Es lo que garantiza que en inferencia cada valor
        # se escale con la media y el std que le corresponden.
        x_train = df[self.numerical_features]
        preprocessor.fit(x_train)

        # LabelEncoder ordena las clases alfabéticamente:
        #   apple=0, banana=1, ... watermelon=21
        self.label_encoder.fit(df[self.target_feature])

        return preprocessor

    def process_data(
        self, df: pd.DataFrame, preprocessor: ColumnTransformer
    ) -> Tuple[np.ndarray, pd.Series]:
        """
        Aplica lo aprendido y separa features del objetivo.

        Devuelve:
            x_processed : np.ndarray de forma (n_filas, 7), ya escalado
            y           : pd.Series de enteros 0..21
        """
        # .transform(), nunca .fit_transform(): el preprocesador llega ya
        # ajustado. Reajustarlo aquí recalcularía la media con los datos de
        # validación, que es justo lo que queremos evitar.
        x = df[self.numerical_features]
        x_processed = preprocessor.transform(x)

        # Mismo criterio con el objetivo: transform, no fit_transform.
        y = pd.Series(
            self.label_encoder.transform(df[self.target_feature]),
            index=df.index,
            name=self.target_feature,
        )

        return x_processed, y

    # -- Utilidades para inferencia ----------------------------------------

    @property
    def class_names(self) -> List[str]:
        """Los 22 cultivos en el orden que les asignó el LabelEncoder."""
        return self.label_encoder.classes_.tolist()

    def decode(self, codes) -> np.ndarray:
        """Traduce números de vuelta a nombres: 7 -> 'coffee'."""
        return self.label_encoder.inverse_transform(codes)


# ---------------------------------------------------------------------------
# Banco de pruebas: define qué significa "terminado" para esta fase.
# ---------------------------------------------------------------------------

DATASET_PATH = (
    Path(__file__).resolve().parents[4]
    / "datasets"
    / "ia_crop_recommendation_csv_crop_v1.0.0_training_20260812"
    / "crop_recommendation.csv"
)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    print(f"CSV cargado: {df.shape}")

    prep = CropDataPreprocessor()
    preprocessor = prep.fit_preprocessor(df)
    x, y = prep.process_data(df, preprocessor)

    print(f"\nx_processed : {x.shape}      (esperado: (2200, 7))")
    print(f"y           : {y.shape}, valores {y.min()}..{y.max()}   (esperado: 0..21)")
    print(f"clases      : {len(set(y))}                (esperado: 22)")

    # Comprobación del escalado: cada columna debe quedar en media ~0, std ~1.
    x_df = pd.DataFrame(x, columns=prep.numerical_features)
    print(f"\nmedia por columna (~0): {x_df.mean().round(3).to_dict()}")
    print(f"std por columna (~1)  : {x_df.std().round(3).to_dict()}")

    # La prueba que de verdad importa: recuperar el nombre desde el número.
    # Si esto falla, la API devolvería números en vez de cultivos.
    print(f"\nPrimeras 3 filas -> códigos {y.head(3).tolist()}")
    print(f"                 -> cultivos {prep.decode(y.head(3)).tolist()}")
    print(f"\nLas 22 clases: {prep.class_names}")


if __name__ == "__main__":
    main()
