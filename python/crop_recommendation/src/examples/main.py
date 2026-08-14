"""
Análisis exploratorio del dataset de recomendación de cultivos.

Responde a cinco preguntas cuyas respuestas necesitan las fases siguientes:
  1. Forma y columnas          -> qué estamos manejando
  2. Rangos de las variables   -> serán las validaciones de la API (fase 10)
  3. Balance de clases         -> decide si hace falta ponderar la pérdida
  4. Nulos                     -> decide si hace falta imputar (fase 04)
  5. Cultivos parecidos        -> anticipa los errores del modelo (fase 08)

Ejecutar desde python/crop_recommendation:
    python src/examples/main.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
# __file__ es la ruta de ESTE archivo. Anclando las rutas a él en vez de a la
# carpeta desde donde se ejecuta, el script funciona sin importar dónde estés
# parado en la terminal.
#
#   main.py -> examples -> src -> crop_recommendation -> python -> IA
#   parents:     [0]       [1]         [2]                [3]      [4]

SERVICE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    REPO_ROOT
    / "datasets"
    / "ia_crop_recommendation_csv_crop_v1.0.0_training_20260812"
    / "crop_recommendation.csv"
)

TARGET = "label"

# Que pandas no corte las tablas anchas al imprimir.
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def titulo(texto: str) -> None:
    """Imprime un separador legible. El yo del mes que viene lo agradecerá."""
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Carga y forma
    # -----------------------------------------------------------------------
    titulo("1. QUÉ ESTAMOS MANEJANDO")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print(f"Archivo   : {DATASET_PATH.name}")
    print(f"Filas     : {df.shape[0]}")
    print(f"Columnas  : {df.shape[1]}")
    print(f"\nTipos de dato:\n{df.dtypes}")
    print(f"\nPrimeras filas:\n{df.head()}")

    # Las 7 variables de entrada son todas menos el objetivo.
    features = [c for c in df.columns if c != TARGET]

    # -----------------------------------------------------------------------
    # 2. Rangos -> validaciones de la API
    # -----------------------------------------------------------------------
    titulo("2. RANGOS DE LAS VARIABLES  (serán los Field(ge=…, le=…) de la API)")

    resumen = df[features].describe().T[["min", "max", "mean", "std"]]
    print(resumen.round(2))

    print("\nPara copiar a schemas.py en la fase 10:")
    for col in features:
        print(f"    {col:<12} ge={df[col].min():.1f}, le={df[col].max():.1f}")

    # -----------------------------------------------------------------------
    # 3. Balance de clases -> ¿hace falta ponderar la pérdida?
    # -----------------------------------------------------------------------
    titulo("3. BALANCE DE CLASES")

    conteo = df[TARGET].value_counts()
    print(f"Cultivos distintos: {conteo.size}")
    print(f"Mínimo por clase  : {conteo.min()}")
    print(f"Máximo por clase  : {conteo.max()}")

    if conteo.min() == conteo.max():
        print("\n-> Dataset perfectamente balanceado.")
        print("   No hace falta pos_weight, y accuracy es una métrica honesta.")
    else:
        print("\n-> Hay desbalance: habrá que ponderar la pérdida.")

    print(f"\nCultivos:\n{sorted(df[TARGET].unique())}")

    # -----------------------------------------------------------------------
    # 4. Nulos -> ¿hace falta imputar?
    # -----------------------------------------------------------------------
    titulo("4. VALORES NULOS")

    nulos = df.isnull().sum()
    total = int(nulos.sum())

    if total == 0:
        print("Sin nulos. El preprocesador de la fase 04 no necesita imputador.")
    else:
        print(f"Hay {total} nulos:\n{nulos[nulos > 0]}")

    # -----------------------------------------------------------------------
    # 5. Cultivos parecidos -> los errores futuros del modelo
    # -----------------------------------------------------------------------
    titulo("5. QUÉ CULTIVOS SE PARECEN ENTRE SÍ")

    # El perfil de cada cultivo es la media de sus 7 variables.
    perfiles = df.groupby(TARGET)[features].mean()

    # Estandarizamos por columna para que todas pesen igual en la distancia:
    # 'rainfall' llega a 300 y 'ph' a 14, así que sin esto la lluvia dominaría.
    z = (perfiles - perfiles.mean()) / perfiles.std()

    # Distancia euclidiana entre cada par de perfiles.
    # El truco de las dimensiones (X[:, None] - X[None, :]) resta cada fila
    # contra todas las demás de una vez, sin escribir dos bucles.
    x = z.to_numpy()
    distancias = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))

    # Solo la mitad superior de la matriz: (a,b) y (b,a) son el mismo par,
    # y la diagonal es la distancia de cada cultivo consigo mismo (cero).
    nombres = perfiles.index.to_list()
    pares = [
        (nombres[i], nombres[j], distancias[i, j])
        for i in range(len(nombres))
        for j in range(i + 1, len(nombres))
    ]
    pares.sort(key=lambda p: p[2])

    print("Los 8 pares más parecidos (menor distancia = más fáciles de confundir):\n")
    for a, b, d in pares[:8]:
        print(f"    {d:6.2f}   {a}  ~  {b}")

    print("\nGuarda estos nombres: son los que esperarás ver confundidos")
    print("en la matriz de confusión de la fase 08.")

    print(f"\nPerfil medio de cada cultivo:\n{perfiles.round(1)}")


if __name__ == "__main__":
    main()
