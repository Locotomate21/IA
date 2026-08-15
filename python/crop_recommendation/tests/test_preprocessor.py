"""
Pruebas del preprocesador.

Cubren los dos fallos silenciosos de esta pieza: un escalado que no escala,
y un codificador de etiquetas que no sabe volver.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from conftest import NUM_CLASSES, NUM_FEATURES
from src.processing.main import CropDataPreprocessor


def test_forma_y_clases(dataset):
    prep = CropDataPreprocessor()
    preprocessor = prep.fit_preprocessor(dataset)
    x, y = prep.process_data(dataset, preprocessor)

    assert x.shape == (len(dataset), NUM_FEATURES)
    assert y.min() == 0 and y.max() == NUM_CLASSES - 1
    assert len(set(y)) == NUM_CLASSES


def test_el_escalado_deja_media_cero_y_std_uno(dataset):
    """
    Lo que define a StandardScaler. Si fallara, las variables seguirían en
    escalas distintas y las de números grandes dominarían el entrenamiento.
    """
    prep = CropDataPreprocessor()
    preprocessor = prep.fit_preprocessor(dataset)
    x, _ = prep.process_data(dataset, preprocessor)

    assert np.allclose(x.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(x.std(axis=0), 1, atol=1e-2)


def test_el_codificador_traduce_ida_y_vuelta(dataset):
    """
    Sin esto, la API devolvería 7 en vez de "coffee".
    """
    prep = CropDataPreprocessor()
    prep.fit_preprocessor(dataset)

    nombres = prep.class_names
    assert len(nombres) == NUM_CLASSES

    codigos = list(range(NUM_CLASSES))
    devueltos = prep.decode(codigos)
    assert list(devueltos) == nombres


def test_el_orden_de_columnas_es_estable(dataset):
    """
    El preprocesador debe seleccionar por lista explícita, no por el orden
    en que vengan las columnas. Si no, en inferencia escalaría cada valor
    con la media de otra variable, sin lanzar ningún error.
    """
    prep = CropDataPreprocessor()
    preprocessor = prep.fit_preprocessor(dataset)
    x_normal, _ = prep.process_data(dataset, preprocessor)

    # Mismas columnas, orden invertido.
    barajado = dataset[list(reversed(dataset.columns))]
    x_barajado, _ = prep.process_data(barajado, preprocessor)

    assert np.allclose(x_normal, x_barajado)


def test_ajustar_solo_con_entrenamiento(dataset):
    """
    Ajustar con un subconjunto y aplicar a otro debe dar medias distintas de
    cero en el segundo. Si diera exactamente cero, el preprocesador se
    estaría reajustando dentro de process_data: fuga de información.

    El corte va estratificado a propósito. El CSV está ORDENADO POR CULTIVO,
    así que un dataset.iloc[:mitad] dejaría fuera clases enteras y el
    LabelEncoder fallaría al encontrarlas después. Es exactamente el motivo
    por el que train.py usa stratify.
    """
    df_a, df_b = train_test_split(
        dataset, test_size=0.5, random_state=42, stratify=dataset["label"]
    )

    prep = CropDataPreprocessor()
    preprocessor = prep.fit_preprocessor(df_a)
    x_b, _ = prep.process_data(df_b, preprocessor)

    assert not np.allclose(x_b.mean(axis=0), 0, atol=1e-6)


def test_el_csv_esta_ordenado_por_cultivo(dataset):
    """
    Deja constancia del hallazgo: un corte sin estratificar perdería clases
    enteras. Si algún día el dataset llega barajado, esta prueba avisará de
    que la premisa cambió.
    """
    primeras_100 = dataset["label"].head(100).nunique()
    assert primeras_100 == 1, "El CSV ya no está agrupado por cultivo"
