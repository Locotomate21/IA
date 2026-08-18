# Crop Recommendation API

Servicio que recomienda **qué cultivo sembrar** a partir de las condiciones de
suelo y clima de un terreno, usando una red neuronal MLP entrenada sobre 22
cultivos.

```
POST /crop_recommend
{ "N": 90, "P": 42, "K": 43, "temperature": 20.9,
  "humidity": 82.0, "ph": 6.5, "rainfall": 202.9 }

200 OK
{ "crop": "rice",
  "confidence": 0.6833,
  "alternatives": [
    { "crop": "jute",   "confidence": 0.3163 },
    { "crop": "coffee", "confidence": 0.0004 },
    { "crop": "papaya", "confidence": 0.0    }
  ] }
```

Devuelve alternativas a propósito: con 22 clases, quedarse solo con la ganadora
desperdicia información que el modelo ya calculó. Si el primero saca 0,68 y el
segundo 0,32, quien decide merece saberlo.

## Resultados

| Experimento | Cambio respecto al 01 | Accuracy | F1 macro |
|-------------|----------------------|---------:|---------:|
| 01 | base: `[64,32]`, dropout 0.2, lr 0.001 | 0.9886 | 0.9885 |
| 02 | arquitectura `[128,64,32]` | 0.9955 | 0.9954 |
| 03 | dropout 0.5 | 0.9818 | 0.9817 |
| **04** | **learning rate 0.01** | **0.9977** | **0.9977** |

El modelo en producción es el del experimento **04**: un error en 440 muestras
de validación, con la arquitectura pequeña. El tamaño del paso importó más que
el tamaño de la red.

Más regularización no resultó mejor: el experimento 03 fue el peor y el único
que agotó las 100 épocas sin activar el early stopping. Con solo 7 variables de
entrada, un dropout de 0.5 impide aprender.

## Límites conocidos

**Arroz y yute se confunden.** Aparece en los cuatro experimentos. No es un
fallo del modelo: sus perfiles de suelo y clima están a distancia 1.39 en el
espacio de 7 dimensiones, casi el mismo punto. La distinción no está en los
datos, así que ninguna arquitectura la recupera. El techo lo pone el dataset.

Se detectó en el análisis exploratorio **antes** de entrenar, calculando
distancias euclidianas entre los perfiles medios de cada cultivo.

**Fuera de rango la respuesta no es fiable.** El softmax siempre reparte 1 entre
las 22 clases, así que siempre habrá una ganadora, tenga sentido o no. Con
`N=300` el servicio responde `coffee` con 0,85 de confianza — y añade un aviso:

```json
{ "crop": "coffee", "confidence": 0.8548,
  "warnings": ["N=300.0 esta fuera del rango de entrenamiento (0.0-140.0);
                la prediccion es menos fiable"] }
```

La confianza alta no es sinónimo de acierto.

## Dataset

Crop Recommendation Dataset (Kaggle): 2.200 filas, 7 variables numéricas
(N, P, K, temperatura, humedad, pH, precipitación) y 22 cultivos con 100 filas
cada uno.

Perfectamente balanceado y sin nulos, así que el preprocesamiento no necesita
imputador ni ponderación de clases: solo `StandardScaler` sobre las 7 variables
y `LabelEncoder` sobre el objetivo.

**Ojo:** el CSV viene **ordenado por cultivo**. Un corte train/val sin
estratificar dejaría clases enteras fuera. Por eso `train.py` usa `stratify`, y
hay una prueba que deja constancia de la premisa.

## Estructura

```
crop_recommendation/
├── Dockerfile                       imagen multi-etapa
├── requirements.txt                 solo inferencia (va a la imagen)
├── requirements_training.txt        entrenamiento y pruebas
├── config/training/experiments/     4 experimentos versionados
├── models/                          .pt + preprocesador + label encoder
├── reports/                         matrices de confusión
├── tests/                           18 pruebas
└── src/
    ├── processing/main.py           escalado y codificación
    ├── training/model.py            el MLP
    ├── training/train.py            entrenamiento + MLflow
    ├── inference/predictor.py       carga de artefactos y predicción
    ├── server/schemas.py            validación de entrada y salida
    ├── server/app.py                endpoints
    └── examples/main.py             análisis exploratorio
```

## Entorno

Requiere **Python 3.11** (la misma versión que la imagen Docker).

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate         # Git Bash
python -m pip install -r requirements_training.txt
python --version                      # debe decir 3.11.x
```

## Uso

### Explorar los datos

```bash
python src/examples/main.py
```

Imprime rangos, balance de clases, nulos y los pares de cultivos con perfiles
más parecidos.

### Entrenar

```bash
python src/training/train.py \
  --config config/training/experiments/04-crop_recommendation-mlp-crop-v100-training.yaml
```

Genera tres artefactos en `models/` —modelo, preprocesador y codificador— y una
matriz de confusión en `reports/`. Los tres son necesarios: sin el
preprocesador los datos entrantes se escalarían con otros números, y sin el
codificador la respuesta sería `7` en vez de `"coffee"`.

### Ver los experimentos

```bash
mlflow ui --backend-store-uri sqlite:///../../.mlflow/mlflow.db
```

### Levantar la API

```bash
uvicorn src.server.app:app --reload --port 8000
```

Documentación interactiva en http://localhost:8000/docs

### Pruebas

```bash
python -m pytest tests/ -q
```

18 pruebas: formas y propiedades del modelo, correcciones del preprocesador y
comportamiento de la API.

## Docker

```bash
docker build -t crop-recommendation:1.0 .
docker run -d -p 8000:8080 --name crop-api crop-recommendation:1.0
curl http://localhost:8000/health
```

Dentro del contenedor la API escucha en el puerto que indique la variable
`PORT` (8080 por defecto), como espera Cloud Run.

La imagen pesa **1,78 GB**, casi todo PyTorch. Es el motivo de que
`requirements.txt` contenga solo lo imprescindible para inferir: `mlflow`,
`matplotlib` y `pytest` viven en `requirements_training.txt` y no viajan.

Para reconstruir, primero para y elimina el contenedor:

```bash
docker stop crop-api && docker rm crop-api
```

## Configuración

Cada experimento es un YAML. Cambiar la arquitectura, el optimizador o los
hiperparámetros no requiere tocar código:

```yaml
model_config:
  architecture:
    hidden_layers: [64, 32]
    dropout_rate: 0.2
    use_batch_norm: true
    activation_fn: "ReLU"

training_params:
  epochs: 100
  batch_size: 32
  optimizer:
    name: "Adam"
    learning_rate: 0.01
  early_stopping:
    patience: 10
    delta: 0.001
```

El campo `mlops_config.tracking_uri` admite `"local"` (SQLite dentro del repo),
`"auto"` (respeta la variable de entorno `MLFLOW_TRACKING_URI`) o la URL de un
servidor MLflow.

## Atribución

Este servicio implementa arquitecturas de Deep Learning basadas en los
materiales educativos de [inGeniia.co](https://www.ingeniia.co). El modelo
original de Credit Scoring, del que parte esta estructura, fue desarrollado por
el equipo de inGeniia.

El dataset, el modelo multiclase y la implementación de este servicio son
propios.
