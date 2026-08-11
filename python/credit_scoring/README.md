# credit_scoring

Servicio de scoring crediticio: entrenamiento, evaluación y publicación de un
modelo de riesgo sobre el dataset `german_credit_risk`.

## Estructura

```
credit_scoring/
├── .venv/              # entorno virtual, python 3.11 (ignorado)
├── Dockerfile
├── requirements.txt
├── artifacts/          # salidas intermedias del entrenamiento (ignorado)
├── config/             # configuración declarativa (Hydra / OmegaConf)
├── mlruns/             # tracking local de MLflow (ignorado)
├── models/             # modelos serializados (ignorado)
├── reports/            # métricas y gráficas generadas (ignorado)
├── tests/              # pruebas
└── src/
    ├── examples/       # scripts de ejemplo de punta a punta
    ├── inference/      # carga del modelo y predicción
    ├── processing/     # ingesta, limpieza y feature engineering
    ├── server/         # API de servicio
    └── training/       # entrenamiento y evaluación
```

Las carpetas ignoradas conservan un `.gitignore` propio, de modo que la
estructura existe en el repo pero su contenido nunca se versiona.

## Entorno

Requiere **Python 3.11**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

## Datos

Los datos no viven en el repo. Se recuperan con DVC desde la raíz:

```bash
dvc pull
```

Ver [../../datasets/README.md](../../datasets/README.md).

## Configuración

`config/config.yaml` centraliza rutas, hiperparámetros y ajustes de MLflow.
Los valores se pueden sobreescribir por línea de comandos:

```bash
python -m src.training.train training.epochs=50 model.hidden_layers=[128,64]
```

## Docker

```bash
docker build -t credit_scoring .
docker run --rm -p 8000:8000 credit_scoring
```
