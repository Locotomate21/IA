# IA — Servicios de Machine Learning

Monorepo de servicios de ML llevados de punta a punta: del dataset al
contenedor. Cada servicio vive aislado, con su propio entorno, sus
dependencias y su imagen Docker.

## Servicios

| Servicio | Tarea | Modelo | Estado |
|----------|-------|--------|--------|
| [crop_recommendation](python/crop_recommendation/) | Recomendar qué cultivo sembrar según suelo y clima | MLP multiclase, 22 clases | Funcionando · 99,77 % |

## Estructura

```
IA/
├── datasets/                 # punteros y datos versionados (no van a git)
│   └── <proyecto>_<servicio>_<formato>_<nombre>_<version>_<tarea>_<fecha>/
│
├── python/
│   ├── crop_recommendation/  # servicio de recomendación de cultivos
│   │   ├── Dockerfile
│   │   ├── requirements.txt             # solo inferencia (va a la imagen)
│   │   ├── requirements_training.txt    # entrenamiento y pruebas
│   │   ├── config/training/experiments/ # un YAML por experimento
│   │   ├── models/                      # artefactos generados
│   │   ├── reports/                     # métricas y gráficas
│   │   ├── tests/
│   │   └── src/
│   │       ├── processing/   # preprocesamiento
│   │       ├── training/     # modelo y entrenamiento
│   │       ├── inference/    # carga de artefactos y predicción
│   │       ├── server/       # API
│   │       └── examples/     # análisis exploratorio
│   │
│   └── shared/mlops/         # utilidades comunes (MLflow)
│
└── .mlflow/                  # tracking local (no versionado)
```

## Convenciones

**Un entorno por servicio.** El `.venv/` vive dentro de la carpeta del servicio
y nunca se versiona. No hay entornos en la raíz: dos entornos activables se
confunden con facilidad.

**Dos archivos de dependencias.** `requirements.txt` contiene solo lo necesario
para inferir y es lo único que entra en la imagen Docker.
`requirements_training.txt` añade entrenamiento y pruebas. Esa separación
mantiene la imagen ligera.

**La configuración manda.** Cada experimento es un YAML en
`config/training/experiments/`. Cambiar la arquitectura, el optimizador o los
hiperparámetros no requiere tocar código.

**Las salidas no se versionan.** `models/`, `reports/` y `mlruns/` llevan su
propio `.gitignore` con `*` y `!.gitignore`: la estructura existe en el
repositorio, el contenido nunca.

**Los datos van aparte.** En `datasets/` la convención de nombres codifica
proyecto, servicio, formato, dataset, versión, tarea y fecha. Ver
[datasets/README.md](datasets/README.md).

## Puesta en marcha

```bash
cd python/crop_recommendation
py -3.11 -m venv .venv
source .venv/Scripts/activate        # Git Bash
python -m pip install -r requirements_training.txt
```

Cada servicio documenta su uso en su propio README.

## Atribución

La arquitectura de este monorepo está basada en los materiales educativos de
[inGeniia.co](https://www.ingeniia.co). El diseño original del servicio de
Credit Scoring, del que parte esta estructura, fue desarrollado por el equipo
de inGeniia.

Los servicios de este repositorio son implementaciones propias sobre datasets
distintos.
