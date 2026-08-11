# IA — genIA_services

Monorepo de servicios de IA. Cada proyecto vive aislado dentro de su stack
(`python/`, `java/`), los datasets se versionan con DVC y las imágenes de
contenedor se definen por proyecto en `container_images/`.

## Arquitectura

```
genIA_services/
├── datasets/                 # punteros DVC a los datos versionados (no los datos)
│   └── <proyecto>_<tipo>_<nombre>_<version>_<tarea>_<timestamp>.dvc
│
├── python/
│   ├── credit_scoring/       # proyecto de scoring crediticio
│   │   ├── .venv/            # entorno virtual (python 3.11, no versionado)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   ├── artifacts/        # salidas de entrenamiento (no versionado)
│   │   ├── config/           # configuración declarativa (Hydra / OmegaConf)
│   │   ├── mlruns/           # tracking local de MLflow (no versionado)
│   │   ├── models/           # modelos serializados (no versionado)
│   │   ├── reports/          # métricas, gráficas y reportes (no versionado)
│   │   ├── tests/            # pruebas unitarias e integración
│   │   └── src/
│   │       ├── examples/     # scripts de ejemplo / notebooks ejecutables
│   │       ├── inference/    # carga de modelo y predicción
│   │       ├── processing/   # ingesta, limpieza y feature engineering
│   │       ├── server/       # API de servicio del modelo
│   │       └── training/     # entrenamiento y evaluación
│   └── project_2/
│
├── java/                     # proyectos del stack Java
│
└── container_images/         # imágenes base compartidas por proyecto
    ├── proyecto_1/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── README.md
    └── proyecto_2/
```

## Convenciones

- **Un proyecto = un entorno virtual.** El `.venv/` vive dentro de la carpeta del
  proyecto y nunca se versiona.
- **Código fuente bajo `src/`.** Las carpetas de datos y salidas (`artifacts/`,
  `models/`, `mlruns/`, `reports/`) quedan fuera de `src/` y se ignoran en git;
  cada una conserva su `.gitignore` para que la estructura sí exista en el repo.
- **Datos por DVC.** En `datasets/` solo se versionan los archivos `.dvc`. Ver
  [datasets/README.md](datasets/README.md) para la convención de nombres.

## Puesta en marcha

```bash
cd python/credit_scoring
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
