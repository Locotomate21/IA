# datasets

Los datos **no se versionan en git**. Esta carpeta los organiza en local y fija
la convención de nombres; su contenido está ignorado salvo este README.

## Convención de nombres

Cada dataset vive en una carpeta cuyo nombre codifica su procedencia:

```
<proyecto>_<servicio>_<formato>_<nombre>_<version>_<tarea>_<fecha>/
```

| Campo | Descripción | Ejemplo |
| --- | --- | --- |
| `proyecto` | Repositorio dueño del dato | `ia` |
| `servicio` | Servicio que lo consume | `crop_recommendation` |
| `formato` | Formato en disco | `csv`, `parquet` |
| `nombre` | Identificador del dataset | `crop` |
| `version` | Versión semántica | `v1.0.0` |
| `tarea` | Uso previsto | `training`, `eval` |
| `fecha` | Fecha de corte (`YYYYMMDD`) | `20260812` |

En uso ahora mismo:

```
datasets/
└── ia_crop_recommendation_csv_crop_v1.0.0_training_20260812/
    └── crop_recommendation.csv
```

**Por qué el nombre va en la carpeta y no en el archivo.** Un dataset puede
tener varios archivos —train/val/test, o miles de imágenes— y así la versión se
declara una sola vez, sin riesgo de que se desincronice entre ellos.

**Por qué el nombre es tan largo.** Cuando existan tres versiones del mismo
dato, esa carpeta es la única forma de saber cuál entrenó qué modelo. Los YAML
de experimento apuntan a ella por ruta completa.

## Cómo obtener los datos

Actualmente cada quien descarga el dataset y lo coloca siguiendo la convención.
El de cultivos es *Crop Recommendation Dataset*, disponible en Kaggle.

## Versionado con DVC

**No está configurado todavía.** Si el proyecto crece hasta necesitarlo, el
flujo sería:

```bash
dvc init
dvc remote add -d storage <url-del-remoto>

dvc add datasets/ia_crop_recommendation_csv_crop_v1.0.0_training_20260812
git add datasets/*.dvc datasets/.gitignore
git commit -m "Versionar dataset de cultivos"
dvc push
```

A partir de ahí, `dvc pull` recupera los datos en cualquier máquina y git solo
guarda los punteros `.dvc`, nunca los archivos.
