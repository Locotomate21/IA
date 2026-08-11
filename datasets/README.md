# datasets

Esta carpeta versiona **punteros DVC**, no los datos. Cada archivo `.dvc`
referencia un dataset almacenado en el remoto de DVC; el archivo de datos real
está ignorado por git.

## Convención de nombres

```
<proyecto>_<tipo_de_archivo>_<nombre>_<version>_<tarea>_<timestamp>.dvc
```

| Campo             | Descripción                                  | Ejemplo              |
| ----------------- | -------------------------------------------- | -------------------- |
| `proyecto`        | Servicio o dominio dueño del dato            | `genia_services`     |
| `tipo_de_archivo` | Formato en disco                             | `csv`, `parquet`     |
| `nombre`          | Identificador del dataset                    | `german_credit_risk` |
| `version`         | Versión semántica del dataset                | `v1.0.0`             |
| `tarea`           | Uso previsto                                 | `training`, `eval`   |
| `timestamp`       | Fecha de corte de los datos (`YYYYMMDD`)     | `20250824`           |

Ejemplo:

```
genia_services_csv_german_credit_risk_v1.0.0_training_20250824.dvc
```

## Uso

```bash
# registrar un dataset nuevo
dvc add datasets/genia_services_csv_german_credit_risk_v1.0.0_training_20250824.csv
git add datasets/*.dvc datasets/.gitignore

# recuperar los datos en otra máquina
dvc pull
```
