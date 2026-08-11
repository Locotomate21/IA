# proyecto_1

Imagen base del `proyecto_1`.

```bash
docker build -t proyecto_1 .
docker run --rm -it proyecto_1
```

Las dependencias de la imagen se declaran en `requirements.txt` (fijadas por
versión, para que la construcción sea reproducible).
