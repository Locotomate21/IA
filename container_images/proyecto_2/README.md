# proyecto_2

Imagen base del `proyecto_2`.

```bash
docker build -t proyecto_2 .
docker run --rm -it proyecto_2
```

Las dependencias de la imagen se declaran en `requirements.txt` (fijadas por
versión, para que la construcción sea reproducible).
