# container_images

Definiciones de imágenes de contenedor reutilizables. Cada subcarpeta describe
una imagen independiente y se construye por separado.

```
container_images/
└── <proyecto>/
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

Construcción:

```bash
docker build -t <proyecto> container_images/<proyecto>
```

A diferencia del `Dockerfile` que vive dentro de cada proyecto de `python/`
—que empaqueta ese servicio concreto— las imágenes de esta carpeta sirven como
base común: fijan la versión de Python y las dependencias pesadas compartidas,
para que las imágenes de servicio se construyan rápido encima de ellas.
