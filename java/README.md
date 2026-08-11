# java

Proyectos del stack Java. Cada proyecto se crea como subcarpeta autocontenida
con su propio build (Maven o Gradle), siguiendo la misma separación que los
proyectos de `python/`: código fuente bajo `src/`, y las salidas de build
(`target/`, `build/`) fuera del control de versiones.

```
java/
└── <proyecto>/
    ├── pom.xml | build.gradle
    ├── Dockerfile
    ├── README.md
    └── src/
        ├── main/java/
        └── test/java/
```
