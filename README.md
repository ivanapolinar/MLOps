# MLOps

Comandos Rápidos
----------------

- `make pipeline-class`
  - Ejecuta pruebas unitarias (solo `tests/`).
  - Corre el pipeline end-to-end con la clase `MLOpsPipeline`:
    limpia datos, entrena (base + tuning), registra/loguea en MLflow,
    y genera evidencias en `reports/`.
  - Levanta la API (`uvicorn`), valida endpoints con pruebas de API y
    detiene el servidor automáticamente.

- `make pipeline-deploy msg="TuNombre: descripción"`
  - Actualiza el repo y datos: `git pull --rebase` y `dvc pull`.
  - Ejecuta `pipeline-class` (tests + pipeline + pruebas de API).
  - Versiona y envía artefactos a DVC remoto (p. ej., S3): `dvc add/push`
    para `data/clean/steel_energy_clean.csv` y `models/final_model.joblib`.
  - Sube cambios a Git: `git add/commit/push` (usa `msg` si se indica,
    o un mensaje por defecto si no).
  - Construye la imagen Docker con el modelo, la publica en Docker Hub
    (requiere `docker login` y `DOCKER_REGISTRY_USER`) y ejecuta
    smoke tests (`/health`, `/predict`) a partir de la imagen que vuelve a descargar.

- `make prepare-update`
  - Prepara tu entorno local antes de trabajar: hace `git pull --ff-only`
    (o usa `NO_PULL=true` para omitir), `dvc pull` limitado (solo .dvc de datos
    y modelo si existe `dvc.lock`) y ejecuta `pipeline-class` para validar que
    todo corre correctamente con lo último del remoto. Restaura tus cambios
    locales (stash) al finalizar. No versiona ni sube artefactos.

Notas
- Para registrar en el Model Registry de MLflow, exporta variables de entorno:
  `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`, `MLFLOW_REGISTER_IN_REGISTRY=true`
  y `MLFLOW_REGISTERED_MODEL_NAME`.
- Asegura credenciales AWS para `dvc push` si usas S3.


## Inicializacion de ambientes

Para reconstruir el entorno local (Python 3.11, virtualenv, dependencias base, DVC y MLflow) usa los scripts que viven en `docs/initAmbiente/`:

- `init.bat` automatiza el flujo completo en Windows/PowerShell (incluye manejo de virtualenv, instalación por etapas y preparación de DVC/MLflow).
- `init.sh` hace lo mismo en Linux o WSL.

El detalle paso a paso, flags soportados y requisitos están documentados en `docs/initAmbiente/README.md`. Refiérete a ese archivo cada vez que necesites provisionar una nueva máquina o ajustar el comportamiento de los scripts.

## Contenerización del servicio ML

Se añadió un `Dockerfile` en la raíz que empaqueta la API de FastAPI (`src/api/main.py`), las dependencias del proyecto y el modelo serializado (`models/best_rf_model.joblib`). El contenedor expone el puerto `8000` y arranca Uvicorn con `src.api.main:app`.

### Prueba local con Uvicorn

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Con el servidor arriba, valida `/predict` usando las categorías en mayúsculas esperadas por el modelo:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Usage_kWh": 1000,
        "Lagging_Current_Reactive.Power_kVarh": 50,
        "Leading_Current_Reactive_Power_kVarh": 20,
        "CO2(tCO2)": 5,
        "Lagging_Current_Power_Factor": 0.95,
        "Leading_Current_Power_Factor": 0.9,
        "NSM": 2000,
        "mixed_type_col": 1,
        "WeekStatus": "WEEKDAY",
        "Day_of_week": "MONDAY"
      }'
```

- Construir localmente:

  ```bash
  docker build -t ml-service:latest .
  ```

- Ejecutar la API (el puerto 8000 del host queda enlazado al contenedor):

  ```bash
  docker run --rm -p 8000:8000 ml-service:latest
  ```

  Opcionalmente puedes sobreescribir la ruta del modelo con `-e MLOPS_MODEL_PATH=/path/custom`.

- Publicar en DockerHub usando tags versionadas (adapta `DOCKERHUB_USER` a tu cuenta):

  ```bash
  export DOCKERHUB_USER=<tu_usuario>
  docker tag ml-service:latest ${DOCKERHUB_USER}/ml-service:1.0.0
  docker push ${DOCKERHUB_USER}/ml-service:1.0.0
  docker tag ml-service:latest ${DOCKERHUB_USER}/ml-service:latest
  docker push ${DOCKERHUB_USER}/ml-service:latest
  ```

  Mantén el tag semántico (`1.0.0`) sincronizado con `MODEL_VERSION` en `src/api/main.py` y usa `latest` como alias para el último release estable.

### Automatizar build/push/test con Make

Define tu usuario de Docker Hub (y asegúrate de haber ejecutado `docker login`) antes de lanzar el flujo:

```bash
export DOCKER_REGISTRY_USER=ivan2909  # ajusta a tu usuario
make docker-release
```

Este objetivo construye la imagen, la etiqueta como `${DOCKER_REGISTRY_USER}/ml-service:1.0.0`, la publica, la vuelve a descargar para validar, levanta un contenedor temporal y ejecuta los endpoints `/health` y `/predict` para confirmar que el modelo responde correctamente. `make pipeline-deploy` lo invoca automáticamente al final para que la entrega a producción siempre genere y verifique la imagen.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── dvc.yaml      # yaml file to run make_dataset and training
    ├── params.yaml     # yaml file for trainign params
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



## Commits
Para los commit por favor usar la siguiente estructura en el comentario:

NombreTeamMember: Qué se realizó (agregó|actualizó|eliminó|etc.) en los archivos (archivo.txt, archivo.csv, etc)

P.e. **"Ivan: Se crean directorios y se reubican datasets steel_energy_modified.csv y steel_energy_original.csv"**

---

## Configuración de DVC con S3

Este proyecto usa DVC para versionar datos y un bucket S3 llamado **mlops-steel-energy-dvc-storage** como almacenamiento remoto.

### 1. Configura tus credenciales de AWS

Debes tener tus credenciales de AWS (Access Key ID y Secret Access Key) configuradas en tu sistema.  
Ejecuta en tu terminal:

```bash
aws configure
```
Ingresa:
- AWS Access Key ID: [tu clave]
- AWS Secret Access Key: [tu clave secreta]
- Default region name: us-east-1
- Default output format: json

O bien, puedes exportar las variables en tu `~/.zshrc` (Mac/Linux) o configurarlas en el entorno de Windows.

### 2. Sincroniza tu repo y datos

Después de configurar, ejecuta:

```bash
git pull
dvc pull
```

Esto traerá la configuración y descargará los datos desde S3.

### 3. Uso normal de DVC

- Para subir datos:
  ```bash
  dvc repro
  git commit -m "TuNombre: Descripción del cambio"
  dvc push
  git push
  ```
- Para descargar datos:
  ```bash
  dvc pull
  ```

---

## Preguntas frecuentes

- **¿Cómo sé si tengo acceso al S3?**  
  Prueba:
  ```bash
  aws s3 ls s3://mlops-dvc-storage-ivan/data   
  ```
  Si ves archivos/carpetas, tienes acceso.

- **¿Qué hago si no tengo credenciales?**  
  Solicítalas al administrador del bucket (quien creó el S3).

- **¿Puedo usar MLflow?**  
  Sí, puedes trackear tus experimentos y modelos. Si necesitas ayuda, consulta con el equipo.

---

## Notas finales

- **No subas archivos grandes (.csv, modelos) directamente a Git. Usa DVC y S3.**
- **No compartas tus credenciales AWS.**
- Si tienes problemas con DVC o acceso, consulta esta guía o pregunta al equipo.

---

## MLflow: Experimentos y Registro de Modelos

Esta sección resume cómo crear/usar experimentos con MLflow, tanto en modo local (por archivo) como con servidor (UI + Model Registry).

### Modo 1: Local por archivo (sin servidor)
- Configura `.env` (o variables de entorno):
  - `MLFLOW_TRACKING_URI=file:./mlruns`
  - `MLFLOW_EXPERIMENT=steel_energy`
  - `MLFLOW_REGISTER_IN_REGISTRY=false`
- Entrena y registra corridas en `./mlruns`:
  - `make data && make train`
- Abrir la UI para explorar experimentos y corridas:
  - `make mlflow-ui`
  - Abre `http://localhost:5000`
- Crear un experimento explícitamente (opcional):
  - `mlflow experiments create -n steel_energy`
  - Nota: el código ya crea el experimento si no existe (`mlflow.set_experiment`).

### Modo 2: Servidor MLflow (UI + Model Registry)
- Inicia el servidor (artefactos locales, con proxy de artefactos):
  - `make mlflow-server-local-modern`
- Configura `.env` (o exporta en shell):
  - `export MLFLOW_TRACKING_URI=http://localhost:5000`
  - `export MLFLOW_EXPERIMENT=steel_energy`
  - `export MLFLOW_REGISTER_IN_REGISTRY=true`
  - `export MLFLOW_REGISTERED_MODEL_NAME=SteelEnergyRF`
- Crea el experimento (opcional, también se crea automáticamente desde el código):
  - Desde la UI: botón “Create” en la vista de Experimentos
  - O por CLI: `mlflow experiments create -n steel_energy`
- Entrena y registra corridas/modelos:
  - `make data && make train`
- Barrido de hiperparámetros (múltiples corridas):
  - `make sweep`
- Promoción/rollback de modelos:
  - En la UI, sección “Models” → elige versión → “Transition to Staging/Production” o vuelve a una versión anterior.

### Consejos y resolución de problemas
- Ver a qué servidor apunta el cliente:
  - `python3 -c "import mlflow; print(mlflow.get_tracking_uri())"`
- Error 500 al subir artefactos con servidor:
  - Asegura usar el target “modern” que activa `--serve-artifacts`.
  - Verifica que `MLFLOW_TRACKING_URI` apunte a `http://localhost:5000` y que el server esté levantado.
- Si trabajas en modo local (file store) y no ves corridas en la UI del servidor:
  - Usa `make mlflow-ui` para leer `./mlruns` o cambia `MLFLOW_TRACKING_URI` a `http://localhost:5000` y levanta el server.
