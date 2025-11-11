# MLOps

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
  - `MLFLOW_TRACKING_URI=http://localhost:5000`
  - `MLFLOW_EXPERIMENT=steel_energy`
  - `MLFLOW_REGISTER_IN_REGISTRY=true`
  - `MLFLOW_REGISTERED_MODEL_NAME=SteelEnergyRF`
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
