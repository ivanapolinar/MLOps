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
    ├── setup_dvc_s3.sh      # Script automático para Mac/Linux
    ├── setup_dvc_s3.bat     # Script automático para Windows
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

### 2. Configura el remote de DVC

#### **En MacOS/Linux:**
Ejecuta el script de setup:

```bash
./setup_dvc_s3.sh
```

#### **En Windows:**
Ejecuta el script de setup (doble clic o desde CMD):

```bat
setup_dvc_s3.bat
```

Esto configurará automáticamente el remote de DVC para el bucket S3.

### 3. Sincroniza tu repo y datos

Después de configurar, ejecuta:

```bash
git pull
dvc pull
```

Esto traerá la configuración y descargará los datos desde S3.

### 4. Uso normal de DVC

- Para subir datos:
  ```bash
  dvc add data/tu_archivo.csv
  git add data/tu_archivo.csv.dvc .gitignore
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
  aws s3 ls s3://mlops-steel-energy-dvc-storage/data   
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
