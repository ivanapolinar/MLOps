.PHONY: clean data lint sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALES                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# Cargar variables desde .env si existe
ifneq (,$(wildcard .env))
include .env
export $(shell sed -n 's/^[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=.*/\1/p' .env)
endif
BUCKET = mlops-dvc-storage-ivan/data
PROFILE = default
PROJECT_NAME = steel_energy
PYTHON_INTERPRETER = python3
REQ_CACHE_DIR := .cache
REQ_STAMP := $(REQ_CACHE_DIR)/requirements.stamp

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMANDOS                                                                       #
#################################################################################

## Instalar dependencias de Python (solo si cambian)
requirements: $(REQ_STAMP)

$(REQ_STAMP): requirements.txt | $(REQ_CACHE_DIR) test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	@touch $(REQ_STAMP)

$(REQ_CACHE_DIR):
	@mkdir -p $(REQ_CACHE_DIR)

## Generar dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/steel_energy_modified.csv data/clean/steel_energy_clean.csv

train: requirements
	$(PYTHON_INTERPRETER) src/models/train_model.py data/clean/steel_energy_clean.csv models/best_rf_model.joblib reports/figures

visualization: requirements
	$(PYTHON_INTERPRETER) src/visualization/visualize.py

# Ejecuta inferencia por lote con el modelo entrenado
# Entrada: data/clean/steel_energy_clean.csv
# Salidas: reports/predictions.csv, reports/metrics.json y figura en reports/figures
predict: requirements
	$(PYTHON_INTERPRETER) src/models/predict_model.py data/clean/steel_energy_clean.csv models/best_rf_model.joblib reports/predictions.csv reports/metrics.json reports/figures

## Ejecutar pruebas unitarias (gráficos sin interfaz)
test: requirements
	MPLBACKEND=Agg $(PYTHON_INTERPRETER) -m pytest -q

## Ejecutar solo pruebas de predict_model (rápido)
test-predict: requirements
	MPLBACKEND=Agg $(PYTHON_INTERPRETER) -m pytest -q tests/test_predict_model.py

## Eliminar archivos Python compilados
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Análisis estático con flake8
lint:
	flake8 src tests

## Levantar servidor MLflow con almacenamiento local (UI + Registry)
mlflow-server-local:
	mlflow server --backend-store-uri sqlite:///$(PROJECT_DIR)/mlruns.db \
	 --default-artifact-root $(PROJECT_DIR)/.mlflow_artifacts \
	 --host 0.0.0.0 --port 5000

## Levantar servidor MLflow con artefactos en S3 (requiere credenciales AWS)
mlflow-server-s3:
	mlflow server --backend-store-uri sqlite:///$(PROJECT_DIR)/mlruns.db \
	 --default-artifact-root s3://mlops-dvc-storage-ivan/mlflow-artifacts \
	 --host 0.0.0.0 --port 5000

## (Alternativa) Flags modernos si tu versión de MLflow los soporta
mlflow-server-local-modern:
	mlflow server --backend-store-uri sqlite:///$(PROJECT_DIR)/mlruns.db \
	 --serve-artifacts --artifacts-destination $(PROJECT_DIR)/.mlflow_artifacts \
	 --host 0.0.0.0 --port 5000

mlflow-server-s3-modern:
	mlflow server --backend-store-uri sqlite:///$(PROJECT_DIR)/mlruns.db \
	 --serve-artifacts --artifacts-destination s3://mlops-dvc-storage-ivan/mlflow-artifacts \
	 --host 0.0.0.0 --port 5000

## UI de MLflow para almacenamiento local por archivo
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

## Inicializar archivo .env con variables por defecto
env-init:
	@if [ -f .env ]; then echo ".env ya existe"; else cp .env.example .env && echo "Creado .env desde .env.example"; fi

## Ejecutar barrido de hiperparámetros y registrar en MLflow
sweep: requirements
	$(PYTHON_INTERPRETER) src/models/sweep.py data/clean/steel_energy_clean.csv models/best_rf_model_sweep.joblib reports/figures

## Subir datos a S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Descargar datos desde S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Versionar archivos con DVC
dvc_commit:
	@if [ -f data/clean/steel_energy_clean.csv ]; then \
		dvc add data/clean/steel_energy_clean.csv; \
	else \
		echo "Archivo data/clean/steel_energy_clean.csv NO existe"; \
	fi
	@if [ -f models/best_rf_model.joblib ]; then \
		dvc add models/best_rf_model.joblib; \
	else \
		echo "Archivo models/best_rf_model.joblib NO existe"; \
	fi
	git add data/clean/steel_energy_clean.csv.dvc models/best_rf_model.joblib.dvc .gitignore
	git commit -m "$(msg)"
	dvc push
	git push

dvc:
	@if [ -f data/clean/steel_energy_clean.csv ]; then \
		dvc add data/clean/steel_energy_clean.csv; \
	else \
		echo "Archivo data/clean/steel_energy_clean.csv NO existe"; \
	fi
	@if [ -f models/best_rf_model.joblib ]; then \
		dvc add models/best_rf_model.joblib; \
	else \
		echo "Archivo models/best_rf_model.joblib NO existe"; \
	fi

## Configurar entorno del intérprete de Python
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Se detectó conda, creando entorno conda."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> Nuevo entorno conda creado. Actívalo con:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Instalando virtualenvwrapper si no está instalado.\nAsegúrate de agregar estas líneas a tu archivo de inicio de la shell\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> Nuevo virtualenv creado. Actívalo con:\nworkon $(PROJECT_NAME)"
endif

## Probar que el entorno de Python está correctamente configurado
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Start the API
serve-api:
	uvicorn src.api.main:app --reload --port 8000

#################################################################################
# REGLAS DEL PROYECTO                                                            #
#################################################################################



#################################################################################
# Comandos auto-documentados                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspirado en <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# Explicación del script sed:
# /^##/:
# 	* guarda la línea en el hold space
# 	* limpia la línea actual
# 	* Bucle:
# 		* añade salto de línea + línea al hold space
# 		* ve a la siguiente línea
# 		* si la línea empieza con comentario de doc, elimina el prefijo y repite
# 	* elimina prerequisitos del target
# 	* añade el hold space (+ salto de línea) a la línea
# 	* reemplaza saltos de línea y comentarios por `---`
# 	* imprime la línea
# Notas: se requieren expresiones separadas porque las etiquetas no pueden ir separadas por punto y coma; ver <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Reglas disponibles:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
