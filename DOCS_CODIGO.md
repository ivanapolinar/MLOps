# Proyecto MLOps | Fase 2 - Equipo 55
## Documentación técnica de código

---

## 1. Introducción

Este documento describe los módulos implementados por **Mario** durante la Fase 2 del proyecto **MLOps**, incluyendo su estructura, dependencias, flujo de ejecución y funciones principales.  

Los módulos documentados son:
- `src/data/make_dataset.py`
- `src/models/train_with_mlflow.py`

Ambos scripts fueron desarrollados bajo buenas prácticas de **reproducibilidad, trazabilidad y control de versiones**.

---

## 2. Módulo: `make_dataset.py`

### 2.1 Propósito
Automatizar el proceso de carga, limpieza y generación de datasets intermedios y finales a partir de los datos crudos.

### 2.2 Dependencias principales
```python
import os
import pandas as pd
from pathlib import Path
```

### 2.3 Flujo de ejecución
1. Entrada: `steel_energy_original.csv`
2. Limpieza: eliminación de nulos, duplicados, estandarización de nombres.
3. Salida:
   - `data/interim/steel_energy_interim.csv`
   - `data/clean/steel_energy_clean.csv`

### 2.4 Función principal
```python
def make_dataset(input_filepath, output_interim, output_clean):
    "Genera datasets intermedios y limpios a partir de un dataset crudo."
```

### 2.5 Ejecución
```bash
python src/data/make_dataset.py
```

---

## 3. Módulo: `train_with_mlflow.py`

### 3.1 Propósito
Entrenar modelos de regresión, aplicar **tracking de experimentos con MLflow**, y registrar los modelos en el **Model Registry**.

### 3.2 Dependencias principales
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
```

### 3.3 Flujo general
1. Carga del dataset limpio (`steel_energy_clean.csv`)
2. Separación en variables predictoras y objetivo (`Usage_kWh`)
3. Entrenamiento con distintos valores de `alpha`
4. Registro de métricas y modelo en MLflow

### 3.4 Métricas resultantes
| α | MAE | RMSE | R² |
|---|------|------|------|
| 0.01 | **2.5400** | **4.2562** | **0.9841** |
| 0.1 | 2.7999 | 4.4899 | 0.9823 |
| 1.0 | 4.6981 | 6.9238 | 0.9578 |
| 10.0 | 6.6031 | 9.5701 | 0.9194 |

### 3.5 Registro del modelo
```python
mlflow.register_model(model_uri=model_uri, name="steel_energy_ridge_model")
```

**Salida esperada:**
```
Registered model 'steel_energy_ridge_model' already exists.
Created version '5' of model 'steel_energy_ridge_model'.
```

### 3.6 Reproducibilidad
```bash
conda activate mlops_tests
python src/models/train_with_mlflow.py
mlflow ui
```

---

**Autor:** Mario  
**Equipo:** 55  
**Fecha:** Octubre 2025  
