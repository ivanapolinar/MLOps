# Proyecto MLOps | Fase 2 - Equipo 55

## 1. Contexto general

Este proyecto forma parte del desarrollo continuo del curso de **MLOps**, correspondiente a la *Fase 2 - Avance del Proyecto*.  
El trabajo continúa a partir de los resultados obtenidos en la Fase 1 (EDA, preprocesamiento y versionado de datos), enfocándose ahora en la **automatización, control de versiones y reproducibilidad de modelos** mediante buenas prácticas de ingeniería.

---

## 2. Rol de Mario

Mario fue responsable de los siguientes entregables dentro de la Fase 2:
- ✅ Implementación de **pruebas unitarias** para el módulo `MakeDataset`
- ✅ Desarrollo y ejecución de **experimentos con MLflow**
- ✅ **Documentación técnica del código** (estructuración y reproducibilidad)

---

## 3. Estructura general del proyecto

```
MLOps/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── clean/
│
├── notebooks/
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   └── models/
│       └── train_with_mlflow.py
│
├── tests/
│   └── test_make_dataset.py
│
└── README_Fase2.md
```

---

## 4. Pruebas unitarias (`MakeDataset`)

### 4.1 Objetivo
Validar que el proceso de generación de datasets (`make_dataset.py`) funciona correctamente, asegurando la creación de archivos intermedios y limpios sin errores ni datos vacíos.

### 4.2 Archivo de pruebas
`tests/test_make_dataset.py`

### 4.3 Ejecución
```bash
pytest -v tests/test_make_dataset.py
```

### 4.4 Resultados esperados
```
tests/test_make_dataset.py::test_make_dataset_creates_files PASSED
tests/test_make_dataset.py::test_generated_files_are_not_empty PASSED
tests/test_make_dataset.py::test_no_missing_values_in_clean_file PASSED
```

✔️ Todas las pruebas pasan exitosamente, validando la integridad de los datos y la correcta ejecución del flujo ETL inicial.

---

## 5. Experimentos con MLflow

### 5.1 Configuración
MLflow se ejecutó localmente:
```bash
mlflow ui
```
Panel disponible en: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 5.2 Experimento principal
`steel_energy_training`

Modelo base: **Ridge Regression**  
Dataset: `steel_energy_clean.csv`  
Variable objetivo: `Usage_kWh`

### 5.3 Métricas registradas

| α (alpha) | MAE | RMSE | R² |
|------------|------|------|------|
| 0.01 | **2.5400** | **4.2562** | **0.9841** |
| 0.1  | 2.7999 | 4.4899 | 0.9823 |
| 1.0  | 4.6981 | 6.9238 | 0.9578 |
| 10.0 | 6.6031 | 9.5701 | 0.9194 |

El mejor modelo se obtuvo con **α = 0.01**, alcanzando un **R² = 0.9841**.

### 5.4 Visualización y comparación
Todos los *runs* fueron registrados y pueden visualizarse en el panel de MLflow.  
Desde la interfaz, se compararon las métricas y curvas de rendimiento (MAE, RMSE, R²) entre diferentes configuraciones de `alpha`.

---

## 6. Registro y versionado de modelos

### 6.1 Registro automático en el Model Registry
El mejor modelo (`alpha = 0.01`) se registró bajo el nombre:

```
steel_energy_ridge_model
```

Creando versiones automáticas (v1, v2, v3, v4, v5) en el registro.

### 6.2 Verificación en MLflow
Panel → **Models → steel_energy_ridge_model**

Cada versión incluye:
- Artefactos (`model.pkl`, `conda.yaml`, `MLmodel`)
- Firma (`signature`)
- Input example
- Métricas asociadas

---

## 7. Evidencia visual

- 📊 *Comparación de runs:* Disponible en [http://127.0.0.1:5000](http://127.0.0.1:5000)
- 🧩 *Model Registry:* `steel_energy_ridge_model` con versiones hasta v5

---

## 8. Conclusiones

- Se implementaron pruebas unitarias que garantizan la reproducibilidad del pipeline.  
- MLflow permitió controlar versiones, métricas y registros de manera automatizada.  
- El modelo óptimo (α = 0.01) alcanzó un desempeño sobresaliente (**R² = 0.9841**).  
- El flujo completo es **reproducible y auditable**, cumpliendo con los lineamientos de la Fase 2 del curso.

---

## 9. Comandos clave de ejecución

```bash
# Crear datasets
python src/data/make_dataset.py

# Ejecutar pruebas unitarias
pytest -v tests/test_make_dataset.py

# Entrenar y registrar modelos con MLflow
python src/models/train_with_mlflow.py

# Iniciar el panel de experimentos
mlflow ui
```
