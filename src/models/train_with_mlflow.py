"""
train_with_mlflow.py
--------------------
Entrenamiento del modelo con registro automático en MLflow
Fase 2 - Equipo 55
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
import numpy as np
import os

# ===============================
# 1. Configuración inicial
# ===============================

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("steel_energy_training")

# Cargar dataset limpio
data_path = "data/clean/steel_energy_clean.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"No se encontró el archivo {data_path}")

df = pd.read_csv(data_path)

# Suponiendo que la columna objetivo se llama 'energy_consumption'
# (ajústala según tu dataset)
target_col = "Usage_kWh"
if target_col not in df.columns:
    raise KeyError(f"No se encontró la columna objetivo '{target_col}' en el dataset.")

# ===============================
# 1. Preparación del dataset
# ===============================

y = df[target_col]

# Eliminar la columna objetivo y todas las columnas no numéricas
X = df.drop(columns=[target_col])

# Filtrar solo columnas numéricas
X = X.select_dtypes(include=["number"])

# Si hay categóricas relevantes, puedes codificarlas más adelante con one-hot encoding
X = X.astype("float64")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 2. Entrenamiento y logging
# ===============================

solver = "auto"

for alpha in [0.01, 0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"ridge_alpha_{alpha}"):
        mlflow.log_param("model_type", "Ridge Regression")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("solver", solver)

        model = Ridge(alpha=alpha, solver=solver)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        from mlflow.models.signature import infer_signature

        # === Registro del modelo con firma e input_example ===
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, model.predict(X_test))

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            signature=signature)


        print(f" Run alpha={alpha} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")


# =======================================================
# Registro del mejor modelo (alpha=0.01) en Model Registry
# =======================================================

        import mlflow

        # Nombre oficial del modelo
        model_name = "steel_energy_ridge_model"

        # Buscar el run con alpha=0.01 (mayor R²)
        best_run = None
        best_r2 = -1.0

        runs = mlflow.search_runs(experiment_names=["steel_energy_training"])

        for _, row in runs.iterrows():
            if row["params.alpha"] == "0.01" and row["metrics.R2"] > best_r2:
                best_run = row
                best_r2 = row["metrics.R2"]

        if best_run is not None:
            run_id = best_run["run_id"]
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri=model_uri, name=model_name)
            print(f" Modelo con alpha=0.01 registrado exitosamente como '{model_name}'")
        else:
            print(" No se encontró el modelo con alpha=0.01.")