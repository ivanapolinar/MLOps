"""
mlflow_test.py
--------------
Prueba básica de integración MLflow
Fase 2 - Proyecto MLOps Equipo 55
"""

import mlflow
import random

# Configurar el tracking local (creará una carpeta "mlruns" en el proyecto)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("steel_energy_baseline_test")

# Iniciar un experimento
with mlflow.start_run(run_name="prueba_local"):
    # Registrar parámetros y métricas de prueba
    mlflow.log_param("alpha", 0.1)
    mlflow.log_param("l1_ratio", 0.5)

    accuracy = random.uniform(0.8, 0.95)
    loss = random.uniform(0.1, 0.3)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", loss)

    print("Experimento registrado en MLflow con éxito")
    print(f"Accuracy: {accuracy:.3f} | Loss: {loss:.3f}")
