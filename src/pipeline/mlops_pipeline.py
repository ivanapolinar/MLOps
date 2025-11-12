"""Pipeline MLOps integral para el flujo de modelado.

Esta clase integra las etapas principales del proyecto:
- Extracción y limpieza de datos.
- Preprocesamiento y partición train/test.
- Entrenamiento base y optimizado (tuning).
- Evaluación y exportación de artefactos (figuras, importancias).
- Persistencia del modelo final y logging en MLflow.
- Predicción por lote (batch) y generación de métricas.

La implementación reutiliza las utilidades existentes en los módulos
`src/data/make_dataset.py`, `src/models/train_model.py` y
`src/models/predict_model.py`, manteniendo compatibilidad con el
código y pruebas actuales.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd

from src.data import make_dataset as md
from src.models import train_model as tm
from src.models import predict_model as pm


class MLOpsPipeline:
    """Orquestador del flujo MLOps de punta a punta.

    Parámetros opcionales permiten configurar MLflow desde código; si no se
    proveen, se leen las variables de entorno ya soportadas por los módulos.
    """

    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment: Optional[str] = None,
        register_in_registry: bool = False,
        registered_model_name: str = "SteelEnergyRF",
    ) -> None:
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment = mlflow_experiment or "steel_energy"
        self.register_in_registry = register_in_registry
        self.registered_model_name = registered_model_name

        # Configuración inicial de MLflow si se solicita
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        if self.mlflow_experiment:
            try:
                mlflow.set_experiment(self.mlflow_experiment)
            except Exception:
                # Evitar fallos si la creación del experimento no es posible
                pass

    # ==================== Datos ==================== #

    def process_raw_to_clean(
        self, input_filepath: str, output_filepath: str
    ) -> pd.DataFrame:
        """Procesa datos crudos → limpios y los guarda.

        Retorna el DataFrame limpio para pasos posteriores.
        """
        df_raw = md.load_data(input_filepath)
        df_cleaned, num_cols, object_cols, date_cols = md.clean_data(df_raw)
        df_imputed = md.impute_data(df_cleaned, num_cols, object_cols, date_cols)
        df_final = md.drop_null_targets(df_imputed, target_col="Load_Type")
        md.save_data(df_final, output_filepath)
        return df_final

    # ==================== Entrenamiento ==================== #

    def split(self, df: pd.DataFrame):
        """Partición train/test usando la configuración por defecto."""
        return tm.split_data(df)

    def build_preprocessing(self, X_train: pd.DataFrame):
        """Construye el preprocesamiento y retorna (preproc, num_cols, cat_cols)."""
        return tm.build_preprocessing(X_train)

    def train_base(
        self,
        X_train,
        y_train,
        preprocessing,
        params: Optional[dict] = None,
    ):
        """Entrena un modelo base usando RandomForest + preprocesamiento."""
        return tm.train_base_model(X_train, y_train, preprocessing, params or {})

    def tune_hyperparams(self, model, X_train, y_train):
        """Ajuste de hiperparámetros (RandomizedSearchCV)."""
        return tm.hyperparameter_tuning(model, X_train, y_train)

    def evaluate(self, model, X_test, y_test, figures_dir: str, name: str):
        """Evalúa el modelo y genera figura de matriz de confusión."""
        return tm.evaluate_model(model, X_test, y_test, figures_dir, name=name)

    def export_feature_importance(
        self, model, num_cols, cat_cols, figures_dir: str
    ):
        """Exporta importancias y gráfico top-15 a figures_dir."""
        return tm.save_feature_importance(model, num_cols, cat_cols, figures_dir)

    def persist_model(self, model, model_path: str) -> None:
        """Guarda el modelo a disco (joblib)."""
        tm.save_model(model, model_path)

    # ==================== Predicción ==================== #

    def batch_predict(
        self,
        input_path: str,
        model_path: str,
        predictions_path: str,
        metrics_path: Optional[str] = None,
        figures_dir: Optional[str] = None,
    ) -> None:
        """Realiza inferencia por lote y exporta artefactos.

        Reutiliza el módulo `predict_model` manteniendo el formato actual
        de salidas (CSV de predicciones, JSON de métricas y figura opcional).
        """
        pm.ensure_dirs(predictions_path, metrics_path, figures_dir)
        df = pm.load_dataset(input_path)
        model = (
            mlflow.sklearn.load_model(model_path)
            if model_path.endswith("mlflow")
            else None
        )
        # Si el path no es un modelo logueado en MLflow, usar joblib
        if model is None:
            import joblib

            model = joblib.load(model_path)
        X = pm.prepare_features(df)
        y_true = df["Load_Type"] if "Load_Type" in df.columns else None
        y_pred = model.predict(X)
        out = df.copy()
        out["Prediction"] = y_pred
        pm.attach_probabilities(model, X, out)
        out.to_csv(predictions_path, index=False)
        pm.maybe_metrics_and_figures(y_true, y_pred, metrics_path, figures_dir)

    # ==================== Orquestación de entrenamiento ==================== #

    def run_training_pipeline(
        self,
        input_clean_csv: str,
        model_path: str,
        figures_dir: str,
    ) -> Tuple[float, str, str]:
        """Ejecuta entrenamiento base + tuning y exporta artefactos.

        Retorna una tupla con:
        (mejor_accuracy, path_feature_importances_csv, path_top_features_png)
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        df = tm.load_data(input_clean_csv)
        X_train, X_test, y_train, y_test = self.split(df)
        preprocessing, num_cols, cat_cols = self.build_preprocessing(X_train)

        with mlflow.start_run():
            # Entrenamiento base
            base_params: dict = {}
            base_model = self.train_base(
                X_train, y_train, preprocessing, base_params
            )
            mlflow.log_params(base_params)
            acc_base, fig_base, _ = self.evaluate(
                base_model, X_test, y_test, figures_dir, name="base"
            )
            mlflow.log_metric("base_accuracy", acc_base)
            if fig_base:
                mlflow.log_artifact(fig_base)

            # Tuning
            best_model, best_params = self.tune_hyperparams(
                base_model, X_train, y_train
            )
            mlflow.log_params(best_params)
            acc_opt, fig_opt, _ = self.evaluate(
                best_model, X_test, y_test, figures_dir, name="optimized"
            )
            mlflow.log_metric("optimized_accuracy", acc_opt)
            if fig_opt:
                mlflow.log_artifact(fig_opt)

            # Importancias
            fi_csv, top_png = self.export_feature_importance(
                best_model, num_cols, cat_cols, figures_dir
            )
            mlflow.log_artifact(fi_csv)
            mlflow.log_artifact(top_png)

            # Guardar modelo final para PROD
            self.persist_model(best_model, model_path)

            # Loguear modelo con ejemplo y firma; registrar si procede
            X_example = X_test[:2].copy()
            int_cols = list(
                X_example.select_dtypes(include="integer").columns
            )
            if int_cols:
                X_example[int_cols] = X_example[int_cols].astype("float64")

            tracking_uri = mlflow.get_tracking_uri() or ""
            can_register = (
                self.register_in_registry and tracking_uri.startswith("http")
            )
            if can_register:
                mlflow.sklearn.log_model(
                    best_model,
                    name="model",
                    input_example=X_example,
                    signature=mlflow.models.infer_signature(
                        X_example, best_model.predict(X_example)
                    ),
                    registered_model_name=self.registered_model_name,
                )
            else:
                mlflow.sklearn.log_model(
                    best_model,
                    name="model",
                    input_example=X_example,
                    signature=mlflow.models.infer_signature(
                        X_example, best_model.predict(X_example)
                    ),
                )

        return acc_opt, fi_csv, top_png
