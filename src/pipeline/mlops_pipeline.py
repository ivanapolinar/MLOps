from __future__ import annotations

import os
import joblib
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd

from src.data import make_dataset as md
from src.models import train_model as tm
from src.models import predict_model as pm


class MLOpsPipeline:
    """
    Pipeline MLOps completo:
    limpieza, split, entrenamiento, tuning,
    evaluación y guardado.
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

        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        if self.mlflow_experiment:
            try:
                mlflow.set_experiment(self.mlflow_experiment)
            except Exception:
                pass

    # ================== DATA ================== #

    def process_raw_to_clean(
        self,
        input_filepath: str,
        output_filepath: str,
    ) -> pd.DataFrame:
        df_raw = md.load_data(input_filepath)
        df_cleaned, num_cols, object_cols, date_cols = md.clean_data(df_raw)
        df_imputed = md.impute_data(
            df_cleaned,
            num_cols,
            object_cols,
            date_cols,
        )
        df_final = md.drop_null_targets(
            df_imputed,
            target_col="Load_Type",
        )
        md.save_data(df_final, output_filepath)
        return df_final

    # ================== SPLIT ================== #

    def split(self, df: pd.DataFrame):
        return tm.split_data(df)

    def build_preprocessing(self, X_train: pd.DataFrame):
        return tm.build_preprocessing(X_train)

    # ================== TRAIN ================== #

    def train_base(
        self,
        X_train,
        y_train,
        preprocessing,
        params: Optional[dict] = None,
    ):
        return tm.train_base_model(
            X_train,
            y_train,
            preprocessing,
            params or {},
        )

    def tune_hyperparams(self, model, X_train, y_train):
        return tm.hyperparameter_tuning(
            model,
            X_train,
            y_train,
        )

    def evaluate(
        self,
        model,
        X_test,
        y_test,
        figures_dir: str,
        name: str,
    ):
        return tm.evaluate_model(
            model,
            X_test,
            y_test,
            figures_dir,
            name=name,
        )

    def export_feature_importance(
        self,
        model,
        num_cols,
        cat_cols,
        figures_dir: str,
    ):
        return tm.save_feature_importance(
            model,
            num_cols,
            cat_cols,
            figures_dir,
        )

    def persist_model(self, model, model_path: str) -> None:
        tm.save_model(model, model_path)

    # ================== BATCH PREDICT ================== #

    def batch_predict(
        self,
        input_path: str,
        model_path: str,
        predictions_path: str,
        metrics_path: Optional[str] = None,
        figures_dir: Optional[str] = None,
    ) -> None:
        pm.ensure_dirs(
            predictions_path,
            metrics_path,
            figures_dir,
        )

        df = pm.load_dataset(input_path)

        model = (
            mlflow.sklearn.load_model(model_path)
            if model_path.endswith("mlflow")
            else joblib.load(model_path)
        )

        X = pm.prepare_features(df)
        y_true = (
            df["Load_Type"]
            if "Load_Type" in df.columns
            else None
        )

        y_pred = model.predict(X)

        out = df.copy()
        out["Prediction"] = y_pred
        pm.attach_probabilities(model, X, out)
        out.to_csv(predictions_path, index=False)

        pm.maybe_metrics_and_figures(
            y_true,
            y_pred,
            metrics_path,
            figures_dir,
        )

    # ================== TRAINING ORCHESTRATION ================== #

    def run_training_pipeline(
        self,
        input_clean_csv: str,
        model_path: str,
        figures_dir: str,
    ):
        # Create dirs
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        df = tm.load_data(input_clean_csv)
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.split(df)

        preprocessing, num_cols, cat_cols = self.build_preprocessing(
            X_train,
        )

        with mlflow.start_run():
            # Base
            base_params: dict = {}
            base_model = self.train_base(
                X_train,
                y_train,
                preprocessing,
                base_params,
            )
            mlflow.log_params(base_params)

            acc_base, fig_base, _ = self.evaluate(
                base_model,
                X_test,
                y_test,
                figures_dir,
                name="base",
            )
            mlflow.log_metric("base_accuracy", acc_base)

            if fig_base:
                mlflow.log_artifact(fig_base)

            # Tuning
            best_model, best_params = self.tune_hyperparams(
                base_model,
                X_train,
                y_train,
            )
            mlflow.log_params(best_params)

            acc_opt, fig_opt, _ = self.evaluate(
                best_model,
                X_test,
                y_test,
                figures_dir,
                name="optimized",
            )
            mlflow.log_metric("optimized_accuracy", acc_opt)

            if fig_opt:
                mlflow.log_artifact(fig_opt)

            # Feature importances
            fi_csv, top_png = self.export_feature_importance(
                best_model,
                num_cols,
                cat_cols,
                figures_dir,
            )
            mlflow.log_artifact(fi_csv)
            mlflow.log_artifact(top_png)

            # Save model
            self.persist_model(best_model, model_path)

            # Log MLflow model
            X_example = X_test[:2].copy()

            int_cols = list(
                X_example.select_dtypes(include="integer").columns
            )
            if int_cols:
                X_example[int_cols] = X_example[int_cols].astype(
                    "float64"
                )

            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                input_example=X_example,
                signature=mlflow.models.infer_signature(
                    X_example,
                    best_model.predict(X_example),
                ),
            )

        return acc_opt, fi_csv, top_png
