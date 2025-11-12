from __future__ import annotations

import os
import logging
import click
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from dotenv import load_dotenv
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==================== Helper Classes ==================== #


# Class representing the data repository, in this case, the data file
class DataRepository:
    """
    Purpose:
        - Read csv files in a local or remote repository
    """

    def load(self, input_path: str) -> pd.DataFrame:
        """
        Inputs:
            - input_path: File path for csv file
        Outputs:
            - DataFrame: Pandas dataframe with csv data
        Purpose:
            - Load data
        """
        return pd.read_csv(input_path)


# Class helping the data splitting based on a target variable
class DataSplitter:
    """
    Purpose:
        - Split data in train and test.
    """

    def __init__(
        self,
        target: str = "Load_Type",
        cols_to_drop: list[str] = ["date"],
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Inputs:
            - target: Variable to predict, default Load_Type
            - cols_to_drop: Columns to drop from the dataframe
            - test_size: Percentage representing the size for test data
            - random_state: Random seed for train test splitting, default 42
        Purpose:
            - Store data splitting attributes
        """
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.cols_to_drop = cols_to_drop

    def split(self, df: pd.DataFrame) -> list:
        """
        Inputs:
            - df: Pandas dataframe with the data to split
        Outputs:
            - List representing train test split
        Purpose:
            - Performing train test split
        """
        y = df[self.target]
        X = df.drop(columns=[self.target] + self.cols_to_drop)
        return train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )


# Class that helps making the preprocessing pipeline
class PreprocessingBuilder:
    """
    Purpose:
        - Handle the creation of preprocessing pipeline
    """

    def build(self, X: pd.DataFrame):
        """
        Inputs:
            - X: Features dataframe
        Outputs:
            - preprocessing: Column transformer
            - cat_cols: List of categorical columns
            - num_cols: List of numerical columns
        Purpose:
            - Build preprocessing pipeline
        """
        cat_cols = list(X.select_dtypes("object").columns)
        num_cols = list(X.select_dtypes("number").columns)
        preprocessing = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(), cat_cols),
            ]
        )
        return preprocessing, cat_cols, num_cols


# Class that handles model training
class ModelTrainer:
    """
    Purpose:
        - Handle model training
    """

    def __init__(self, random_state: int = 42, params: dict | None = None):
        """
        Inputs:
            - random_state: Random seed for model training, default 42
            - params: Dictionary containing hyperparameter for model training
        Purpose:
            - Store modeling attributes
        """
        self.random_state = random_state
        self.params = params

    def build_pipeline(self, preprocessing: ColumnTransformer) -> Pipeline:
        """
        Inputs:
            - preprocessing: Column transformer with features transformation
        Outputs:
            - Pipeline with two steps, preprocessing and model
        Purpose:
            - Build training pipeline
        """
        rf = RandomForestClassifier(
            random_state=self.random_state,
            **(self.params or {})
        )
        return make_pipeline(preprocessing, rf)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocessing: ColumnTransformer,
    ) -> Pipeline:
        """
        Inputs:
            - X_train: Pandas dataframe with features
            - y_train: Pandas series with target
            - preprocessing: Column transformer with features transformation
        Outputs:
            - model: Trained modeling pipeline
        Purpose:
            - Build and train modeling pipeline
        """
        model = self.build_pipeline(preprocessing)
        model.fit(X_train, y_train)
        return model


# Class that handles evaluation and image generation
class ModelEvaluator:
    """
    Purpose:
        - Model evaluation and image generation
    """

    def evaluate(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        figures_dir: str,
        name: str = "base",
    ):
        """
        Inputs:
            - model: Model that will be use for evaluation
            - X_test: Pandas dataframe with test features
            - y_test: Pandas series with test target
            - figures_dir: Directory to store evaluation images
            - name: Descriptive name of the evalution run, default base
        Outputs:
            - acc: Accuracy of the evaluation
            - fig_path: Path of the confusion matrix image
            - report: Classification report
        Purpose:
            - Evaluate the model performance
        """
        logger = logging.getLogger(__name__)
        y_pred = model.predict(X_test)
        # Evitar UndefinedMetricWarning cuando alguna clase no es predicha
        report = classification_report(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        logger.info("Classification Report (%s):\n%s", name, report)
        logger.info("Accuracy (%s): %.6f", name, acc)

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues"
        )
        plt.title(f"Confusion Matrix - RandomForest ({name})")
        plt.xlabel("Predictions")
        plt.ylabel("Real")
        plt.tight_layout()
        fig_path = f"{figures_dir}/confusion_matrix_{name}.png"
        plt.savefig(fig_path)
        plt.close()

        return acc, fig_path, report


# Class that handles hyperparameter tunning
class HyperparameterTuner:
    """
    Purpose:
        - Handle hyperparameter tuning
    """

    def __init__(
        self,
        param_dist: dict | None = None,
        n_iter: int | None = None,
        cv: int | None = None,
    ):
        """
        Inputs:
            - param_dist: Dictionary with hyperparameters distributions
        Purpose:
            - Store hyperparameters attributes
        """
        self.param_dist = (
            param_dist
            if param_dist is not None
            else {
                "randomforestclassifier__n_estimators": [100, 200, 400],
                "randomforestclassifier__max_depth": [10, 20, None],
                "randomforestclassifier__min_samples_split": [2, 5, 10],
                "randomforestclassifier__min_samples_leaf": [1, 2, 4],
                "randomforestclassifier__max_features": ["sqrt", "log2"],
            }
        )
        # Permitir configurar por variables de entorno sin romper defaults
        try:
            self.n_iter = (
                int(os.getenv("TUNE_N_ITER", "30"))
                if n_iter is None else n_iter
            )
        except Exception:
            self.n_iter = 30
        try:
            self.cv = (
                int(os.getenv("TUNE_CV", "3"))
                if cv is None else cv
            )
        except Exception:
            self.cv = 3

    def tune(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42,
    ):
        """
        Inputs:
            - model: Pipeline with modeling steps
            - X_train: Pandas dataframe with train features
            - y_train: Pandas series with train target
            - random_state: Random seed for hyperparameter tunning, default 42
        Outputs:
            - rf_rscv.best_estimator_: Best model
            - rf_rscv.best_params_: Best model hyperparameters
        Purpose:
            - Perform random search hyperparameter tunning
        """
        rf_rscv = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring="accuracy",
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )
        rf_rscv.fit(X_train, y_train)
        print("Best Hyperparameters:")
        print(rf_rscv.best_params_)

        return rf_rscv.best_estimator_, rf_rscv.best_params_


# Class that exports feature importances and graphs them
class FeatureImportanceExporter:
    """
    Purpose:
        - Obtain most important features and store graphical representations
    """

    def export(self, model: Pipeline, figures_dir: str):
        """
        Inputs:
            - model: Trained pipeline with modeling steps
            - figures_dir: Directory to store the generated image
        Outputs:
            - fi_path: Feature importance data path
            - top_feat_path: Top 15 most important features image path
        Purpose:
            - Visualize the most important features
        """
        rf_final = model.named_steps["randomforestclassifier"]
        ct = model.named_steps["columntransformer"]
        feature_names = ct.get_feature_names_out()

        importances = rf_final.feature_importances_
        feature_importances = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        fi_path = f"{figures_dir}/feature_importances.csv"
        feature_importances.head(15).to_csv(fi_path, index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importances.head(15),
            x="Importance",
            y="Feature"
        )
        plt.title("Top 15 Most Important Variables - RandomForest")
        plt.xlabel("Importance")
        plt.ylabel("Variable")
        plt.tight_layout()
        top_feat_path = f"{figures_dir}/top_features.png"
        plt.savefig(top_feat_path)
        plt.close()

        return fi_path, top_feat_path


class ModelPersister:
    """
    Purpose:
        - Save the trained model
    """

    def save(self, model: Pipeline, model_path: str):
        """
        Inputs:
            - model: Trained pipeline with modeling steps
            - model_path: Path to store the model
        Purpose:
            - Save the trained model
        """
        # Comprimir el modelo para reducir tamaño en disco
        joblib.dump(model, model_path, compress=3)
        print(f"Model saved in {model_path}")


# ==================== Orchestrator Class ==================== #


# Orchestrates all experiment run
class ExperimentRunner:
    """
    Purpose:
        - Handle a complete experiment run
    """

    def __init__(self):
        """
        Purpose:
            - Use classes as attributes for easy accessing
        """
        self.data_repo = DataRepository()
        self.splitter = DataSplitter()
        self.pre_builder = PreprocessingBuilder()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.tuner = HyperparameterTuner()
        self.fi_exporter = FeatureImportanceExporter()
        self.persister = ModelPersister()

    def run(self, input_path: str, model_path: str, figures_dir: str):
        """
        Inputs:
            - input_path: Data file path
            - model_path: Path to store the model
            - figures_dir: Directory to store generated images
        Purpose:
            - Handle a complete mlflow experiment run
        """
        # Make dirs for outputs
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        # Read Data
        df = self.data_repo.load(input_path)

        # Split data
        X_train, X_test, y_train, y_test = self.splitter.split(df)

        # Preprocessing pipeline and feature cols
        preprocessing, num_cols, cat_cols = self.pre_builder.build(X_train)

        # Start run with mlflow
        mlflow.set_experiment("rf-experiment")
        with mlflow.start_run():
            # Base training
            base_params = {}
            base_trainer = ModelTrainer(params=base_params)
            rf_model = base_trainer.fit(X_train, y_train, preprocessing)
            mlflow.log_params(base_params)

            # Base Metric logging
            acc_base, fig_base, report_base = self.evaluator.evaluate(
                rf_model, X_test, y_test, figures_dir, name="base"
            )
            mlflow.log_metric("base_accuracy", acc_base)
            mlflow.log_artifact(fig_base)

            # Hyperparameter tunning
            best_model, best_params = self.tuner.tune(
                rf_model,
                X_train,
                y_train
            )
            mlflow.log_params(best_params)

            # Best Metric Logging
            acc_opt, fig_opt, report_opt = self.evaluator.evaluate(
                best_model, X_test, y_test, figures_dir, name="optimized"
            )
            mlflow.log_metric("optimized_accuracy", acc_opt)
            mlflow.log_artifact(fig_opt)

            # Calculate feature importances
            fi_path, top_feat_path = self.fi_exporter.export(
                best_model,
                figures_dir
            )
            mlflow.log_artifact(fi_path)
            mlflow.log_artifact(top_feat_path)

            # Persist model
            self.persister.save(best_model, model_path)

            # Log example model
            # Evitar warning de esquema de MLflow con columnas enteras sin NAs
            X_example = X_test[:2].copy()
            int_cols = list(X_example.select_dtypes(include="integer").columns)
            if int_cols:
                X_example[int_cols] = X_example[int_cols].astype("float64")
            mlflow.sklearn.log_model(
                best_model,
                name="model",
                input_example=X_example,
                signature=mlflow.models.infer_signature(
                    X_example, best_model.predict(X_example)
                ),
            )


# ==================== Envoltorios (API funcional) ==================== #


def load_data(input_path: str) -> pd.DataFrame:
    """Carga un dataset CSV usando DataRepository.

    Parámetros:
        input_path: Ruta al archivo CSV de entrada.

    Retorna:
        DataFrame con los datos cargados.
    """
    return DataRepository().load(input_path)


def split_data(df: pd.DataFrame):
    """Divide el DataFrame en train/test usando la configuración por defecto.

    Retorna:
        X_train, X_test, y_train, y_test
    """
    return DataSplitter().split(df)


def build_preprocessing(X: pd.DataFrame):
    """Construye el preprocesamiento y retorna tuplas.

    Retorna:
        (preprocessing, num_cols, cat_cols)

    Nota:
        Internamente PreprocessingBuilder.build retorna
        (preprocessing, cat_cols, num_cols), pero por compatibilidad
        devolvemos (preprocessing, num_cols, cat_cols).
    """
    preprocessing, cat_cols, num_cols = PreprocessingBuilder().build(X)
    return preprocessing, num_cols, cat_cols


def train_base_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing: ColumnTransformer,
    params: dict | None = None,
):
    """Entrena pipeline RF con preprocesamiento y parámetros."""
    return ModelTrainer(params=params).fit(X_train, y_train, preprocessing)


def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: str,
    name: str = "base",
):
    """Evalúa el modelo y genera métricas y figura de confusión."""
    return ModelEvaluator().evaluate(
        model,
        X_test,
        y_test,
        figures_dir,
        name=name,
    )


def hyperparameter_tuning(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Ejecuta RandomizedSearchCV y retorna mejor modelo y parámetros."""
    return HyperparameterTuner().tune(model, X_train, y_train)


def save_feature_importance(
    model: Pipeline,
    num_cols,
    cat_cols,
    figures_dir: str,
):
    """Exporta importancias.

    num_cols/cat_cols se mantienen por compatibilidad.
    """
    return FeatureImportanceExporter().export(model, figures_dir)


def save_model(model: Pipeline, model_path: str):
    """Persiste el modelo entrenado en disco."""
    return ModelPersister().save(model, model_path)


# ==================== CLI ==================== #


# Orchestrates all model and experiment run
@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path())
@click.argument("figures_dir", type=click.Path())
def main(input_path, model_path, figures_dir):
    """
    Inputs:
        - input_path: Data file path
        - model_path: Path to store the model
        - figures_dir: Directory to store generated images
    Purpose:
        - Call a experiment run
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Cargar variables de entorno desde .env si existe
    try:
        load_dotenv(override=True)
    except Exception:
        pass

    # Configuración de MLflow desde variables de entorno (si existen)
    # MLFLOW_TRACKING_URI: por ejemplo, http://localhost:5000
    # o una ruta local tipo ./mlruns
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # MLFLOW_EXPERIMENT: nombre del experimento
    # (por defecto 'steel_energy')
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "steel_energy")
    try:
        mlflow.set_experiment(experiment_name)
    except Exception:
        pass
    df = load_data(input_path)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessing, num_cols, cat_cols = build_preprocessing(X_train)

    # MLflow tracking
    with mlflow.start_run():
        # Entrenamiento base y log de parámetros
        base_params = {}
        rf_model = train_base_model(
            X_train, y_train,
            preprocessing,
            base_params
        )
        mlflow.log_params(base_params)
        acc_base, fig_base, report_base = evaluate_model(
            rf_model,
            X_test,
            y_test,
            figures_dir,
            name="base"
        )
        mlflow.log_metric("base_accuracy", acc_base)
        mlflow.log_artifact(fig_base)

        # Hiperparámetros
        best_model, best_params = hyperparameter_tuning(
            rf_model,
            X_train,
            y_train
        )
        mlflow.log_params(best_params)
        acc_opt, fig_opt, report_opt = evaluate_model(
            best_model,
            X_test,
            y_test,
            figures_dir,
            name="optimized"
        )
        mlflow.log_metric("optimized_accuracy", acc_opt)
        mlflow.log_artifact(fig_opt)

        # Importancia de variables
        fi_path, top_feat_path = save_feature_importance(
            best_model,
            num_cols,
            cat_cols,
            figures_dir
        )
        mlflow.log_artifact(fi_path)
        mlflow.log_artifact(top_feat_path)

        # Guardar modelo local
        save_model(best_model, model_path)

        # Loguear modelo en MLflow;
        # registrar en el Model Registry solo si está habilitado
        # Evitar warning de esquema de MLflow con columnas enteras sin NAs
        X_example = X_test[:2].copy()
        int_cols = list(X_example.select_dtypes(include="integer").columns)
        if int_cols:
            X_example[int_cols] = X_example[int_cols].astype("float64")
        register_flag = (
            os.getenv("MLFLOW_REGISTER_IN_REGISTRY", "false").lower()
            == "true"
        )
        tracking_uri = mlflow.get_tracking_uri() or ""
        can_register = register_flag and tracking_uri.startswith("http")
        if can_register:
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                input_example=X_example,
                signature=mlflow.models.infer_signature(
                    X_example,
                    best_model.predict(X_example)
                ),
                registered_model_name=os.getenv(
                    "MLFLOW_REGISTERED_MODEL_NAME",
                    "SteelEnergyRF",
                )
            )
        else:
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                input_example=X_example,
                signature=mlflow.models.infer_signature(
                    X_example,
                    best_model.predict(X_example)
                ),
            )


if __name__ == "__main__":
    main()
