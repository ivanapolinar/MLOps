import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier

import joblib
import mlflow
import mlflow.sklearn




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

# Class helping the data splitting based on a target variable, default values represent our current experiment
class DataSplitter:
    """
    Purpose:
        - Split data in train and test.
    """
    def __init__(self, target: str = 'Load_Type', cols_to_drop: list[str] = ['date'], test_size: float = 0.2, random_state: int = 42):
        """
        Inputs:
            - target: Variable to predict, default Load_Type
            - cols_to_drop: Columns to drop from the dataframe, default column date
            - test_size: Percentage representing the size for test data, default 0.2
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
            - Performing train test split with the corresponding data splitting attributes
        """
        y = df[self.target]
        X = df.drop(columns=[self.target] + self.cols_to_drop)
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
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
            - preprocessing: Column transformer with standard scaler for numerical columns and one hot encoder for categorical columns
            - cat_cols: List of categorical columns
            - num_cols: List of numerical columns
        Purpose:
            - Build preprocessing pipeline
        """
        cat_cols = list(X.select_dtypes('object').columns)
        num_cols = list(X.select_dtypes('number').columns)
        preprocessing = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(), cat_cols)
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
            - params: Dictionary containing hyperparameter for model training, focus on Random forest classifier
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
            - Pipeline with two steps, one with features preprocessing and other with model training
        Purpose:
            - Build training pipeline
        """
        rf = RandomForestClassifier(random_state=self.random_state, **self.params)
        return make_pipeline(preprocessing, rf)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, preprocessing: ColumnTransformer) -> Pipeline:
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
    def evaluate(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, figures_dir: str, name: str = 'base'):
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
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        print(f"Classification Report ({name}):")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", acc)

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues"
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
    def __init__(self, param_dist: dict | None = None):
        """
        Inputs:
            - param_dist: Dictionary with hyperparameters distributions, default None
        Purpose:
            - Store hyperparameters attributes
        """
        self.param_dist = param_dist if param_dist is not None else {
            "randomforestclassifier__n_estimators": [100, 200, 400],
            "randomforestclassifier__max_depth": [10, 20, None],
            "randomforestclassifier__min_samples_split": [2, 5, 10],
            "randomforestclassifier__min_samples_leaf": [1, 2, 4],
            "randomforestclassifier__max_features": ["sqrt", "log2"]
        }

    def tune(self, model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
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
            - Perform random search hyperparameter tunning and return best available model with its hyperparameters
        """
        rf_rscv = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.param_dist,
            n_iter=30,
            cv=3,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1,
            random_state=42
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
        rf_final: RandomForestClassifier = model.named_steps["randomforestclassifier"]
        ct: ColumnTransformer = model.named_steps["columntransformer"]
        feature_names = ct.get_feature_names_out()

        importances = rf_final.feature_importances_
        feature_importances = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        fi_path = f"{figures_dir}/feature_importances.csv"
        feature_importances.head(15).to_csv(fi_path, index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importances.head(15), x="Importance", y="Feature")
        plt.title("Top 15 Most Important Variables - RandomForest")
        plt.xlabel("Importance")
        plt.ylabel("Variable")
        plt.tight_layout()
        top_feat_path = f"{figures_dir}/top_features.png"
        plt.savefig(top_feat_path)
        plt.close()

        return fi_path, top_feat_path

class ModelPersister():
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
        joblib.dump(model, model_path)
        print(f"Model saved in {model_path}")




# ==================== Orchestrator Class ==================== #

# Orchestrates all preprocessing, training, evaluation, and logging for model processing
class ExperimentRunner():
    """
        Purpose:
            - Handle a complete experiment run with base and optimized hyperparameters
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
            acc_base, fig_base, report_base = self.evaluator.evaluate(rf_model, X_test, y_test, figures_dir, name='base')
            mlflow.log_metric('base_accuracy', acc_base)
            mlflow.log_artifact(fig_base)

            # Hyperparameter tunning
            best_model, best_params = self.tuner.tune(rf_model, X_train, y_train)
            mlflow.log_params(best_params)

            # Best Metric Logging
            acc_opt, fig_opt, report_opt = self.evaluator.evaluate(best_model, X_test, y_test, figures_dir, name='optimized')
            mlflow.log_metric('optimized_accuracy', acc_opt)
            mlflow.log_artifact(fig_opt)

            # Calculate feature importances
            fi_path, top_feat_path = self.fi_exporter.export(best_model, num_cols, cat_cols, figures_dir)
            mlflow.log_artifact(fi_path)
            mlflow.log_artifact(top_feat_path)

            # Persist model
            self.persister.save(best_model, model_path)

            # Log example model
            X_example = X_test[:2]
            mlflow.sklearn.log_model(
                best_model,
                name='model',
                input_example=X_example,
                signature=mlflow.models.infer_signature(
                    X_example, best_model.predict(X_example)
                )
            )




# ==================== CLI ==================== #

# Orchestrates all preprocessing, training, evaluation, and logging for model processing
@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.argument('figures_dir', type=click.Path())
def main(input_path, model_path, figures_dir):
    """
        Inputs:
            - input_path: Data file path
            - model_path: Path to store the model
            - figures_dir: Directory to store generated images
        Purpose:
            - Call a experiment run
    """
    runner = ExperimentRunner()
    runner.run(input_path, model_path, figures_dir)


if __name__ == "__main__":
    main()