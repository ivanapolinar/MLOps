import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
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
from dotenv import load_dotenv


def load_data(input_path):
    """
    Load a dataset from a CSV file.

    Parameters
    ----------
    input_path : str
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        Loaded dataframe.
    """
    df = pd.read_csv(input_path)
    return df


def split_data(df, target="Load_Type", test_size=0.2, random_state=42):
    """
    Split a dataframe into train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    target : str, default="Load_Type"
        Name of the target column.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Split features and target arrays for training and testing.
    """
    y = df[target]
    X = df.drop(columns=[target, 'date'], errors="ignore")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessing(X):
    """
    Create a preprocessing pipeline for numeric and categorical features.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    Returns
    -------
    preprocessing : sklearn.compose.ColumnTransformer
        Preprocessing pipeline.
    num_cols : list
        List of numeric columns.
    cat_cols : list
        List of categorical columns.
    """
    cat_cols = list(X.select_dtypes('object').columns)
    num_cols = list(X.select_dtypes('number').columns)
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    return preprocessing, num_cols, cat_cols


def train_base_model(X_train, y_train, preprocessing, params=None):
    """
    Train a Random Forest model with preprocessing.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training target.
    preprocessing : sklearn.compose.ColumnTransformer
        Preprocessing pipeline.
    params : dict, optional
        Parameters for RandomForestClassifier.

    Returns
    -------
    rf_model : sklearn.pipeline.Pipeline
        Trained pipeline.
    """
    rf = RandomForestClassifier(random_state=42, **(params or {}))
    rf_model = make_pipeline(preprocessing, rf)
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_model(model, X_test, y_test, figures_dir, name="base"):
    """
    Evaluate a trained model and save the confusion matrix plot.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model.
    X_test : pandas.DataFrame
        Test features.
    y_test : pandas.Series
        Test target.
    figures_dir : str
        Directory to save figures.
    name : str, default="base"
        Name for the evaluation run.

    Returns
    -------
    acc : float
        Accuracy score.
    fig_path : str
        Path to the saved confusion matrix plot.
    report : dict
        Classification report as a dict.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    print(f"Reporte de clasificación ({name}):")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", acc)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title(f"Matriz de confusión - RandomForest ({name})")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    fig_path = f"{figures_dir}/confusion_matrix_{name}.png"
    plt.savefig(fig_path)
    plt.close()
    return acc, fig_path, report


def hyperparameter_tuning(model, X_train, y_train):
    """
    Perform hyperparameter tuning for a Random Forest pipeline.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Model pipeline.
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training target.

    Returns
    -------
    best_estimator_ : sklearn.pipeline.Pipeline
        Model pipeline with best parameters.
    best_params_ : dict
        Best hyperparameters found.
    """
    param_dist = {
        "randomforestclassifier__n_estimators": [100, 200, 400],
        "randomforestclassifier__max_depth": [10, 20, None],
        "randomforestclassifier__min_samples_split": [2, 5, 10],
        "randomforestclassifier__min_samples_leaf": [1, 2, 4],
        "randomforestclassifier__max_features": ["sqrt", "log2"]
    }
    rf_random = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    rf_random.fit(X_train, y_train)
    print("Mejores hiperparámetros encontrados:")
    print(rf_random.best_params_)
    return rf_random.best_estimator_, rf_random.best_params_


def save_feature_importance(model, num_cols, cat_cols, figures_dir):
    """
    Save feature importances and plot for a trained Random Forest pipeline.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained pipeline.
    num_cols : list
        List of numeric columns.
    cat_cols : list
        List of categorical columns.
    figures_dir : str
        Directory for saving figures.

    Returns
    -------
    fi_path : str
        Path to saved feature importances CSV.
    top_feat_path : str
        Path to top features plot.
    """
    rf_final = model.named_steps["randomforestclassifier"]
    ohe = model.named_steps["columntransformer"].named_transformers_["cat"]
    encoded_cat_cols = ohe.get_feature_names_out(cat_cols)
    final_feature_names = np.concatenate([num_cols, encoded_cat_cols])
    importances = rf_final.feature_importances_
    feature_importances = pd.DataFrame({
        "Feature": final_feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    fi_path = f"{figures_dir}/feature_importances.csv"
    feature_importances.head(15).to_csv(fi_path, index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances.head(15), x="Importance", y="Feature")
    plt.title("Top 15 Variables más importantes - RandomForest")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    top_feat_path = f"{figures_dir}/top_features.png"
    plt.savefig(top_feat_path)
    plt.close()
    return fi_path, top_feat_path


def save_model(model, model_path):
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline.
    model_path : str
        Path to save the model.
    """
    joblib.dump(model, model_path)
    print(f"Modelo guardado en {model_path}")


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.argument('figures_dir', type=click.Path())
def main(input_path, model_path, figures_dir):
    """
    Main training routine:
    - Loads data
    - Splits into train/test
    - Builds preprocessing pipeline
    - Trains and evaluates base and tuned models
    - Saves results and logs with MLflow
    - Exports feature importances and model

    Parameters
    ----------
    input_path : str
        Path to the dataset (CSV).
    model_path : str
        Path to save the trained model.
    figures_dir : str
        Directory to save figures.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Cargar variables de entorno desde .env si existe
    try:
        load_dotenv(override=True)
    except Exception:
        pass

    # Configuración de MLflow desde variables de entorno (si existen)
    # MLFLOW_TRACKING_URI: por ejemplo, http://localhost:5000 o ruta local ./mlruns
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # MLFLOW_EXPERIMENT: nombre del experimento (por defecto 'steel_energy')
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

        # Loguear modelo en MLflow; registrar en el Model Registry solo si está habilitado
        X_example = X_test[:2]
        register_flag = os.getenv("MLFLOW_REGISTER_IN_REGISTRY", "false").lower() == "true"
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
                registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "SteelEnergyRF")
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
