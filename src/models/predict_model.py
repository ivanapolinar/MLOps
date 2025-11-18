import os
import pandas as pd
import joblib
import click
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ======================================================
# LOAD DATASET
# ======================================================
def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV and parse date if present."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ======================================================
# PREPARE FEATURES
# ======================================================
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not used by the model."""
    df = df.copy()
    for col in ["Load_Type", "date"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


# ======================================================
# HELPERS NEEDED BY PIPELINE
# ======================================================
def ensure_dirs(
    preds_path: str,
    metrics_path: str | None = None,
    figures_dir: str | None = None,
) -> None:
    """Create required directories."""
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)


def attach_probabilities(model, X, out_df) -> None:
    """If model has predict_proba, attach per-class probabilities."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        for i in range(proba.shape[1]):
            out_df[f"Prob_{i}"] = proba[:, i]


def maybe_metrics_and_figures(
    y_true,
    y_pred,
    metrics_path: str,
    figures_dir: str,
) -> None:
    """
    Save metrics + confusion matrix figure.
    Tests expect EXACTLY these 4 parameters.
    """
    if y_true is None:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": None, "report": {}}, f)
        return

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
        ),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    os.makedirs(figures_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    fig_path = os.path.join(
        figures_dir,
        "confusion_matrix_predict.png",
    )
    plt.savefig(fig_path)
    plt.close()


# ======================================================
# BATCH PREDICTOR
# ======================================================
class BatchPredictor:
    """Batch prediction helper used by tests."""

    def predict_file(
        self,
        input_path: str,
        model_path: str,
        preds_path: str,
        metrics_path: str,
        figures_dir: str,
    ) -> None:
        df = load_dataset(input_path)
        X = prepare_features(df)

        model = joblib.load(model_path)
        y_pred = model.predict(X)

        out = df.copy()
        out["Prediction"] = y_pred
        attach_probabilities(model, X, out)
        out.to_csv(preds_path, index=False)

        y_true = df["Load_Type"] if "Load_Type" in df.columns else None

        maybe_metrics_and_figures(
            y_true,
            y_pred,
            metrics_path,
            figures_dir,
        )


# ======================================================
# CLICK MAIN GROUP (needed for tests)
# ======================================================
@click.group()
def main():
    """Main CLI entry expected by tests."""
    pass


def _callback(
    input_path,
    model_path,
    preds_path,
    metrics_path,
    figures_dir,
):
    """Callback used by tests."""
    BatchPredictor().predict_file(
        input_path=input_path,
        model_path=model_path,
        preds_path=preds_path,
        metrics_path=metrics_path,
        figures_dir=figures_dir,
    )


# Tests expect: predict_model.main.callback
main.callback = _callback


@main.command()
@click.argument("input_path")
@click.argument("model_path")
@click.argument("preds_path")
@click.argument("metrics_path")
@click.argument("figures_dir")
def cli(input_path, model_path, preds_path, metrics_path, figures_dir):
    """CLI entrypoint."""
    _callback(input_path, model_path, preds_path, metrics_path, figures_dir)


if __name__ == "__main__":
    main()
