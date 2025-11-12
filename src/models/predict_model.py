"""CLI para generar predicciones y métricas a partir de un modelo .joblib.

Incluye la clase `BatchPredictor` que encapsula el flujo de inferencia por lote
y mantiene compatibilidad con las funciones existentes y la CLI.
"""

# Maneja serialización de métricas a JSON
import json
# Operaciones de sistema de archivos (rutas y creación de carpetas)
import os
# Tipado para argumentos opcionales
from typing import Optional

# Construcción de CLI
import click
# Carga de modelos serializados
import joblib
# Manipulación de datos tabulares
import pandas as pd
# Métricas
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
# Graficación
import matplotlib.pyplot as plt
# Estética de gráficos
import seaborn as sns


class BatchPredictor:
    """Orquesta la inferencia por lote y exportación de artefactos.

    Métodos principales:
      - `predict_file(...)`: Carga datos y modelo, predice, exporta CSV y
        opcionalmente métricas y figura de confusión.
    """

    def predict_file(
        self,
        input_path: str,
        model_path: str,
        predictions_path: str,
        metrics_path: Optional[str] = None,
        figures_dir: Optional[str] = None,
    ) -> None:
        ensure_dirs(predictions_path, metrics_path, figures_dir)
        df = load_dataset(input_path)
        model = joblib.load(model_path)
        X = prepare_features(df)
        y_true = df['Load_Type'] if 'Load_Type' in df.columns else None
        y_pred = model.predict(X)
        out = df.copy()
        out['Prediction'] = y_pred
        attach_probabilities(model, X, out)
        out.to_csv(predictions_path, index=False)
        maybe_metrics_and_figures(y_true, y_pred, metrics_path, figures_dir)


def load_dataset(path: str) -> pd.DataFrame:
    # Lee el CSV desde la ruta indicada a un DataFrame
    df = pd.read_csv(path)
    # Si existe una columna 'date', intenta convertirla a datetime
    # (sin fallar si hay errores)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Devuelve el DataFrame cargado (con 'date' parseado si aplica)
    return df


def prepare_features(
    df: pd.DataFrame, target: str = 'Load_Type'
) -> pd.DataFrame:
    # Elimina del DataFrame las columnas objetivo y de fecha si existen
    # El modelo entrenado incluye el preprocesamiento,
    # así que sólo pasamos features
    return df.drop(
        columns=[c for c in [target, 'date'] if c in df.columns],
        errors='ignore',
    )


def save_confusion_matrix(
    y_true,
    y_pred,
    figures_dir: str,
    name: str = 'predict',
) -> str:
    # Asegura que exista la carpeta de figuras
    os.makedirs(figures_dir, exist_ok=True)
    # Crea una nueva figura con tamaño específico
    plt.figure(figsize=(6, 4))
    # Dibuja un mapa de calor con la matriz de confusión
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt='d',
        cmap='Blues',
    )
    # Título y ejes descriptivos
    plt.title(f"Matriz de confusión ({name})")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    # Ajusta el layout para evitar recortes
    plt.tight_layout()
    # Construye la ruta del archivo de figura a guardar
    fig_path = os.path.join(figures_dir, f"confusion_matrix_{name}.png")
    # Guarda la figura en disco
    plt.savefig(fig_path)
    # Cierra la figura para liberar memoria
    plt.close()
    # Devuelve la ruta del archivo generado
    return fig_path


@click.command()
# CSV de entrada a puntuar
@click.argument('input_path', type=click.Path(exists=True))
# Modelo .joblib entrenado
@click.argument('model_path', type=click.Path(exists=True))
# Salida CSV de predicciones
@click.argument('predictions_path', type=click.Path())
# Salida JSON de métricas (opcional)
@click.argument('metrics_path', required=False, default=None)
# Carpeta para figuras (opcional)
@click.argument('figures_dir', required=False, default=None)
def main(input_path: str,
         model_path: str,
         predictions_path: str,
         metrics_path: Optional[str] = None,
         figures_dir: Optional[str] = None):
    """Genera predicciones con un modelo entrenado sobre un CSV.

    - input_path: CSV de entrada (mismas columnas que entrenamiento).
    - model_path: Ruta al .joblib entrenado (pipeline sklearn).
    - predictions_path: CSV con columna Prediction y probabilidades.
    - metrics_path: (opcional) JSON con métricas si el CSV trae Load_Type.
    - figures_dir: (opcional) carpeta para matriz de confusión
      si hay etiquetas.
    """
    # Delegar en la clase para mantener un único flujo de inferencia
    BatchPredictor().predict_file(
        input_path,
        model_path,
        predictions_path,
        metrics_path,
        figures_dir,
    )


if __name__ == '__main__':
    # Permite ejecutar el script como comando
    main()


def ensure_dirs(
    predictions_path: str,
    metrics_path: Optional[str] = None,
    figures_dir: Optional[str] = None,
):
    """Crea carpetas de salida necesarias si no existen."""
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)


def attach_probabilities(model, X, out):
    """Agrega columnas Prob_* si el modelo soporta predict_proba."""
    if not hasattr(model, 'predict_proba'):
        return
    try:
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            out['Prob_1'] = proba[:, 1]
            return
        classes_ = getattr(model, 'classes_', [])
        for idx, cls in enumerate(classes_):
            out[f'Prob_{cls}'] = proba[:, idx]
    except Exception:
        # No interrumpe si hay algún problema calculando probabilidades
        pass


def maybe_metrics_and_figures(
    y_true,
    y_pred,
    metrics_path: Optional[str],
    figures_dir: Optional[str],
):
    """Genera métricas y figura si existen etiquetas verdaderas."""
    if y_true is None:
        return
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'report': classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }
    if metrics_path:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    if figures_dir:
        save_confusion_matrix(y_true, y_pred, figures_dir, name='predict')
