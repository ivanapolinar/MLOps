import json  # Maneja serialización de métricas a JSON
import os  # Operaciones de sistema de archivos (rutas y creación de carpetas)
from typing import Optional  # Tipado para argumentos opcionales

import click  # Construcción de CLI
import joblib  # Carga de modelos serializados
import numpy as np  # Operaciones numéricas (referenciado por tipado/funciones)
import pandas as pd  # Manipulación de datos tabulares
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Métricas
import matplotlib.pyplot as plt  # Graficación
import seaborn as sns  # Estética de gráficos


def load_dataset(path: str) -> pd.DataFrame:
    # Lee el CSV desde la ruta indicada a un DataFrame
    df = pd.read_csv(path)
    # Si existe una columna 'date', intenta convertirla a datetime (sin fallar si hay errores)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Devuelve el DataFrame cargado (con 'date' parseado si aplica)
    return df


def prepare_features(df: pd.DataFrame, target: str = 'Load_Type') -> pd.DataFrame:
    # Elimina del DataFrame las columnas objetivo y de fecha si existen
    # El modelo entrenado incluye el preprocesamiento, así que sólo pasamos features
    return df.drop(columns=[c for c in [target, 'date'] if c in df.columns], errors='ignore')


def save_confusion_matrix(y_true, y_pred, figures_dir: str, name: str = 'predict') -> str:
    # Asegura que exista la carpeta de figuras
    os.makedirs(figures_dir, exist_ok=True)
    # Crea una nueva figura con tamaño específico
    plt.figure(figsize=(6, 4))
    # Dibuja un mapa de calor con la matriz de confusión
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
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
@click.argument('input_path', type=click.Path(exists=True))  # CSV de entrada a puntuar
@click.argument('model_path', type=click.Path(exists=True))  # Modelo .joblib entrenado
@click.argument('predictions_path', type=click.Path())  # Salida CSV de predicciones
@click.argument('metrics_path', required=False, default=None)  # Salida JSON de métricas (opcional)
@click.argument('figures_dir', required=False, default=None)  # Carpeta para figuras (opcional)
def main(input_path: str,
         model_path: str,
         predictions_path: str,
         metrics_path: Optional[str] = None,
         figures_dir: Optional[str] = None):
    """Genera predicciones con un modelo entrenado sobre un CSV.

    - input_path: CSV de entrada (mismas columnas que entrenamiento).
    - model_path: Ruta al .joblib entrenado (pipeline sklearn).
    - predictions_path: CSV con columna Prediction y probabilidades si existen.
    - metrics_path: (opcional) JSON con métricas si el CSV trae Load_Type.
    - figures_dir: (opcional) carpeta para la matriz de confusión si hay etiquetas.
    """

    # Asegura carpeta de salida para predicciones
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    # Asegura carpeta para métricas si se solicita
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    # Asegura carpeta para figuras si se provee
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)

    # Carga el dataset de entrada
    df = load_dataset(input_path)
    # Carga el modelo entrenado (.joblib) que contiene el pipeline
    model = joblib.load(model_path)

    # Prepara solo las features (sin target ni fecha)
    X = prepare_features(df)
    # Obtiene etiquetas verdaderas si están presentes para evaluar
    y_true = df['Load_Type'] if 'Load_Type' in df.columns else None

    # Predicciones del modelo
    y_pred = model.predict(X)
    # Copia del DataFrame original para anexar predicciones
    out = df.copy()
    # Agrega columna de predicción
    out['Prediction'] = y_pred
    # Si el modelo expone predict_proba, intenta guardar probabilidades
    if hasattr(model, 'predict_proba'):
        try:
            # Matriz de probabilidades por clase
            proba = model.predict_proba(X)
            # Caso binario: guarda la probabilidad de la clase positiva
            if proba.shape[1] == 2:
                out['Prob_1'] = proba[:, 1]
            else:
                # Multiclase: n columnas Prob_<clase>
                classes_ = getattr(model, 'classes_', [])
                for idx, cls in enumerate(classes_):
                    out[f'Prob_{cls}'] = proba[:, idx]
        except Exception:
            # No interrumpe si hay algún problema calculando probabilidades
            pass

    # Exporta el CSV de predicciones
    out.to_csv(predictions_path, index=False)

    # Si hay etiquetas verdaderas, calcula y exporta métricas y figura
    if y_true is not None:
        # Construye el diccionario de métricas básicas
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'report': classification_report(y_true, y_pred, output_dict=True),
        }
        # Guarda JSON si se indicó una ruta
        if metrics_path:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        # Genera matriz de confusión si se pidió carpeta de figuras
        if figures_dir:
            save_confusion_matrix(y_true, y_pred, figures_dir, name='predict')


if __name__ == '__main__':
    # Permite ejecutar el script como comando
    main()
