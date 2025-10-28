# Manejo de rutas y verificación de archivos
import os
# Lectura del archivo de métricas JSON
import json
# Construcción y lectura de DataFrames
import pandas as pd

# Importamos el módulo a probar
from src.models import predict_model  # Funciones del script de predicción

# Componentes de sklearn para entrenar un pipeline simple para las pruebas
# Preprocesamiento
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Combina transformaciones por tipo
from sklearn.compose import ColumnTransformer
# Crea el pipeline
from sklearn.pipeline import make_pipeline
# Modelo de clasificación
from sklearn.ensemble import RandomForestClassifier
# Serialización del modelo entrenado
import joblib


def _make_dummy_df():
    """Crea un DataFrame pequeño y reproducible para pruebas."""
    # Datos numéricos sencillos y una variable categórica
    df = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6, 7, 8],  # Numérica
        'f2': [2, 3, 3, 4, 2, 1, 2, 4],  # Numérica
        'cat': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],  # Categórica
        'Load_Type': [0, 1, 1, 0, 1, 0, 1, 0],  # Objetivo binario
        'date': pd.date_range('2022-01-01', periods=8, freq='D')  # Fecha
    })
    return df


def _train_and_dump_model(df: pd.DataFrame, model_path: str):
    """Entrena un pipeline minimal y lo guarda a disco para las pruebas."""
    # Separamos X e y; removemos columnas no-feature ('Load_Type', 'date')
    y = df['Load_Type']
    X = df.drop(columns=['Load_Type', 'date'])
    # Detectamos columnas por tipo
    num_cols = list(X.select_dtypes('number').columns)
    cat_cols = list(X.select_dtypes('object').columns)
    # Definimos el preprocesamiento: escala numéricas y OHE para categóricas
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    # Modelo base RandomForest (rápido y con predict_proba)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    # Unimos preprocesamiento y modelo en un pipeline
    pipe = make_pipeline(pre, rf)
    # Entrenamos el pipeline
    pipe.fit(X, y)
    # Guardamos el pipeline entrenado para usarlo en las pruebas
    joblib.dump(pipe, model_path)


def test_load_dataset_parses_date(tmp_path):
    """load_dataset debe leer CSV y parsear 'date' a datetime si existe."""
    # Construimos un CSV temporal con una columna 'date' como string
    df = _make_dummy_df()
    csv_path = tmp_path / 'input.csv'
    df.to_csv(csv_path, index=False)
    # Cargamos con la función del módulo
    df_loaded = predict_model.load_dataset(str(csv_path))
    # Afirmamos que la columna 'date' quedó en dtype datetime64[ns]
    assert pd.api.types.is_datetime64_any_dtype(df_loaded['date'])


def test_prepare_features_drops_target_and_date():
    """prepare_features elimina 'Load_Type' y 'date' si están presentes."""
    # Creamos un DataFrame de ejemplo
    df = _make_dummy_df()
    # Aplicamos la función que prepara X
    X = predict_model.prepare_features(df)
    # Comprobamos que no incluye las columnas removidas
    assert 'Load_Type' not in X.columns
    assert 'date' not in X.columns


def test_end_to_end_callback_creates_outputs(tmp_path):
    """
    La función callback de Click debe
    generar predicciones y métricas/figuras.
    """
    # Rutas temporales para modelo, entradas y salidas
    model_path = tmp_path / 'model.joblib'
    input_path = tmp_path / 'input.csv'
    predictions_path = tmp_path / 'preds.csv'
    metrics_path = tmp_path / 'metrics.json'
    figures_dir = tmp_path / 'figs'

    # Preparamos datos de entrada y entrenamos un modelo compatible
    df = _make_dummy_df()
    df.to_csv(input_path, index=False)
    _train_and_dump_model(df, str(model_path))

    # Ejecutamos la función real (callback) detrás del comando Click
    predict_model.main.callback(
        str(input_path),
        str(model_path),
        str(predictions_path),
        str(metrics_path),
        str(figures_dir)
    )

    # Verificamos la existencia de artefactos de salida
    assert os.path.exists(predictions_path)
    assert os.path.exists(metrics_path)
    # La figura sólo existe si hay etiquetas; en este caso sí
    assert os.path.exists(
        os.path.join(str(figures_dir), 'confusion_matrix_predict.png')
    )

    # Validamos el contenido básico del CSV de predicciones
    preds_df = pd.read_csv(predictions_path)
    assert 'Prediction' in preds_df.columns
    # Si el modelo expone predict_proba, debe existir Prob_1 en binario
    assert any(col.startswith('Prob_') for col in preds_df.columns)

    # Validamos el JSON de métricas tiene accuracy y report
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    assert 'accuracy' in metrics and 'report' in metrics
