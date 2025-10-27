import os  # Manejo de rutas y entorno
import pandas as pd  # Creación de DataFrames de prueba

from src.models import sweep  # Módulo a probar


def _make_dummy_df():
    """Crea un DataFrame pequeño y reproducible para pruebas de barrido."""
    return pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6, 7, 8],
        'f2': [2, 3, 3, 4, 2, 1, 2, 4],
        'cat': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'Load_Type': [0, 1, 1, 0, 1, 0, 1, 0],
        # La función de split descarta 'date', pero la incluimos por consistencia
        'date': pd.date_range('2022-01-01', periods=8, freq='D')
    })


def test_param_grid_default_keys():
    """Verifica que el grid por defecto tiene las claves esperadas y sin 'random_state'."""
    grid = sweep.param_grid_default()
    # Claves esperadas (sin random_state)
    expected = {
        'n_estimators',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'max_features',
    }
    assert set(grid.keys()) == expected


def test_expand_grid_limit():
    """Verifica que expand_grid limita a como máximo 60 combinaciones."""
    big_grid = {
        'a': list(range(5)),      # 5
        'b': list(range(4)),      # 4
        'c': list(range(4)),      # 4 -> 5*4*4 = 80 > 60
    }
    combos = sweep.expand_grid(big_grid)
    assert len(combos) == 60


def test_sweep_end_to_end(tmp_path, monkeypatch):
    """Ejecuta el barrido con un grid mínimo y verifica artefactos de salida."""
    # Datos de entrada
    df = _make_dummy_df()
    input_csv = tmp_path / 'input.csv'
    df.to_csv(input_csv, index=False)

    # Salidas esperadas
    model_out = tmp_path / 'model.joblib'
    figures_dir = tmp_path / 'figures'

    # Forzar tracking local por archivo en una carpeta temporal
    mlruns_dir = tmp_path / 'mlruns'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'file:{mlruns_dir}')
    monkeypatch.setenv('MLFLOW_EXPERIMENT', 'steel_energy_test')
    # Asegurar que no se intente registrar en el Model Registry
    monkeypatch.setenv('MLFLOW_REGISTER_IN_REGISTRY', 'false')

    # Reducir tamaño del grid para acelerar la prueba
    def tiny_grid():
        return {
            'n_estimators': [10],
            'max_depth': [5],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt'],
        }

    monkeypatch.setattr(sweep, 'param_grid_default', tiny_grid)

    # Ejecutar el comando Click vía la función callback
    sweep.main.callback(str(input_csv), str(model_out), str(figures_dir))

    # Verificar que el modelo fue generado
    assert os.path.exists(model_out)
    # Verificar archivos de importancia de variables generados por el mejor modelo
    fi_csv = figures_dir / 'feature_importances.csv'
    top_png = figures_dir / 'top_features.png'
    assert os.path.exists(fi_csv)
    assert os.path.exists(top_png)

