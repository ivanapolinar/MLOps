"""
Pruebas unitarias para el módulo make_dataset.py.
Estas pruebas validan la correcta ejecución de las funciones principales de limpieza,
imputación y guardado de datos definidas en src/data/make_dataset.py.
"""

import os
import pandas as pd
import numpy as np
import pytest
from src.data import make_dataset


@pytest.fixture
def sample_dataframe():
    """Crea un DataFrame de ejemplo para las pruebas."""
    data = {
        "date": ["01/01/2024 00:00", "01/01/2024 00:15", None],
        "Load_Type": ["Residential", None, "Commercial"],
        "WeekStatus": ["Weekday", "Weekend", None],
        "Day_of_week": ["Monday", "Saturday", None],
        "Value": ["10.5", "invalid", "30.0"],
    }
    return pd.DataFrame(data)


def test_load_data(tmp_path):
    """Verifica que load_data cargue correctamente un archivo CSV."""
    file_path = tmp_path / "input.csv"
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df.to_csv(file_path, index=False)

    loaded_df = make_dataset.load_data(file_path)
    assert not loaded_df.empty
    assert list(loaded_df.columns) == ["A", "B"]


def test_clean_data(sample_dataframe):
    """Valida que clean_data identifique columnas por tipo y las limpie correctamente."""
    cleaned_df, num_cols, object_cols, date_cols = make_dataset.clean_data(sample_dataframe.copy())

    # Verificaciones básicas
    assert isinstance(cleaned_df, pd.DataFrame)
    assert "date" in date_cols
    assert all(col in cleaned_df.columns for col in object_cols)
    assert all(isinstance(x, str) for x in object_cols)
    assert cleaned_df["date"].dtype.kind in ["M"]  # datetime


def test_impute_data(sample_dataframe):
    """Prueba que impute_data complete valores nulos correctamente."""
    df_cleaned, num_cols, object_cols, date_cols = make_dataset.clean_data(sample_dataframe.copy())
    df_imputed = make_dataset.impute_data(df_cleaned, num_cols, object_cols, date_cols)

    # Validar que las fechas nulas fueron completadas
    assert df_imputed["date"].isna().sum() == 0
    # Validar que las columnas categóricas también fueron imputadas
    assert df_imputed["WeekStatus"].isna().sum() == 0
    assert df_imputed["Day_of_week"].isna().sum() == 0


def test_drop_null_targets(sample_dataframe):
    """Confirma que drop_null_targets elimina filas con target nulo."""
    df_cleaned, num_cols, object_cols, date_cols = make_dataset.clean_data(sample_dataframe.copy())
    df_imputed = make_dataset.impute_data(df_cleaned, num_cols, object_cols, date_cols)
    df_result = make_dataset.drop_null_targets(df_imputed, target_col="Load_Type")

    # Ninguna fila debe tener el target nulo
    assert df_result["Load_Type"].isna().sum() == 0


def test_save_data(tmp_path):
    """Valida que save_data guarde correctamente el DataFrame en disco."""
    df = pd.DataFrame({"A": [10, 20], "B": [30, 40]})
    output_path = tmp_path / "output.csv"

    make_dataset.save_data(df, output_path)

    assert os.path.exists(output_path)
    loaded_df = pd.read_csv(output_path)
    assert loaded_df.equals(df)
