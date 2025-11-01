"""
Pruebas unitarias para la clase Dataset ubicada en src/data/dataset.py
"""

import os
import pandas as pd
import pytest
from src.data.dataset import Dataset


@pytest.fixture
def sample_dataset(tmp_path):
    """
    Crea un archivo CSV temporal para las pruebas.
    """
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": ["x", "y", "z"]
    })
    df.to_csv(csv_path, index=False)

    # Retorna una instancia de Dataset con la ruta y la columna objetivo
    return Dataset(path=str(csv_path), target_column="C")


def test_load_data(sample_dataset):
    """Verifica que load_data lea correctamente un archivo CSV."""
    df = sample_dataset.load_data()
    assert not df.empty
    assert list(df.columns) == ["A", "B", "C"]


def test_filter_columns(sample_dataset):
    """Verifica que filter_columns devuelva solo las columnas solicitadas."""
    df = sample_dataset.load_data()
    filtered = sample_dataset.filter_columns(df, ["A", "B"])
    assert list(filtered.columns) == ["A", "B"]
    assert filtered.shape[1] == 2


def test_split_data(sample_dataset):
    """Verifica que split_data divida correctamente el dataset en train/test."""
    df = sample_dataset.load_data()
    train_df, test_df = sample_dataset.split_data(df, test_size=0.33)
    total_rows = len(train_df) + len(test_df)
    assert total_rows == len(df)
    assert abs(len(test_df) - 1) <= 1  # Aproximadamente una división 1/3


def test_save_data(sample_dataset, tmp_path):
    """Verifica que save_data escriba correctamente el DataFrame en disco."""
    df = sample_dataset.load_data()
    output_path = tmp_path / "output.csv"
    sample_dataset.save_data(df, str(output_path))
    assert os.path.exists(output_path)
    loaded = pd.read_csv(output_path)
    assert loaded.equals(df)
