"""
Pruebas unitarias para el módulo dataset.py.
Estas pruebas validan la carga, filtrado, división y guardado de datos.
"""

import os
import pandas as pd
import pytest
from src.data.dataset import Dataset


@pytest.fixture
def sample_csv(tmp_path):
    """Crea un archivo CSV temporal para las pruebas."""
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [10, 20, 30, 40],
        "target": ["A", "B", "A", "B"],
    })
    file_path = tmp_path / "sample.csv"
    data.to_csv(file_path, index=False)
    return file_path


def test_load_data(sample_csv):
    """Verifica que el método load_data cargue correctamente los datos."""
    dataset = Dataset(path=sample_csv, target_column="target")
    df = dataset.load_data()
    assert not df.empty
    assert list(df.columns) == ["feature1", "feature2", "target"]


def test_filter_columns(sample_csv):
    """Prueba que filter_columns conserve solo las columnas indicadas."""
    dataset = Dataset(path=sample_csv, target_column="target")
    df = dataset.load_data()
    filtered_df = dataset.filter_columns(df, ["feature1", "target"])
    assert list(filtered_df.columns) == ["feature1", "target"]


def test_split_data(sample_csv):
    """Verifica que split_data divida correctamente los datos."""
    dataset = Dataset(path=sample_csv, target_column="target")
    df = dataset.load_data()
    train_df, test_df = dataset.split_data(df, test_size=0.25)
    total_rows = len(train_df) + len(test_df)
    assert total_rows == len(df)
    assert 0 < len(test_df) < len(df)


def test_save_data(tmp_path):
    """Confirma que save_data guarda el DataFrame en la ruta especificada."""
    dataset = Dataset(path="", target_column="target")
    df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
    output_path = tmp_path / "output.csv"
    dataset.save_data(df, output_path)

    # Validar que el archivo fue creado
    assert os.path.exists(output_path)

    # Validar que el contenido coincide
    loaded_df = pd.read_csv(output_path)
    assert loaded_df.equals(df)
