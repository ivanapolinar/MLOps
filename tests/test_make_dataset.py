"""
test_make_dataset.py
--------------------
Pruebas unitarias para el módulo src/data/make_dataset.py

Objetivo:
- Validar que el script make_dataset genera correctamente los archivos intermedios y limpios.
- Confirmar que los archivos resultantes no estén vacíos ni contengan valores nulos.
- Garantizar reproducibilidad del flujo de datos.

Autor: Equipo 55 - Fase 2 MLOps
"""

import os
import pandas as pd
import pytest
from src.data.make_dataset import make_dataset


@pytest.fixture
def setup_paths(tmp_path):
    """Crea rutas temporales para pruebas unitarias."""
    input_path = "data/raw/steel_energy_original.csv"
    interim_path = tmp_path / "steel_energy_interim.csv"
    clean_path = tmp_path / "steel_energy_clean.csv"
    return input_path, interim_path, clean_path


def test_make_dataset_creates_files(setup_paths):
    """Verifica que make_dataset genera los archivos esperados."""
    input_path, interim_path, clean_path = setup_paths

    make_dataset(str(input_path), str(interim_path), str(clean_path))

    assert interim_path.exists(), "No se generó el archivo intermedio"
    assert clean_path.exists(), "No se generó el archivo limpio"


def test_generated_files_are_not_empty(setup_paths):
    """Verifica que los archivos generados no estén vacíos."""
    input_path, interim_path, clean_path = setup_paths

    make_dataset(str(input_path), str(interim_path), str(clean_path))

    df_interim = pd.read_csv(interim_path)
    df_clean = pd.read_csv(clean_path)

    assert not df_interim.empty, "El archivo intermedio está vacío"
    assert not df_clean.empty, "El archivo limpio está vacío"


def test_no_missing_values_in_clean_file(setup_paths):
    """Valida que el dataset limpio no contenga valores nulos."""
    input_path, interim_path, clean_path = setup_paths

    make_dataset(str(input_path), str(interim_path), str(clean_path))

    df_clean = pd.read_csv(clean_path)
    assert not df_clean.isnull().values.any(), "El archivo limpio contiene valores nulos"
