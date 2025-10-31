# tests/test_data.py
import os
import pandas as pd
import pytest

# Ajusta la ruta de import a tu módulo real:
# p. ej.: from src.steel_energy.data.make_dataset import load_data
from src.data.make_dataset import load_data  # <-- ADAPTA ESTO

def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame), "load_data debe retornar un DataFrame"
    assert df.shape[0] > 0, "El DataFrame está vacío"

def test_load_data_expected_columns():
    df = load_data()
    expected = {
        "date","Usage_kWh","Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh","CO2(tCO2)",
        "Lagging_Current_Power_Factor","Leading_Current_Power_Factor",
        "NSM","WeekStatus","Day_of_week","Load_Type"
    }
    assert expected.issubset(df.columns), f"Faltan columnas: {expected - set(df.columns)}"

def test_load_data_invalid_path_raises(monkeypatch):
    # Si tu load_data acepta ruta, simula un path inválido
    def fake_read_csv(*args, **kwargs):
        raise FileNotFoundError("archivo no existe")
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    with pytest.raises(FileNotFoundError):
        load_data()
