# tests/conftest.py
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def tiny_df():
    """DataFrame pequeño y estable para pruebas rápidas."""
    return pd.DataFrame({
        "date": ["2025-01-01 00:15", "2025-01-01 00:30"],
        "Usage_kWh": [3.2, 4.0],
        "Lagging_Current_Reactive.Power_kVarh": [2.95, 4.46],
        "Leading_Current_Reactive_Power_kVarh": [0.0, 0.0],
        "CO2(tCO2)": [0.0, 0.0],
        "Lagging_Current_Power_Factor": [73.21, 66.77],
        "Leading_Current_Power_Factor": [100.0, 100.0],
        "NSM": [900, 1800],
        "WeekStatus": ["Weekday", "Weekday"],
        "Day_of_week": ["Monday", "Monday"],
        "Load_Type": ["Light_Load", "Light_Load"],
    })

@pytest.fixture
def rng():
    """Semilla controlada para reproducibilidad."""
    return np.random.default_rng(42)
