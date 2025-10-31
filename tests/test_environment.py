"""
Pruebas básicas de entorno y configuración del proyecto.
"""

import pytest
import os

def test_environment_exists():
    """Verifica que las carpetas esenciales del proyecto existan."""
    expected_dirs = ["data", "models", "src", "reports"]
    for d in expected_dirs:
        assert os.path.isdir(d), f"La carpeta {d} no existe"

def test_imports():
    """Verifica que las librerías principales estén disponibles."""
    try:
        import pandas
        import sklearn
        import numpy
    except ImportError as e:
        pytest.fail(f"Falta una dependencia: {e}")
