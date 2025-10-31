"""
make_dataset.py
----------------
Módulo responsable de la carga y preparación de datos brutos del proyecto Steel Industry Energy Consumption.
Implementa la función principal `make_dataset` que lee el dataset original, aplica transformaciones mínimas
y guarda las versiones intermedia y limpia bajo las carpetas /data/interim y /data/clean.

Autor: Equipo 55 - Fase 2 MLOps
"""

import os
import pandas as pd


def make_dataset(input_filepath: str, interim_filepath: str, clean_filepath: str) -> None:
    """
    Carga los datos originales, realiza limpieza básica y guarda resultados.

    Parámetros
    ----------
    input_filepath : str
        Ruta del archivo CSV original (data/raw/).
    interim_filepath : str
        Ruta donde se almacenará la versión intermedia (data/interim/).
    clean_filepath : str
        Ruta donde se almacenará la versión final limpia (data/clean/).

    Retorna
    -------
    None
    """
    # 1. Cargar dataset original
    df = pd.read_csv(input_filepath)

    # 2. Limpieza mínima de ejemplo (ajustaremos más adelante según pruebas)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. Guardar dataset intermedio y limpio
    os.makedirs(os.path.dirname(interim_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(clean_filepath), exist_ok=True)
    df.to_csv(interim_filepath, index=False)
    df.to_csv(clean_filepath, index=False)

    print(f"Datasets generados:\n - Interim: {interim_filepath}\n - Clean: {clean_filepath}")


if __name__ == "__main__":
    # Ejemplo de uso (para pruebas locales)
    make_dataset(
        input_filepath="data/raw/steel_energy_original.csv",
        interim_filepath="data/interim/steel_energy_interim.csv",
        clean_filepath="data/clean/steel_energy_clean.csv",
    )
