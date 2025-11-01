"""
Clase Dataset para cargar, procesar, filtrar y dividir conjuntos de datos.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os


class Dataset:
    """Maneja la carga, procesamiento y división de datos."""

    def __init__(self, path: str, target_column: str):
        """
        Inicializa un objeto Dataset con la ruta del archivo y el nombre
        de la columna objetivo.

        Args:
            path (str): Ruta al archivo CSV.
            target_column (str): Nombre de la columna objetivo.
        """
        self.path = path
        self.target_column = target_column
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Carga los datos desde el archivo CSV especificado."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Archivo no encontrado: {self.path}")
        self.data = pd.read_csv(self.path)
        return self.data

    def filter_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Retorna un DataFrame solo con las columnas indicadas.

        Args:
            df (pd.DataFrame): DataFrame original.
            columns (list): Lista de columnas a conservar.
        """
        return df[columns].copy()

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Args:
            df (pd.DataFrame): DataFrame a dividir.
            test_size (float): Proporción para el conjunto de prueba.
            random_state (int): Semilla para reproducibilidad.
        """
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        return train_df, test_df

    def save_data(self, df: pd.DataFrame, output_path: str):
        """
        Guarda el DataFrame en formato CSV.

        Args:
            df (pd.DataFrame): DataFrame a guardar.
            output_path (str): Ruta donde se guardará el archivo.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
