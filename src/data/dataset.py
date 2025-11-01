"""
Módulo para la clase Dataset: maneja carga, filtrado, división y guardado de datos.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    """
    Clase para manejar operaciones básicas de datasets: carga, filtrado, división y guardado.
    """

    def __init__(self, path: str, target_column: str):
        """
        Inicializa la clase Dataset.

        Args:
            path (str): Ruta del archivo CSV.
            target_column (str): Nombre de la columna objetivo (target).
        """
        self.path = path
        self.target_column = target_column
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Carga el dataset desde un archivo CSV.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.path}")
        self.data = pd.read_csv(self.path)
        return self.data

    def filter_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Filtra el DataFrame para incluir solo las columnas especificadas.

        Args:
            df (pd.DataFrame): DataFrame original.
            columns (list): Lista de columnas a mantener.

        Returns:
            pd.DataFrame: DataFrame filtrado.
        """
        return df[columns]

    def split_data(
            self,
            df: pd.DataFrame,
            test_size: float = 0.2,
            random_state: int = 42):
        """
        Divide el dataset en conjuntos de entrenamiento y prueba.

        Args:
            df (pd.DataFrame): DataFrame original.
            test_size (float): Proporción del conjunto de prueba.
            random_state (int): Semilla para la aleatoriedad.

        Returns:
            tuple: (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state)
        return train_df, test_df

    def save_data(self, df: pd.DataFrame, output_path: str):
        """
        Guarda el DataFrame en un archivo CSV.

        Args:
            df (pd.DataFrame): DataFrame a guardar.
            output_path (str): Ruta de salida del archivo CSV.
        """
        df.to_csv(output_path, index=False)
