"""
Script para la limpieza, imputación y guardado de datos procesados.
"""

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import os


def load_data(input_filepath):
    """Carga los datos desde la ruta especificada."""
    return pd.read_csv(input_filepath)


def clean_numeric(df, num_cols):
    """Limpia columnas numéricas y convierte valores a tipo float."""
    for col in num_cols:
        df[col] = df[col].replace(regex=r"[^0-9.\-]", value=np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def clean_object(df, object_cols):
    """Limpia columnas categóricas, eliminando espacios y valores nulos."""
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace(
            {'NAN': np.nan, 'NONE': np.nan,
             'NULL': np.nan, '': np.nan}
        )
    return df


def clean_date(df, date_cols):
    """Limpia columnas de fecha y las convierte a tipo datetime."""
    for col in date_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = pd.to_datetime(
            df[col],
            format='%d/%m/%Y %H:%M',
            errors='coerce'
        )
    return df


def cast_types(df, col_mapping):
    """Convierte columnas a los tipos definidos en col_mapping."""
    for dtype, cols in col_mapping.items():
        for col in cols:
            if col not in df.columns:
                continue
            if dtype == "float":
                df[col] = df[col].astype(float)
            elif dtype == "int":
                df[col] = df[col].astype("Int64")
            elif dtype == "object":
                df[col] = df[col].astype("string")
            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_data(df):
    """
    Realiza limpieza general del dataset y casteo de tipos.
    Devuelve DataFrame limpio y listas de columnas por tipo.
    """
    date_cols = ['date']
    object_cols = ['WeekStatus', 'Load_Type', 'Day_of_week']
    num_cols = list(set(df.columns) - set(date_cols) - set(object_cols))
    col_mapping = {
        'datetime': date_cols,
        'object': object_cols,
        'float': num_cols
    }
    df = clean_numeric(df, num_cols)
    df = clean_object(df, object_cols)
    df = clean_date(df, date_cols)
    df = cast_types(df, col_mapping)
    return df, num_cols, object_cols, date_cols


def impute_data(df, num_cols, object_cols, date_cols):
    """Imputa valores nulos en columnas numéricas, categóricas y fechas."""
    df = df.sort_values('date').reset_index(drop=True)

    if pd.isna(df.loc[0, 'date']):
        first_valid = df['date'].dropna().iloc[0]
        df.loc[0, 'date'] = first_valid - pd.Timedelta(minutes=15)

    for i in range(1, len(df)):
        if pd.isna(df.loc[i, 'date']):
            df.loc[i, 'date'] = (
                df.loc[i - 1, 'date'] + pd.Timedelta(minutes=15)
            )

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    for col in num_cols:
        if col == 'mixed_type_col':
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())

    df['Day_of_week'] = df['Day_of_week'].fillna(
        df['date'].dt.day_name().str.upper()
    )

    df['WeekStatusNum'] = df['date'].dt.dayofweek
    df['WeekStatus'] = df['WeekStatus'].fillna(
        df['WeekStatusNum'].apply(
            lambda x: 'WEEKEND' if x >= 5 else 'WEEKDAY'
        )
    )
    df.drop('WeekStatusNum', axis=1, inplace=True)
    return df


def drop_null_targets(df, target_col='Load_Type'):
    """Elimina filas donde el target es nulo."""
    return df.dropna(subset=[target_col]).reset_index(drop=True)


def save_data(df, output_filepath):
    """Guarda el dataframe en la ruta especificada."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Ejecuta el flujo completo de procesamiento de datos:
    1. Carga los datos crudos.
    2. Limpia valores y tipos de datos.
    3. Imputa valores nulos.
    4. Elimina registros sin target.
    5. Guarda el resultado final.
    """
    logger = logging.getLogger(__name__)
    logger.info('Cargando datos desde %s', input_filepath)
    df_raw = load_data(input_filepath)
    df_cleaned, num_cols, object_cols, date_cols = clean_data(df_raw)
    df_imputed = impute_data(df_cleaned, num_cols, object_cols, date_cols)
    df_final = drop_null_targets(df_imputed, target_col='Load_Type')
    save_data(df_final, output_filepath)
    logger.info('Archivo guardado en %s', output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()
