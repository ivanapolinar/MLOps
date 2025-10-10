import pandas as pd
import numpy as np
from typing import List, Optional

NULL_TOKENS = {"", "nan", "none", "null", "n/a", "na", "?", "--"}

def detectar_strings_en_columna(df: pd.DataFrame, columna: str) -> list:
    """
    Detecta valores no numéricos (strings u otros tipos) en una columna
    que se espera que sea numérica.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada.
    columna : str
        Nombre de la columna a analizar.

    Retorna:
    --------
    list :
        Lista con los valores únicos no numéricos encontrados.
        Si no hay valores no numéricos, devuelve una lista vacía.
    """

    # Asegurarse de que la columna existe
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    # Convertir la columna a tipo string (para poder analizarla uniformemente)
    serie = df[columna].astype(str).str.strip()

    # Intentar convertir a numérico (todo lo que no pueda, quedará como NaN)
    serie_numerica = pd.to_numeric(serie, errors="coerce")

    # Detectar las filas donde la conversión falló (NaN en la serie numérica)
    mask_no_numericos = serie_numerica.isna() & serie.notna()

    # Obtener los valores únicos que no son numéricos
    valores_no_numericos = serie[mask_no_numericos].unique().tolist()

    # Limpiar posibles valores triviales (como 'nan' o cadenas vacías)
    valores_no_numericos = [v for v in valores_no_numericos if v.lower() not in ("nan", "", "none", "null")]

    # Imprimir un resumen y devolver la lista
    if valores_no_numericos:
        print(f"Se encontraron {len(valores_no_numericos)} valores no numéricos en '{columna}':")
        print(valores_no_numericos)
    else:
        print(f"No se encontraron valores no numéricos en '{columna}'.")

    return valores_no_numericos

def limpiar_columna_numerica(df: pd.DataFrame, columna: str, valores_invalidos: list) -> pd.DataFrame:
    """
    Reemplaza valores no numéricos (strings) por NaN y convierte la columna a tipo float.

    Parámetros:
    -----------
    df : pd.DataFrame
        El DataFrame de entrada.
    columna : str
        Nombre de la columna que se desea limpiar.
    valores_invalidos : list
        Lista de valores de texto considerados inválidos (por ejemplo ['error', 'invalid', 'n/a']).

    Retorna:
    --------
    pd.DataFrame :
        El DataFrame con la columna limpiada y convertida a tipo float.
    """

    # Verificar que la columna exista
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    # Convertir la columna a string para detectar los valores inválidos sin errores de tipo
    serie = df[columna].astype(str).str.strip()

    # Normalizar la lista de valores inválidos a minúsculas para comparación robusta
    valores_invalidos_normalizados = [v.lower().strip() for v in valores_invalidos]

    # Crear una máscara booleana que identifica los valores inválidos
    mask_invalidos = serie.str.lower().isin(valores_invalidos_normalizados)

    # Contar cuántos valores inválidos hay
    total_invalidos = mask_invalidos.sum()
    print(f"Se encontraron {total_invalidos} valores no numéricos en '{columna}' que serán reemplazados por NaN.")

    # Reemplazar los valores inválidos por NaN
    df.loc[mask_invalidos, columna] = np.nan

    # Intentar convertir toda la columna a numérico (las que no se puedan, quedan como NaN)
    df[columna] = pd.to_numeric(df[columna], errors='coerce')

    # Confirmar el cambio de tipo
    print(f"La columna '{columna}' ahora es de tipo: {df[columna].dtype}")

    return df

def _parse_with_strict_formats(serie_str: pd.Series, formatos: List[str]) -> pd.Series:
    """
    Intenta parsear una serie de strings a datetime usando una lista de formatos estrictos.
    Retorna una Serie de datetime con NaT en los casos no parseables por ningún formato.
    """
    parsed = pd.Series(pd.NaT, index=serie_str.index, dtype="datetime64[ns]")
    remaining_mask = serie_str.notna()

    for fmt in formatos:
        if not remaining_mask.any():
            break
        # sólo intentamos parsear en los que aún están NaT y tienen texto
        try_mask = remaining_mask & parsed.isna()
        if not try_mask.any():
            continue
        parsed_try = pd.to_datetime(serie_str[try_mask], format=fmt, errors="coerce")
        # donde funcione, asignamos
        ok_mask = parsed_try.notna()
        if ok_mask.any():
            parsed.loc[try_mask[try_mask].index[ok_mask]] = parsed_try[ok_mask]
        # actualizamos remaining (los que siguen NaT)
        remaining_mask = parsed.isna() & serie_str.notna()

    return parsed


def detectar_strings_fecha(
    df: pd.DataFrame,
    columna: str,
    formato_base: str = "%d/%m/%Y %H:%M",
    aceptar_segundos: bool = True
) -> list:
    """
    Devuelve la lista única de valores de `columna` que NO son fechas válidas
    bajo el/los formato(s) estrictos indicados.

    - Ejemplo de formato_base para tu caso: "%d/%m/%Y %H:%M"  (13/01/2018 00:15)
    - Si aceptar_segundos=True, también acepta "%d/%m/%Y %H:%M:%S"
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    # normalizamos a string y quitamos espacios
    s = df[columna].astype("string").str.strip()

    # construimos lista de formatos estrictos a validar
    formatos = [formato_base]
    if aceptar_segundos:
        # si el base no incluye segundos, añadimos una variante con segundos
        if "%S" not in formato_base:
            formatos.append(formato_base + ":%S")

    # parseo ESTRICTO con los formatos dados
    parsed = _parse_with_strict_formats(s, formatos)

    # inválidos = texto no nulo cuyo parseo quedó NaT
    mask_invalid = s.notna() & parsed.isna()

    # extraemos valores únicos problemáticos (filtrando tokens nulos comunes)
    invalid_values = (
        s[mask_invalid]
        .dropna()
        .unique()
        .tolist()
    )
    invalid_values = [v for v in invalid_values if v.strip().lower() not in NULL_TOKENS]

    if invalid_values:
        print(f"No cumplen el formato: {len(invalid_values)} valores en '{columna}'.")
        # opcional: print(invalid_values)
    else:
        print(f"Todos los valores en '{columna}' cumplen el/los formato(s) indicado(s).")

    return invalid_values


def limpiar_columna_fecha(
    df: pd.DataFrame,
    columna: str,
    formato_base: str = "%d/%m/%Y %H:%M",
    aceptar_segundos: bool = True,
    valores_invalidos: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Limpia la columna de fechas:
      - Reemplaza `valores_invalidos` por NaT (si se pasan).
      - Parsea con formatos estrictos (base y, opcionalmente, con segundos).
      - Deja la columna como datetime64[ns] con NaT donde no cumpla formato.

    Úsalo tras 'detectar_strings_fecha' o de forma directa si ya conoces los tokens inválidos.
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    s = df[columna].astype("string").str.strip()

    if valores_invalidos:
        invalid_set = {v.strip().lower() for v in valores_invalidos}
    else:
        invalid_set = set()

    # marcamos tokens inválidos explícitos como NA (antes de parsear)
    if invalid_set:
        mask_tok = s.str.lower().isin(invalid_set)
        if mask_tok.any():
            s = s.mask(mask_tok, other=pd.NA)

    # construimos formatos estrictos
    formatos = [formato_base]
    if aceptar_segundos and "%S" not in formato_base:
        formatos.append(formato_base + ":%S")

    # parseo estricto
    parsed = _parse_with_strict_formats(s, formatos)

    # asignamos
    df[columna] = parsed

    print(f"'{columna}' → datetime64[ns]. Parseadas: {(parsed.notna()).sum()}, NaT: {(parsed.isna()).sum()}.")
    return df