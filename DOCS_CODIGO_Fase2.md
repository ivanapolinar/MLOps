# Proyecto MLOps | Fase 2 - Equipo 55
## Documentación técnica de código (actualizada)

---

## 1. Introducción

Este documento describe los módulos implementados por **Mario** durante la Fase 2 del proyecto **MLOps**, incluyendo su estructura, dependencias, flujo de ejecución, pruebas unitarias y control de estilo PEP8.

Los módulos documentados son:
- `src/data/make_dataset.py`
- `src/data/dataset.py`

Ambos scripts fueron desarrollados siguiendo buenas prácticas de **reproducibilidad, trazabilidad, control de versiones y estilo de código**.

---

## 2. Módulo: `dataset.py`

### 2.1 Propósito
Facilitar el manejo estructurado de datasets: carga, filtrado, división en conjuntos de entrenamiento/prueba y guardado de archivos.

### 2.2 Funcionalidades principales
- **`load_data()`** → Carga archivos CSV.
- **`filter_columns()`** → Selecciona subconjuntos de columnas.
- **`split_data()`** → Divide los datos en entrenamiento y prueba.
- **`save_data()`** → Guarda resultados en disco.

### 2.3 Pruebas unitarias asociadas
Archivo: `tests/test_dataset.py`  
Ejecución:
```bash
pytest -v tests/test_dataset.py
```

**Resultados esperados:**
```
4 passed in 2.41s
```

---

## 3. Módulo: `make_dataset.py`

### 3.1 Propósito
Automatizar el procesamiento de datos crudos:
- Limpieza (valores nulos, tipos de datos)
- Imputación de valores faltantes
- Casteo de tipos (`float`, `object`, `datetime`)
- Guardado del dataset limpio para su uso en el modelado

### 3.2 Flujo de ejecución
1. Entrada: `data/raw/steel_energy_original.csv`
2. Limpieza → `data/interim/steel_energy_interim.csv`
3. Imputación y guardado → `data/clean/steel_energy_clean.csv`

### 3.3 Pruebas unitarias
Archivo: `tests/test_make_dataset.py`  
Ejecución:
```bash
pytest -v tests/test_make_dataset.py
```

**Resultados esperados:**
```
5 passed in 0.95s
```

---

## 4. Control de calidad y estilo PEP8

El código fue validado con **flake8** y formateado con **autopep8**, asegurando cumplimiento con la guía de estilo de Python (PEP8).

Comandos utilizados:
```bash
flake8 src/ tests/
autopep8 --in-place --max-line-length 79 --aggressive --aggressive src/data/*.py
autopep8 --in-place --max-line-length 79 --aggressive --aggressive tests/*.py
```

---

## 5. Conclusión

- Se documentaron y probaron los módulos `make_dataset.py` y `dataset.py`.
- Se garantizó la compatibilidad con PEP8 y la reproducibilidad de los procesos.
- Todos los tests y *lint checks* pasaron correctamente en el pipeline de GitHub Actions.

**Autor:** Mario  
**Equipo:** 55  
**Fecha:** Noviembre 2025
