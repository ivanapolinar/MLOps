# Proyecto MLOps | Fase 2 - Equipo 55

## 1. Contexto general
El proyecto corresponde a la **Fase 2 del curso de MLOps**, enfocada en aplicar buenas prácticas de ingeniería para la automatización, pruebas y versionado del flujo de datos y modelos.

---

## 2. Rol de Mario
Responsable de:
- ✅ Documentación técnica de `dataset.py` y `make_dataset.py`
- ✅ Implementación y validación de **pruebas unitarias**
- ✅ Corrección de estilo PEP8 para pasar los *lint checks*
- ✅ Control de versiones y creación del Pull Request final

---

## 3. Estructura general del proyecto

```
MLOps/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── clean/
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── dataset.py
│   └── models/
│       └── train_with_mlflow.py
│
├── tests/
│   ├── test_dataset.py
│   └── test_make_dataset.py
│
└── README_Fase2_Actualizado.md
```

---

## 4. Pruebas unitarias

### 4.1 Ejecución
```bash
pytest -v tests/
```

### 4.2 Resultados esperados
```
tests/test_dataset.py::test_load_data PASSED
tests/test_make_dataset.py::test_save_data PASSED
...
9 passed in total
```

✔️ Todas las pruebas se ejecutan exitosamente, validando la integridad y reproducibilidad del pipeline.

---

## 5. Linting y formato

### 5.1 Validación de estilo
```bash
flake8 src/ tests/
```

### 5.2 Autoformateo
```bash
autopep8 --in-place --max-line-length 79 --aggressive --aggressive src/data/*.py
autopep8 --in-place --max-line-length 79 --aggressive --aggressive tests/*.py
```

✅ Todos los errores **E501 (longitud de línea)** y **W292 (newline final)** fueron corregidos.

---

## 6. Pipeline en GitHub Actions

Los siguientes *checks* pasaron correctamente:
- 🧪 **Unit tests**
- 🎯 **Lint checks (flake8)**
- 🏷️ **PR title validation**

Esto confirma la integración exitosa en el flujo CI/CD.

---

## 7. Conclusiones

- El código es totalmente reproducible y cumple con PEP8.
- Los módulos de datos y pruebas funcionan sin errores.
- El pipeline automatizado valida calidad, estilo y funcionalidad.
- El PR **feature/dataset-doc-tests** está listo para ser aprobado y mergeado.

**Autor:** Mario  
**Equipo:** 55  
**Fecha:** Noviembre 2025
