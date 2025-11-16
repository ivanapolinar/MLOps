# Inicializador de ambiente MLOps

Los scripts `init.bat` (Windows) e `init.sh` (Linux/WSL) crean un entorno reproducible para este proyecto sin depender de utilidades externas como `make`. Ambos operan desde esta misma carpeta y comparten la configuración declarada en `variables.yml`.

## Flujo general

1. Detectan o instalan Python 3.11 (Windows usa `py`, `winget` o `choco`; Linux intenta `apt`, `dnf` o `brew`).
2. Resuelven la carpeta raíz del proyecto (`project.root_parent` + `project.name`) y crean las subcarpetas listadas en `paths.workspace_subdirs`.
3. Crean o reutilizan un virtualenv (`env.venv_name`) dentro del directorio del proyecto.
4. Instalan todas las dependencias base definidas en `requirements.txt` (sin DVC/Mlflow).
5. Entran al directorio del proyecto para instalar DVC y MLflow con los paquetes fijados en `mlops.dvc_package` y `mlops.mlflow_package`.
6. Ejecutan `dvc init` (a menos que `SKIP_DVC_INIT=1`) y preparan el almacenamiento de MLflow (`mlops.mlflow_backend_uri`, `mlops.mlflow_artifact_uri`).
7. Imprimen un resumen con rutas, interprete y archivo de dependencias utilizados.

## Configuración (`variables.yml`)

El archivo está en formato JSON (válido YAML). Principales campos:

```json
{
  "project": {
    "name": "proyecto",
    "root_parent": "."
  },
  "env": {
    "python_version": "3.11",
    "venv_name": ".venv-mlops"
  },
  "paths": {
    "requirements_file": "requirements.txt",
    "workspace_subdirs": ["data", "notebooks", "models", "reports", "mlruns"]
  },
  "mlops": {
    "dvc_package": "dvc[s3]==3.55.2",
    "mlflow_package": "mlflow==3.6.0",
    "dvc_init_flags": "--no-scm",
    "mlflow_backend_uri": "sqlite:///mlruns.db",
    "mlflow_artifact_uri": "mlruns"
  }
}
```

Puedes clonar este archivo para distintos escenarios y apuntar a otro usando `CONFIG_FILE=Ruta/archivo.json`.

## Uso detallado de `init.bat` (Windows)

### Requisitos previos
- Ejecutar desde PowerShell o `cmd` con permisos de escritura sobre la carpeta actual.
- Tener `py` disponible o un gestor (`winget`/`choco`) para instalar Python si falta.
- **Importante:** habilitar “Win32 long paths” una sola vez en la máquina:
  ```powershell
  reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem `
      /v LongPathsEnabled /t REG_DWORD /d 1 /f
  ```
  Luego reinicia Windows; de lo contrario, pip puede fallar por rutas >260 caracteres.

### Ejecución básica
```powershell
cd C:\Users\<user>\MaestriaIAA\MLOps\initAmbiente
.\init.bat
```

### Flags y variables útiles
- `SKIP_PIP_INSTALL=1` evita reinstalar dependencias (rápido para pruebas).
- `SKIP_DVC_INIT=1` no ejecuta `dvc init` (útil si ya existe `.dvc` custom).
- `CONFIG_FILE=...\variables.json` apunta a una configuración alternativa.

### Implementación
- Detecta Python mediante `py -3.11`, `py -3`, `python3`, `python`. Si no existe, intenta `winget` y luego `choco`.
- El virtualenv se crea en `%PROJECT_ROOT%\%CFG_VENV_NAME%`. El script siempre reusa ese entorno si ya existe.
- Las rutas relativas (`.` o `..\otra`) se expanden contra el directorio del script antes de usarse para evitar errores de sintaxis.
- DVC/Mlflow se instalan únicamente cuando el directorio del proyecto está listo; los paquetes se toman de `variables.yml`, por lo que puedes fijar otras versiones sin tocar el `.bat`.
- `prepare_mlflow` crea la carpeta de artefactos y el archivo SQLite si las URIs no apuntan a un remoto (`s3://`, `azure://`, etc.).

### Salida esperada
Al finalizar verás algo similar a:
```
============================================================
 Ambiente configurado
 -----------------------------------------------------------
 Proyecto     : C:\...\initAmbiente\proyecto
 Entorno venv : C:\...\initAmbiente\proyecto\.venv-mlops
 Python       : py -3.11
 Reqs         : C:\...\initAmbiente\requirements.txt
 -----------------------------------------------------------
 Activa el entorno con:
    call "C:\...\proyecto\.venv-mlops\Scripts\activate.bat"
============================================================
```

## Uso detallado de `init.sh` (Linux/WSL)

### Requisitos previos
- Bash 4+, `python3` y permisos de escritura donde se ejecuta el script.
- Instaladores soportados si Python no está presente: `apt`, `dnf` o `brew`.
- Paquetes estándar (`curl`, `tar`, etc.) que usan los gestores de Python.

### Ejecución básica
```bash
cd /ruta/al/repositorio/docs/initAmbiente
bash init.sh
```

### Variables de entorno
- `SKIP_PIP_INSTALL=1` omite instalaciones (igual que en Windows).
- `SKIP_DVC_INIT=1` evita `dvc init`.
- `INIT_ROOT=/otra/ruta` fuerza la base para resolver rutas relativas.
- `CONFIG_FILE=/ruta/variables.json` usa otra configuración.

### Implementación
- Usa `python3` (o `python`) para parsear `variables.yml` y exportar las variables `CFG_*`.
- El virtualenv se aloja en `$PROJECT_ROOT/$VENV_NAME` y el script lo crea con `python -m venv`. Si ya existe, simplemente lo reactiva.
- Todas las rutas se convierten a absolutas mediante `absolute_path` para que el script pueda ejecutarse desde cualquier carpeta.
- La instalación de DVC/Mlflow se hace dentro de un subshell `(...)` ubicado en `$PROJECT_ROOT`, por lo que cualquier archivo generado (por ejemplo `.dvc/config`) queda directamente en esa carpeta.
- `prepare_mlflow_storage` diferencia entre URIs remotas (http, s3, azure, etc.) y rutas locales; solo crea carpetas/archivos si se trata de un path relativo/absoluto local.

### Salida esperada

```
============================================================
Ambiente configurado
------------------------------------------------------------
Proyecto      : /home/user/MLOps/initAmbiente/proyecto
Virtualenv    : /home/user/MLOps/initAmbiente/proyecto/.venv-mlops
Python        : Python 3.11.x
Requerimientos: /home/user/MLOps/initAmbiente/requirements.txt
Subdirectorios: data notebooks models reports mlruns
------------------------------------------------------------
Activate env  : source "/home/.../.venv-mlops/bin/activate"
Deactivate    : deactivate
============================================================
```

## Problemas frecuentes

| Síntoma | Causa | Solución |
| --- | --- | --- |
| `ERROR: ... enable long paths` seguido de `No se esperaba . en este momento.` | Limitación de rutas en Windows; `pip` falla antes de que el script cierre el bloque. | Habilita `LongPathsEnabled=1` en el registro y reinicia Windows. |
| `pip` pide credenciales o falla con timeout | Proxy corporativo o mirror inexistente. | Exporta `PIP_INDEX_URL`/`PIP_NO_VERIFY`/`PIP_TRUSTED_HOST` según tu red. |
| `dvc init` no encuentra git | Se ejecuta con `--no-scm` por defecto; si necesitas integración con git, elimina `--no-scm` en `variables.yml`. |
| Necesito múltiples proyectos | Crea copias de `variables.yml` y exporta `CONFIG_FILE` antes de correr el script para cada workspace. |

## Buenas prácticas

- Ejecuta `SKIP_PIP_INSTALL=1` si solo necesitas recrear carpetas o probar que el script ya configuró todo correctamente.
- Versiona tus cambios en `variables.yml` para que el resto del equipo tenga el mismo layout.
- Después de correr el script, activa el virtualenv y ejecuta `pip list` o `dvc doctor` para verificar que las dependencias quedaron instaladas correctamente.

Con estas instrucciones deberías poder reconstruir el ambiente en cualquier máquina Windows, Linux o WSL sin pasos manuales adicionales (salvo habilitar rutas largas en Windows la primera vez). Para preguntas adicionales o nuevas opciones, documenta los cambios aquí mismo para mantener una única fuente de verdad.
