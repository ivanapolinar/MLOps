@echo off
setlocal EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem  VARIABLES BASE
rem ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%.") do set "REPO_ROOT=%%~fi"
if not defined CONFIG_FILE set "CONFIG_FILE=%SCRIPT_DIR%variables.yml"

call :load_config
call :prepare_workspace
call :ensure_python
call :create_or_update_venv
call :install_base_requirements
call :create_project_structure
call :install_project_tools
call :prepare_mlflow
call :print_summary

endlocal & exit /b 0

rem ---------------------------------------------------------------------------
rem  SUBRUTINAS
rem ---------------------------------------------------------------------------

:load_config
if not exist "%CONFIG_FILE%" goto set_defaults
for /f "usebackq tokens=* delims=" %%i in (`powershell -NoProfile -Command ^
  "$cfgPath = [System.IO.Path]::GetFullPath($env:CONFIG_FILE);" ^
  "if (-not (Test-Path $cfgPath)) { exit }" ^
  "$cfg = Get-Content -Raw -Path $cfgPath | ConvertFrom-Json;" ^
  "$emit = { param($k,$v) if ($null -eq $v) { return } if ($v -is [array]) { $v = ($v -join ' ') } Write-Output ($k + '=' + $v) };" ^
  "$parent = $null;" ^
  "if ($cfg.project -and $cfg.project.root_parent) { $parent = $cfg.project.root_parent } elseif ($cfg.project -and $cfg.project.base_dir) { $parent = $cfg.project.base_dir }" ^
  "& $emit 'CFG_PROJECT_NAME' $cfg.project.name;" ^
  "& $emit 'CFG_PROJECT_PARENT' $parent;" ^
  "& $emit 'CFG_VENV_NAME' $cfg.env.venv_name;" ^
  "& $emit 'CFG_PY_VERSION' $cfg.env.python_version;" ^
  "& $emit 'CFG_REQUIREMENTS_FILE' $cfg.paths.requirements_file;" ^
  "& $emit 'CFG_WORKSPACE_SUBDIRS' $cfg.paths.workspace_subdirs;" ^
  "& $emit 'CFG_DVC_PACKAGE' $cfg.mlops.dvc_package;" ^
  "& $emit 'CFG_MLFLOW_PACKAGE' $cfg.mlops.mlflow_package;" ^
  "& $emit 'CFG_DVC_FLAGS' $cfg.mlops.dvc_init_flags;" ^
  "& $emit 'CFG_MLFLOW_BACKEND_URI' $cfg.mlops.mlflow_backend_uri;" ^
  "& $emit 'CFG_MLFLOW_ARTIFACT_URI' $cfg.mlops.mlflow_artifact_uri;" ^
  ""`) do (
  set %%i
)

:set_defaults
if not defined CFG_PROJECT_NAME set "CFG_PROJECT_NAME=proyecto"
if not defined CFG_PROJECT_PARENT set "CFG_PROJECT_PARENT=%REPO_ROOT%"
if not defined CFG_VENV_NAME set "CFG_VENV_NAME=.venv-mlops"
if not defined CFG_PY_VERSION set "CFG_PY_VERSION=3.11"
if not defined CFG_REQUIREMENTS_FILE set "CFG_REQUIREMENTS_FILE=requirements.txt"
if not defined CFG_WORKSPACE_SUBDIRS set "CFG_WORKSPACE_SUBDIRS=data notebooks models reports mlruns"
if not defined CFG_DVC_PACKAGE set "CFG_DVC_PACKAGE=dvc[s3]==3.55.2"
if not defined CFG_MLFLOW_PACKAGE set "CFG_MLFLOW_PACKAGE=mlflow==3.6.0"
if not defined CFG_DVC_FLAGS set "CFG_DVC_FLAGS=--no-scm"
if not defined CFG_MLFLOW_BACKEND_URI set "CFG_MLFLOW_BACKEND_URI=sqlite:///mlruns.db"
if not defined CFG_MLFLOW_ARTIFACT_URI set "CFG_MLFLOW_ARTIFACT_URI=mlruns"
exit /b 0

:prepare_workspace
set "PROJECT_PARENT=%CFG_PROJECT_PARENT%"
call :expand_path PROJECT_PARENT
set "PROJECT_NAME=%CFG_PROJECT_NAME%"
set "PROJECT_ROOT=%PROJECT_PARENT%\%PROJECT_NAME%"
set "PROJECT_ROOT=%PROJECT_ROOT:/=\%"
set "REQ_FILE=%CFG_REQUIREMENTS_FILE%"
call :expand_path REQ_FILE
if not exist "%REQ_FILE%" (
  echo [ERROR] No se encontro el archivo de dependencias en "%REQ_FILE%"
  exit /b 1
)
exit /b 0

:ensure_python
set "PY_VERSION=%CFG_PY_VERSION%"
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do (
  set "REQ_MAJOR=%%a"
  set "REQ_MINOR=%%b"
)
if not defined REQ_MAJOR set "REQ_MAJOR=3"
if not defined REQ_MINOR set "REQ_MINOR=11"

set "PYTHON_CMD="
call :detect_python
if defined PYTHON_CMD exit /b 0

echo [INFO] Python %PY_VERSION% no encontrado. Intentando instalar...
call :install_python
call :detect_python
if not defined PYTHON_CMD (
  echo [ERROR] Instala Python %PY_VERSION% manualmente y vuelve a ejecutar init.bat.
  exit /b 1
)
exit /b 0

:detect_python
where py >nul 2>&1
if not errorlevel 1 (
  py -%PY_VERSION% -c "exit()" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=py -%PY_VERSION%"
    exit /b
  )
  py -3 -c "import sys; sys.exit(0 if sys.version_info[:2] >= (%REQ_MAJOR%, %REQ_MINOR%) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    exit /b
  )
)
for %%P in (python3 python) do (
  %%P -c "import sys; sys.exit(0 if sys.version_info[:2] >= (%REQ_MAJOR%, %REQ_MINOR%) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=%%P"
    exit /b
  )
)
exit /b 0

:install_python
where winget >nul 2>&1
if not errorlevel 1 (
  winget install -e --id Python.Python.%PY_VERSION% --accept-source-agreements --accept-package-agreements
  goto :eof
)
where choco >nul 2>&1
if not errorlevel 1 (
  choco install python --version=%PY_VERSION%.0 -y
  goto :eof
)
echo [WARN] Instala Python %PY_VERSION% manualmente (winget/choco no disponibles).
exit /b 1

:create_or_update_venv
set "VENV_PATH=%PROJECT_ROOT%\%CFG_VENV_NAME%"
set "VENV_PATH=%VENV_PATH:/=\%"
if not exist "%PROJECT_PARENT%" mkdir "%PROJECT_PARENT%"
if not exist "%PROJECT_ROOT%" mkdir "%PROJECT_ROOT%"
if exist "%VENV_PATH%" (
  echo [INFO] Reutilizando entorno virtual en "%VENV_PATH%"
) else (
  echo [INFO] Creando entorno virtual en "%VENV_PATH%"
  call %PYTHON_CMD% -m venv "%VENV_PATH%"
)
set "PYTHON_BIN=%VENV_PATH%\Scripts\python.exe"
if not exist "%PYTHON_BIN%" set "PYTHON_BIN=%VENV_PATH%\Scripts\python"
if not exist "%PYTHON_BIN%" (
  echo [ERROR] No se encontro python dentro del entorno virtual.
  exit /b 1
)
exit /b 0

:install_base_requirements
if /i "%SKIP_PIP_INSTALL%"=="1" (
  echo [INFO] SKIP_PIP_INSTALL=1 -> se omite la instalacion de dependencias.
  exit /b 0
)
call "%PYTHON_BIN%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1
call "%PYTHON_BIN%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1
exit /b 0

:create_project_structure
if not exist "%PROJECT_PARENT%" mkdir "%PROJECT_PARENT%"
if not exist "%PROJECT_ROOT%" mkdir "%PROJECT_ROOT%"
for %%S in (%CFG_WORKSPACE_SUBDIRS%) do (
  if not "%%~S"=="" (
    set "SUBDIR=%%~S"
    set "SUBDIR=!SUBDIR:/=\!"
    if not exist "%PROJECT_ROOT%\!SUBDIR!" mkdir "%PROJECT_ROOT%\!SUBDIR!"
  )
)
exit /b 0

:install_project_tools
if not exist "%PROJECT_ROOT%" mkdir "%PROJECT_ROOT%"
pushd "%PROJECT_ROOT%"
if /i not "%SKIP_PIP_INSTALL%"=="1" (
  if defined CFG_DVC_PACKAGE (
    echo [INFO] Instalando %CFG_DVC_PACKAGE%
    call "%PYTHON_BIN%" -m pip install "%CFG_DVC_PACKAGE%"
    if errorlevel 1 (
      popd
      exit /b 1
    )
  )
  if defined CFG_MLFLOW_PACKAGE (
    echo [INFO] Instalando %CFG_MLFLOW_PACKAGE%
    call "%PYTHON_BIN%" -m pip install "%CFG_MLFLOW_PACKAGE%"
    if errorlevel 1 (
      popd
      exit /b 1
    )
  )
) else (
  echo [INFO] SKIP_PIP_INSTALL=1 -> se omite la instalacion de DVC y MLflow.
)
set "DVC_BIN=%VENV_PATH%\Scripts\dvc.exe"
if not exist "%DVC_BIN%" set "DVC_BIN=%VENV_PATH%\Scripts\dvc"
if not exist "%DVC_BIN%" set "DVC_BIN=dvc"
if /i "%SKIP_DVC_INIT%"=="1" (
  echo [INFO] SKIP_DVC_INIT=1 -> se omite dvc init.
) else (
  if exist ".dvc" (
    echo [INFO] DVC ya estaba inicializado.
  ) else (
    call "%DVC_BIN%" init %CFG_DVC_FLAGS%
  )
)
popd
exit /b 0

:prepare_mlflow
setlocal EnableDelayedExpansion
set "art_uri=%CFG_MLFLOW_ARTIFACT_URI%"
set "backend_uri=%CFG_MLFLOW_BACKEND_URI%"

if not "!art_uri!"=="" (
  call :is_remote_uri "!art_uri!" ART_REMOTE
  if /i "!ART_REMOTE!"=="false" (
    call :resolve_project_relative "!art_uri!" ART_PATH
    if not exist "!ART_PATH!" mkdir "!ART_PATH!"
  ) else (
    echo [INFO] URI remota de artefactos detectada (!art_uri!). No se crea carpeta local.
  )
)

if /i "!backend_uri:~0,10!"=="sqlite:///" (
  set "db_rel=!backend_uri:~10!"
  call :resolve_project_relative "!db_rel!" DB_PATH
  for %%i in ("!DB_PATH!") do (
    if not exist "%%~dpi" mkdir "%%~dpi"
    if not exist "%%~fi" type nul > "%%~fi"
  )
)
endlocal
exit /b 0

:is_remote_uri
setlocal EnableDelayedExpansion
set "uri=%~1"
if "!uri!"=="" (
  endlocal & set "%~2=false" & exit /b
)
set "probe=!uri:://=!"
if "!probe!"=="!uri!" (
  endlocal & set "%~2=false" & exit /b
) else (
  endlocal & set "%~2=true" & exit /b
)

:expand_path
set "var=%~1"
set "value=!%var%!"
if "!value!"=="" exit /b
if "!value:~0,2!"=="~/" (
  set "value=%USERPROFILE%\!value:~2!"
) else if "!value!"=="~" (
  set "value=%USERPROFILE%"
)
set "value=!value:/=\!"
if not "!value:~1,1!"==":" (
  if not "!value:~0,2!"=="\\" (
    if "!value:~0,1!"=="\" (
      set "value=%REPO_ROOT%!value!"
    ) else (
      set "value=%REPO_ROOT%\!value!"
    )
  )
)
for %%i in ("!value!") do set "value=%%~fi"
set "%var%=!value!"
exit /b 0

:resolve_project_relative
setlocal EnableDelayedExpansion
set "p=%~1"
if "!p!"=="" (
  endlocal & set "%~2=" & exit /b
)
if "!p:~0,2!"=="\\" goto rpr_abs
if "!p:~1,1!"==":" goto rpr_abs
if "!p:~0,1!"=="\" (
  set "p=!PROJECT_ROOT!!p!"
) else (
  set "p=!PROJECT_ROOT!\!p!"
)
:rpr_abs
set "p=!p:/=\!"
endlocal & set "%~2=%p%"
exit /b 0

:print_summary
echo(
echo ============================================================
echo  Ambiente configurado
echo ------------------------------------------------------------
echo  Proyecto     : %PROJECT_ROOT%
echo  Entorno venv : %VENV_PATH%
echo  Python       : %PYTHON_CMD%
echo  Reqs         : %REQ_FILE%
echo ------------------------------------------------------------
echo  Activa el entorno con:
echo     call "%VENV_PATH%\Scripts\activate.bat"
echo ============================================================
exit /b 0
