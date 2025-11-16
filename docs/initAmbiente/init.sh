#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${INIT_ROOT:-$SCRIPT_DIR}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/variables.yml}"

DEFAULT_PROJECT_NAME="proyecto"
DEFAULT_PROJECT_PARENT="$SCRIPT_DIR"
DEFAULT_VENV_NAME=".venv"
DEFAULT_PY_VERSION="3.11"
DEFAULT_REQUIREMENTS_REL="requirements.txt"
DEFAULT_SUBDIRS="data notebooks models reports mlruns"
DEFAULT_DVC_FLAGS="--no-scm"
DEFAULT_MLFLOW_BACKEND_URI="sqlite:///mlruns.db"
DEFAULT_MLFLOW_ARTIFACT_URI="mlruns"
DEFAULT_DVC_PACKAGE="dvc[s3]==3.55.2"
DEFAULT_MLFLOW_PACKAGE="mlflow==3.6.0"

info() {
  printf '[init.sh] %s\n' "$*"
}

die() {
  printf '[init.sh][ERROR] %s\n' "$*" >&2
  exit 1
}

expand_path() {
  local input="$1"
  case "$input" in
    "~/"*)
      printf '%s/%s' "$HOME" "${input#~/}"
      ;;
    "~")
      printf '%s' "$HOME"
      ;;
    *)
      printf '%s' "$input"
      ;;
  esac
}

absolute_path() {
  local input="$1"
  if [[ -z "$input" ]]; then
    printf ''
    return
  fi
  input="$(expand_path "$input")"
  case "$input" in
    /*)
      printf '%s' "$input"
      ;;
    *)
      printf '%s/%s' "$REPO_ROOT" "$input"
      ;;
  esac
}

CONFIG_EXPORT=""
for interpreter in python3 python; do
  if command -v "$interpreter" >/dev/null 2>&1; then
    if CONFIG_EXPORT=$(CONFIG_FILE="$CONFIG_FILE" "$interpreter" - <<'PY' 2>/dev/null); then
import json
import os
import pathlib
import sys

cfg_path = pathlib.Path(os.environ.get("CONFIG_FILE", ""))
if not cfg_path.exists():
    sys.exit(0)

def emit(key, value):
    if value in (None, ""):
        return
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    text = str(value)
    text = text.replace("'", "'\"'\"'")
    print(f"{key}='{text}'")

try:
    data = json.loads(cfg_path.read_text())
except json.JSONDecodeError as exc:  # noqa: F841
    sys.exit(1)

project = data.get("project", {})
env_cfg = data.get("env", {})
paths = data.get("paths", {})
mlops = data.get("mlops", {})

emit("CFG_PROJECT_NAME", project.get("name"))
emit("CFG_PROJECT_PARENT", project.get("root_parent") or project.get("base_dir"))
emit("CFG_VENV_NAME", env_cfg.get("venv_name"))
emit("CFG_PY_VERSION", env_cfg.get("python_version"))
emit("CFG_REQUIREMENTS_FILE", paths.get("requirements_file"))
emit("CFG_WORKSPACE_SUBDIRS", paths.get("workspace_subdirs"))
emit("CFG_DVC_FLAGS", mlops.get("dvc_init_flags"))
emit("CFG_MLFLOW_BACKEND_URI", mlops.get("mlflow_backend_uri"))
emit("CFG_MLFLOW_ARTIFACT_URI", mlops.get("mlflow_artifact_uri"))
emit("CFG_DVC_PACKAGE", mlops.get("dvc_package"))
emit("CFG_MLFLOW_PACKAGE", mlops.get("mlflow_package"))
PY
      break
    fi
  fi
done

if [[ -n "${CONFIG_EXPORT:-}" ]]; then
  eval "$CONFIG_EXPORT"
fi

PROJECT_NAME="${CFG_PROJECT_NAME:-$DEFAULT_PROJECT_NAME}"
PROJECT_PARENT="${CFG_PROJECT_PARENT:-$DEFAULT_PROJECT_PARENT}"
VENV_NAME="${CFG_VENV_NAME:-$DEFAULT_VENV_NAME}"
PY_VERSION="${CFG_PY_VERSION:-$DEFAULT_PY_VERSION}"
REQ_FILE_REL="${CFG_REQUIREMENTS_FILE:-$DEFAULT_REQUIREMENTS_REL}"
WORKSPACE_SUBDIRS_STR="${CFG_WORKSPACE_SUBDIRS:-$DEFAULT_SUBDIRS}"
DVC_INIT_FLAGS="${CFG_DVC_FLAGS:-$DEFAULT_DVC_FLAGS}"
MLFLOW_BACKEND_URI="${CFG_MLFLOW_BACKEND_URI:-$DEFAULT_MLFLOW_BACKEND_URI}"
MLFLOW_ARTIFACT_URI="${CFG_MLFLOW_ARTIFACT_URI:-$DEFAULT_MLFLOW_ARTIFACT_URI}"
DVC_PACKAGE="${CFG_DVC_PACKAGE:-$DEFAULT_DVC_PACKAGE}"
MLFLOW_PACKAGE="${CFG_MLFLOW_PACKAGE:-$DEFAULT_MLFLOW_PACKAGE}"

PROJECT_PARENT_ABS="$(absolute_path "$PROJECT_PARENT")"
PROJECT_ROOT="${PROJECT_PARENT_ABS%/}/$PROJECT_NAME"
REQ_FILE="$(absolute_path "$REQ_FILE_REL")"
VENV_PATH="$PROJECT_ROOT/$VENV_NAME"

IFS=' ' read -r -a WORKSPACE_SUBDIRS <<< "$WORKSPACE_SUBDIRS_STR"

load_config_error() {
  if [[ -f "$CONFIG_FILE" && -z "${CONFIG_EXPORT:-}" ]]; then
    die "Unable to parse $CONFIG_FILE. Verify it contains valid JSON/YAML."
  fi
}

load_config_error

[[ -f "$REQ_FILE" ]] || die "Requirements file not found at $REQ_FILE"

ensure_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    mkdir -p "$path"
  fi
}

find_python() {
  local required="$1"
  local req_major="${required%%.*}"
  local req_minor="${required#*.}"
  local candidate actual
  local -a bins=("python${required}" "python${req_major}${req_minor}" "python${req_major}.${req_minor}" "python${req_major}" "python3" "python")
  for candidate in "${bins[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if actual=$("$candidate" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null); then
        IFS='.' read -r act_major act_minor <<< "$actual"
        if (( act_major > req_major )) || { (( act_major == req_major )) && (( act_minor >= req_minor )); }; then
          printf '%s' "$candidate"
          return 0
        fi
      fi
    fi
  done
  return 1
}

install_python() {
  local sudo_cmd=""
  if command -v sudo >/dev/null 2>&1; then
    sudo_cmd="sudo"
  fi

  if command -v apt-get >/dev/null 2>&1; then
    info "Installing Python ${PY_VERSION} via apt-get (requires sudo)"
    $sudo_cmd apt-get update
    $sudo_cmd apt-get install -y "python${PY_VERSION}" "python${PY_VERSION}-venv" python3-pip || \
      $sudo_cmd apt-get install -y python3 python3-venv python3-pip
    return
  fi

  if command -v dnf >/dev/null 2>&1; then
    info "Installing Python ${PY_VERSION} via dnf (requires sudo)"
    $sudo_cmd dnf install -y "python${PY_VERSION}" python3-pip || $sudo_cmd dnf install -y python3
    return
  fi

  if command -v yum >/dev/null 2>&1; then
    info "Installing Python ${PY_VERSION} via yum (requires sudo)"
    $sudo_cmd yum install -y "python${PY_VERSION}" python3-pip || $sudo_cmd yum install -y python3
    return
  fi

  if command -v brew >/dev/null 2>&1; then
    info "Installing Python ${PY_VERSION} via Homebrew"
    brew install "python@${PY_VERSION}"
    return
  fi

  die "Automatic Python installation is not supported on this platform. Install Python ${PY_VERSION} manually and re-run."
}

ensure_python() {
  local py_cmd
  if py_cmd="$(find_python "$PY_VERSION")"; then
    printf '%s' "$py_cmd"
    return 0
  fi
  info "Python ${PY_VERSION}+ not detected. Attempting installation."
  install_python
  py_cmd="$(find_python "$PY_VERSION")" || die "Python ${PY_VERSION}+ still missing after installation attempt."
  printf '%s' "$py_cmd"
}

PY_CMD="$(ensure_python)"
info "Using Python interpreter: $PY_CMD"

create_venv() {
  ensure_dir "$PROJECT_PARENT_ABS"
  ensure_dir "$PROJECT_ROOT"
  if [[ -d "$VENV_PATH" ]]; then
    info "Virtual environment already exists at $VENV_PATH"
    return
  fi
  info "Creating virtual environment at $VENV_PATH"
  "$PY_CMD" -m venv "$VENV_PATH"
}

create_venv

VENV_BIN="$VENV_PATH/bin"
[[ -d "$VENV_BIN" ]] || VENV_BIN="$VENV_PATH/Scripts"

# shellcheck disable=SC1091
source "$VENV_BIN/activate"

install_base_requirements() {
  if [[ "${SKIP_PIP_INSTALL:-0}" == "1" ]]; then
    info "SKIP_PIP_INSTALL=1 detected. Skipping dependency installation."
    return
  fi
  info "Upgrading pip/setuptools/wheel"
  python -m pip install --upgrade pip setuptools wheel
  info "Installing dependencies from $REQ_FILE"
  python -m pip install -r "$REQ_FILE"
}

create_project_structure() {
  info "Creating project workspace at $PROJECT_ROOT"
  ensure_dir "$PROJECT_ROOT"
  for dir in "${WORKSPACE_SUBDIRS[@]}"; do
    [[ -z "$dir" ]] && continue
    ensure_dir "$PROJECT_ROOT/$dir"
  done
}

install_project_tools() {
  info "Installing project-specific tools (DVC/MLflow) inside $PROJECT_ROOT"
  (
    cd "$PROJECT_ROOT"
    if [[ "${SKIP_PIP_INSTALL:-0}" == "1" ]]; then
      info "SKIP_PIP_INSTALL=1 detected. Skipping DVC/MLflow installation."
    else
      if [[ -n "$DVC_PACKAGE" ]]; then
        info "Installing $DVC_PACKAGE"
        python -m pip install "$DVC_PACKAGE"
      fi
      if [[ -n "$MLFLOW_PACKAGE" ]]; then
        info "Installing $MLFLOW_PACKAGE"
        python -m pip install "$MLFLOW_PACKAGE"
      fi
    fi
    if [[ "${SKIP_DVC_INIT:-0}" == "1" ]]; then
      info "SKIP_DVC_INIT=1 detected. Skipping dvc init."
    elif [[ -d ".dvc" ]]; then
      info "DVC already initialized in $PROJECT_ROOT"
    else
      info "Initializing DVC repository in $PROJECT_ROOT"
      dvc init $DVC_INIT_FLAGS >/dev/null
    fi
  )
}

prepare_mlflow_storage() {
  local artifacts_path db_path
  artifacts_path="$MLFLOW_ARTIFACT_URI"
  if [[ -z "$artifacts_path" ]]; then
    :
  elif [[ "$artifacts_path" == *://* ]]; then
    info "Skipping local artifact directory creation for remote URI: $artifacts_path"
  else
    if [[ "$artifacts_path" != /* ]]; then
      artifacts_path="$PROJECT_ROOT/$artifacts_path"
    fi
    ensure_dir "$artifacts_path"
  fi

  if [[ "$MLFLOW_BACKEND_URI" == sqlite:///* ]]; then
    db_path="${MLFLOW_BACKEND_URI#sqlite:///}"
    if [[ "$db_path" != /* ]]; then
      db_path="$PROJECT_ROOT/$db_path"
    fi
    ensure_dir "$(dirname "$db_path")"
    [[ -f "$db_path" ]] || touch "$db_path"
  fi
}

install_base_requirements
create_project_structure
install_project_tools
prepare_mlflow_storage

cat <<EOF
============================================================
Ambiente configurado
------------------------------------------------------------
Proyecto      : $PROJECT_ROOT
Virtualenv    : $VENV_PATH
Python        : $("$PY_CMD" -V 2>&1)
Requerimientos: $REQ_FILE
Subdirectorios: ${WORKSPACE_SUBDIRS[*]}
------------------------------------------------------------
Activate env  : source "$VENV_BIN/activate"
Deactivate    : deactivate
============================================================
EOF
