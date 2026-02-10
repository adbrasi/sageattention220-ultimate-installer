#!/usr/bin/env bash
set -Eeuo pipefail

# Ultimate SageAttention 2.2.0 installer for RTX 5090 (sm_120) + CUDA 12.8+
# Workflow:
# 1) Always install/validate PyTorch stack first
# 2) Try prebuilt wheel from Hugging Face (or explicit URL)
# 3) Fallback: build local wheel, install, upload to HF

ACTION="${1:-auto}"

SAGE_VERSION="${SAGE_VERSION:-2.2.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Torch stack strategy
# - default: use TORCH_CHANNEL + CUDA_INDEX_VARIANT
# - exact pin: set TORCH_VERSION/TORCHVISION_VERSION/TORCHAUDIO_VERSION/TRITON_VERSION
TORCH_CHANNEL="${TORCH_CHANNEL:-nightly}"          # nightly|stable
CUDA_INDEX_VARIANT="${CUDA_INDEX_VARIANT:-cu128}"  # cu128
TORCH_VERSION="${TORCH_VERSION:-}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-}"
TRITON_VERSION="${TRITON_VERSION:-}"
TRITON_SPEC="${TRITON_SPEC:-triton>=3.3,<4.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

MIN_CUDA_FOR_SM120="${MIN_CUDA_FOR_SM120:-12.8}"
MIN_TRITON_FOR_50XX="${MIN_TRITON_FOR_50XX:-3.3}"

# Build target for RTX 5090 (sm_120)
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
CUDAARCHS="${CUDAARCHS:-120}"
EXT_PARALLEL="${EXT_PARALLEL:-4}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:---threads 8}"

WORK_DIR="${WORK_DIR:-$PWD/.build-sageattention}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-$PWD/wheelhouse}"
LOG_DIR="${LOG_DIR:-$PWD/logs}"
REMOTE_DIR="${REMOTE_DIR:-sageattention220}"

# Optional direct wheel URL (highest priority)
WHEEL_URL="${WHEEL_URL:-}"

# HF direct-wheel style (the "comfywheel vibe")
HF_DIRECT_REPO_ID="${HF_DIRECT_REPO_ID:-adbrasi/comfywheel}"
HF_DIRECT_REPO_TYPE="${HF_DIRECT_REPO_TYPE:-model}" # dataset|model|space
HF_DIRECT_BRANCH="${HF_DIRECT_BRANCH:-main}"
HF_DIRECT_WHEEL_FILE="${HF_DIRECT_WHEEL_FILE:-}"

# Hugging Face config for manifest + upload
HF_REPO_ID="${HF_REPO_ID:-adbrasi/comfywheel}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}" # dataset|model|space
HF_REPO_BRANCH="${HF_REPO_BRANCH:-main}"
HF_TOKEN="${HF_TOKEN:-}"
HF_PRIVATE="${HF_PRIVATE:-false}"

mkdir -p "$WORK_DIR" "$WHEELHOUSE_DIR" "$LOG_DIR"

log() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<USAGE
Uso:
  ./install_sageattention220_wheel.sh [auto|install|build|publish|init-hf]

Ação padrão: auto
  auto    -> instala stack torch/triton, tenta wheel pronta (HF/URL); se falhar, compila e publica no HF
  install -> instala stack torch/triton e tenta somente wheel pronta (HF/URL)
  build   -> instala stack torch/triton, força build local e instala
  publish -> publica no HF a wheel local mais recente (gerada por build)
  init-hf -> cria/valida o repositório no Hugging Face (sem build)

Variáveis principais:
  PYTHON_BIN=python3
  TORCH_CHANNEL=nightly|stable
  CUDA_INDEX_VARIANT=cu128
  TORCH_CUDA_ARCH_LIST=12.0
  HF_DIRECT_REPO_ID=adbrasi/comfywheel
  HF_REPO_ID=adbrasi/comfywheel
  HF_TOKEN=...

Pin exato de versões (opcional):
  TORCH_VERSION=...
  TORCHVISION_VERSION=...
  TORCHAUDIO_VERSION=...
  TRITON_VERSION=...
USAGE
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Comando obrigatório não encontrado: $cmd"
}

ensure_python() {
  require_cmd "$PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import sys
print(f"[INFO] Python: {sys.version.split()[0]} ({sys.executable})")
PY
}

run_pip() {
  local cmd
  local err_file
  cmd="${1:-}"
  err_file="$WORK_DIR/.pip-fallback.err"

  if "$PYTHON_BIN" -m pip "$@" 2>"$err_file"; then
    rm -f "$err_file"
    return 0
  fi
  shift || true

  case "$cmd" in
    install|uninstall)
      "$PYTHON_BIN" -m pip "$cmd" --break-system-packages "$@"
      ;;
    *)
      "$PYTHON_BIN" -m pip "$cmd" "$@"
      ;;
  esac

  rm -f "$err_file"
}

ensure_pip_ready() {
  run_pip install -U pip setuptools wheel
}

detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_name gpu_cc
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 | xargs || true)"
    gpu_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | xargs || true)"

    log "GPU detectada: ${gpu_name:-desconhecida}"
    log "Compute capability: ${gpu_cc:-desconhecida}"

    if [[ "${gpu_name}" != *"RTX 5090"* ]]; then
      warn "GPU não parece ser RTX 5090. Continuando mesmo assim."
    fi
    if [[ -n "$gpu_cc" && "$gpu_cc" != 12.0* ]]; then
      warn "Compute capability não é 12.0 (esperado para RTX 5090)."
    fi
  else
    warn "nvidia-smi não encontrado; validação de GPU foi pulada."
  fi
}

default_torch_index_url() {
  if [[ "$TORCH_CHANNEL" == "nightly" ]]; then
    printf 'https://download.pytorch.org/whl/nightly/%s' "$CUDA_INDEX_VARIANT"
  elif [[ "$TORCH_CHANNEL" == "stable" ]]; then
    printf 'https://download.pytorch.org/whl/%s' "$CUDA_INDEX_VARIANT"
  else
    die "TORCH_CHANNEL inválido: $TORCH_CHANNEL (use nightly ou stable)"
  fi
}

hf_prefix_for_type() {
  local repo_type="$1"
  case "$repo_type" in
    dataset) printf 'datasets/' ;;
    space) printf 'spaces/' ;;
    *) printf '' ;;
  esac
}

build_hf_resolve_url() {
  local repo_id="$1"
  local repo_type="$2"
  local branch="$3"
  local path="$4"
  local prefix
  prefix="$(hf_prefix_for_type "$repo_type")"
  printf 'https://huggingface.co/%s%s/resolve/%s/%s' "$prefix" "$repo_id" "$branch" "$path"
}

python_cp_tag() {
  "$PYTHON_BIN" - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
}

fetch_hf_manifest() {
  [[ -n "$HF_REPO_ID" ]] || return 1

  local latest_url
  local latest_json="$WORK_DIR/hf-latest.json"

  latest_url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" "${REMOTE_DIR}/latest.json")"
  log "Tentando baixar manifest do HF: $latest_url"

  if [[ -n "$HF_TOKEN" ]]; then
    curl -fsSL -H "Authorization: Bearer $HF_TOKEN" "$latest_url" -o "$latest_json" || return 1
  else
    curl -fsSL "$latest_url" -o "$latest_json" || return 1
  fi

  if [[ ! -s "$latest_json" ]]; then
    return 1
  fi

  echo "$latest_json"
}

parse_manifest_stack() {
  local manifest_path="$1"

  MANIFEST_TORCH_VERSION=""
  MANIFEST_TORCHVISION_VERSION=""
  MANIFEST_TORCHAUDIO_VERSION=""
  MANIFEST_TRITON_VERSION=""
  MANIFEST_TORCH_INDEX_URL=""
  MANIFEST_WHEEL_FILE=""
  MANIFEST_HF_WHEEL_URL=""

  if [[ -z "$manifest_path" || ! -f "$manifest_path" ]]; then
    return 0
  fi

  readarray -t _vals < <("$PYTHON_BIN" - <<PY
import json
with open("$manifest_path", "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("torch_version") or "")
print(d.get("torchvision_version") or "")
print(d.get("torchaudio_version") or "")
print(d.get("triton_version") or "")
print(d.get("torch_index_url") or "")
print(d.get("wheel_file") or "")
print(d.get("hf_wheel_url") or "")
PY
)

  MANIFEST_TORCH_VERSION="${_vals[0]:-}"
  MANIFEST_TORCHVISION_VERSION="${_vals[1]:-}"
  MANIFEST_TORCHAUDIO_VERSION="${_vals[2]:-}"
  MANIFEST_TRITON_VERSION="${_vals[3]:-}"
  MANIFEST_TORCH_INDEX_URL="${_vals[4]:-}"
  MANIFEST_WHEEL_FILE="${_vals[5]:-}"
  MANIFEST_HF_WHEEL_URL="${_vals[6]:-}"
}

install_torch_stack() {
  local source_mode="default"
  local idx_url="${TORCH_INDEX_URL:-}"
  local torch_pkg=""
  local tv_pkg=""
  local ta_pkg=""
  local triton_pkg=""

  if [[ -n "$TORCH_VERSION" && -n "$TORCHVISION_VERSION" && -n "$TORCHAUDIO_VERSION" && -n "$TRITON_VERSION" ]]; then
    source_mode="env-pin"
    torch_pkg="torch==${TORCH_VERSION}"
    tv_pkg="torchvision==${TORCHVISION_VERSION}"
    ta_pkg="torchaudio==${TORCHAUDIO_VERSION}"
    triton_pkg="triton==${TRITON_VERSION}"
    if [[ -z "$idx_url" ]]; then
      idx_url="$(default_torch_index_url)"
    fi
  elif [[ -n "${MANIFEST_TORCH_VERSION:-}" && -n "${MANIFEST_TORCHVISION_VERSION:-}" && -n "${MANIFEST_TORCHAUDIO_VERSION:-}" && -n "${MANIFEST_TRITON_VERSION:-}" ]]; then
    source_mode="manifest-pin"
    torch_pkg="torch==${MANIFEST_TORCH_VERSION}"
    tv_pkg="torchvision==${MANIFEST_TORCHVISION_VERSION}"
    ta_pkg="torchaudio==${MANIFEST_TORCHAUDIO_VERSION}"
    triton_pkg="triton==${MANIFEST_TRITON_VERSION}"
    if [[ -z "$idx_url" ]]; then
      idx_url="${MANIFEST_TORCH_INDEX_URL:-$(default_torch_index_url)}"
    fi
  else
    source_mode="default"
    torch_pkg="torch"
    tv_pkg="torchvision"
    ta_pkg="torchaudio"
    triton_pkg="$TRITON_SPEC"
    if [[ -z "$idx_url" ]]; then
      idx_url="$(default_torch_index_url)"
    fi
  fi

  if [[ "$source_mode" == "default" && "$TORCH_CHANNEL" != "nightly" ]]; then
    warn "Para RTX 5090, o recomendado é TORCH_CHANNEL=nightly com CUDA_INDEX_VARIANT=cu128."
  fi

  TORCH_INDEX_URL_USED="$idx_url"
  export TORCH_INDEX_URL_USED

  log "Instalando stack torch/triton (modo=${source_mode})"
  log "Torch index: $idx_url"

  if [[ "$source_mode" == "default" && "$TORCH_CHANNEL" == "nightly" ]]; then
    run_pip install --force-reinstall --pre "$torch_pkg" "$tv_pkg" "$ta_pkg" --index-url "$idx_url"
  else
    run_pip install --force-reinstall "$torch_pkg" "$tv_pkg" "$ta_pkg" --index-url "$idx_url"
  fi

  run_pip install --force-reinstall -U "$triton_pkg"

  MIN_CUDA_FOR_SM120="$MIN_CUDA_FOR_SM120" \
  MIN_TRITON_FOR_50XX="$MIN_TRITON_FOR_50XX" \
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
import torch

def parse_maj_min(v: str):
    parts = (v or "").split(".")
    if len(parts) < 2:
        return None
    try:
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None

print(f"[INFO] torch: {torch.__version__}")
print(f"[INFO] torchvision: {md.version('torchvision')}")
print(f"[INFO] torchaudio: {md.version('torchaudio')}")
print(f"[INFO] triton: {md.version('triton')}")
print(f"[INFO] torch.version.cuda: {torch.version.cuda}")
print(f"[INFO] torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA não disponível após instalação do stack torch/triton.")

device = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()
triton_v = md.version("triton")
cuda_v = (torch.version.cuda or "").strip()
min_cuda_s = os.environ.get("MIN_CUDA_FOR_SM120", "12.8")
min_triton_s = os.environ.get("MIN_TRITON_FOR_50XX", "3.3")

print(f"[INFO] device: {device}")
print(f"[INFO] capability: {cap}")
print(f"[INFO] arch_list: {arch_list}")

if not cuda_v:
    raise SystemExit("[ERROR] torch.version.cuda vazio; build incompatível para RTX 5090.")

cuda_mm = parse_maj_min(cuda_v)
min_cuda_mm = parse_maj_min(min_cuda_s)
if not cuda_mm or not min_cuda_mm or cuda_mm < min_cuda_mm:
    raise SystemExit(f"[ERROR] CUDA do torch ({cuda_v}) < {min_cuda_s}. Use PyTorch cu128/nightly.")

triton_mm = parse_maj_min(triton_v)
min_triton_mm = parse_maj_min(min_triton_s)
if not triton_mm or not min_triton_mm or triton_mm < min_triton_mm:
    raise SystemExit(f"[ERROR] Triton ({triton_v}) < {min_triton_s}. Atualize o Triton.")

if cap[0] == 12:
    if not any("120" in arch for arch in arch_list):
        raise SystemExit("[ERROR] Torch instalado não expõe sm_120 na arch_list.")
PY
}

ensure_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    log "Usando CUDA_HOME existente: $CUDA_HOME"
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
    log "CUDA_HOME detectado: $CUDA_HOME"
  elif [[ -d "/usr/local/cuda-12.8" ]]; then
    CUDA_HOME="/usr/local/cuda-12.8"
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    log "CUDA_HOME definido para /usr/local/cuda-12.8"
  else
    die "CUDA toolkit/nvcc não encontrado (necessário para build)."
  fi

  "$PYTHON_BIN" - <<'PY'
import subprocess
print("[INFO] nvcc --version:")
print(subprocess.check_output(["nvcc", "--version"], text=True))
PY
}

get_latest_local_wheel() {
  shopt -s nullglob
  local wheels=("$WHEELHOUSE_DIR"/sageattention-"$SAGE_VERSION"-*.whl)
  shopt -u nullglob

  if (( ${#wheels[@]} == 0 )); then
    return 1
  fi

  ls -1t "${wheels[@]}" | head -n1
}

install_wheel_file() {
  local wheel_path="$1"
  [[ -f "$wheel_path" ]] || return 1

  log "Instalando wheel: $wheel_path"
  run_pip install --force-reinstall "$wheel_path" || return 1

  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
print(f"[INFO] sageattention instalado: {md.version('sageattention')}")
PY

  return 0
}

install_wheel_url() {
  local wheel_url="$1"
  [[ -n "$wheel_url" ]] || return 1

  log "Tentando instalação por URL: $wheel_url"
  run_pip install --force-reinstall "$wheel_url" || return 1

  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
print(f"[INFO] sageattention instalado: {md.version('sageattention')}")
PY

  return 0
}

try_install_from_hf() {
  [[ -n "$HF_REPO_ID" ]] || return 1

  local wheel_file="${MANIFEST_WHEEL_FILE:-}"
  local wheel_url

  if [[ -n "${MANIFEST_HF_WHEEL_URL:-}" ]]; then
    install_wheel_url "$MANIFEST_HF_WHEEL_URL" && return 0
  fi

  [[ -n "$wheel_file" ]] || return 1

  wheel_url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" "${REMOTE_DIR}/${wheel_file}")"
  install_wheel_url "$wheel_url"
}

try_install_from_url() {
  local cp_tag wheel_file default_url alt_wheel_file alt_url

  if [[ -n "$WHEEL_URL" ]]; then
    install_wheel_url "$WHEEL_URL" && return 0
    return 1
  fi

  cp_tag="$(python_cp_tag)"

  if [[ -n "$HF_DIRECT_WHEEL_FILE" ]]; then
    wheel_file="$HF_DIRECT_WHEEL_FILE"
  else
    wheel_file="sageattention-${SAGE_VERSION}-${cp_tag}-${cp_tag}-linux_x86_64.whl"
  fi

  default_url="$(build_hf_resolve_url "$HF_DIRECT_REPO_ID" "$HF_DIRECT_REPO_TYPE" "$HF_DIRECT_BRANCH" "$wheel_file")"
  if install_wheel_url "$default_url"; then
    return 0
  fi

  if [[ -z "$HF_DIRECT_WHEEL_FILE" ]]; then
    alt_wheel_file="sageattention-${SAGE_VERSION}-${cp_tag}-${cp_tag}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    alt_url="$(build_hf_resolve_url "$HF_DIRECT_REPO_ID" "$HF_DIRECT_REPO_TYPE" "$HF_DIRECT_BRANCH" "$alt_wheel_file")"
    install_wheel_url "$alt_url" && return 0
  fi

  return 1
}

install_from_any_prebuilt() {
  try_install_from_url && return 0
  try_install_from_hf && return 0
  return 1
}

wheel_sha256() {
  local path="$1"
  sha256sum "$path" | awk '{print $1}'
}

manifest_latest_path() {
  printf '%s/%s/latest.json' "$WHEELHOUSE_DIR" "$REMOTE_DIR"
}

build_manifest() {
  local wheel_path="$1"
  local wheel_file
  local sha
  local hf_url=""
  local out

  wheel_file="$(basename "$wheel_path")"
  sha="$(wheel_sha256 "$wheel_path")"

  if [[ -n "$HF_REPO_ID" ]]; then
    hf_url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" "${REMOTE_DIR}/${wheel_file}")"
  fi

  out="$(manifest_latest_path)"
  mkdir -p "$(dirname "$out")"

  WHEEL_FILE="$wheel_file" \
  WHEEL_SHA256="$sha" \
  HF_WHEEL_URL="$hf_url" \
  TORCH_INDEX_URL_USED="${TORCH_INDEX_URL_USED:-}" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  OUT_MANIFEST="$out" \
  "$PYTHON_BIN" - <<'PY'
import datetime
import importlib.metadata as md
import json
import os
import platform
import sys


def pkg_ver(name):
    try:
        return md.version(name)
    except Exception:
        return None

manifest = {
    "created_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "wheel_file": os.environ["WHEEL_FILE"],
    "wheel_sha256": os.environ["WHEEL_SHA256"],
    "sageattention_version": pkg_ver("sageattention") or "2.2.0",
    "torch_version": pkg_ver("torch"),
    "torchvision_version": pkg_ver("torchvision"),
    "torchaudio_version": pkg_ver("torchaudio"),
    "triton_version": pkg_ver("triton"),
    "torch_index_url": os.environ.get("TORCH_INDEX_URL_USED") or None,
    "target_arch": os.environ["TORCH_CUDA_ARCH_LIST"],
    "python_version": platform.python_version(),
    "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
    "platform": platform.platform(),
    "machine": platform.machine(),
    "hf_wheel_url": os.environ.get("HF_WHEEL_URL") or None,
}

out = os.environ["OUT_MANIFEST"]
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"[INFO] Manifest gerado: {out}")
PY
}

get_hf_python() {
  local hf_python="$PYTHON_BIN"
  local hf_venv="$WORK_DIR/.hf_venv"

  if "$hf_python" - <<'PY' >/dev/null 2>&1
import huggingface_hub
PY
  then
    echo "$hf_python"
    return 0
  fi

  if run_pip install -q -U huggingface_hub >/dev/null 2>&1; then
    echo "$hf_python"
    return 0
  fi

  log "pip global indisponível para huggingface_hub; usando venv auxiliar em $hf_venv"
  if [[ ! -x "$hf_venv/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$hf_venv"
  fi
  "$hf_venv/bin/python" -m pip install -q -U pip
  "$hf_venv/bin/python" -m pip install -q -U huggingface_hub
  echo "$hf_venv/bin/python"
}

publish_to_hf() {
  local wheel_path="$1"
  local latest_json
  local hf_python

  [[ -n "$HF_REPO_ID" ]] || die "HF_REPO_ID não definido."
  [[ -n "$HF_TOKEN" ]] || die "HF_TOKEN não definido."

  hf_python="$(get_hf_python)"
  ensure_hf_repo

  latest_json="$(manifest_latest_path)"
  [[ -f "$latest_json" ]] || die "Manifest não encontrado: $latest_json"

  log "Publicando wheel + manifest no Hugging Face"

  WHEEL_PATH="$wheel_path" \
  LATEST_JSON_PATH="$latest_json" \
  HF_REPO_ID="$HF_REPO_ID" \
  HF_REPO_TYPE="$HF_REPO_TYPE" \
  HF_TOKEN="$HF_TOKEN" \
  HF_PRIVATE="$HF_PRIVATE" \
  REMOTE_DIR="$REMOTE_DIR" \
  "$hf_python" - <<'PY'
import os
from huggingface_hub import HfApi

wheel = os.environ["WHEEL_PATH"]
latest = os.environ["LATEST_JSON_PATH"]
repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "dataset")
token = os.environ["HF_TOKEN"]
remote_dir = os.environ.get("REMOTE_DIR", "sageattention220")
api = HfApi(token=token)

for local in (wheel, latest):
    remote = f"{remote_dir}/{os.path.basename(local)}"
    print(f"[INFO] Upload {local} -> {remote}")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type=repo_type,
    )

print("[INFO] Upload para HF concluído")
PY
}

ensure_hf_repo() {
  local hf_python
  [[ -n "$HF_REPO_ID" ]] || die "HF_REPO_ID não definido."
  [[ -n "$HF_TOKEN" ]] || die "HF_TOKEN não definido."

  hf_python="$(get_hf_python)"

  HF_REPO_ID="$HF_REPO_ID" \
  HF_REPO_TYPE="$HF_REPO_TYPE" \
  HF_TOKEN="$HF_TOKEN" \
  HF_PRIVATE="$HF_PRIVATE" \
  "$hf_python" - <<'PY'
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type=os.environ.get("HF_REPO_TYPE", "dataset"),
    private=os.environ.get("HF_PRIVATE", "false").lower() == "true",
    exist_ok=True,
)
print("[INFO] Repo Hugging Face criado/validado com sucesso")
PY
}

build_wheel() {
  require_cmd git
  ensure_cuda_home

  log "Instalando dependências de build"
  run_pip install -U ninja cmake packaging

  local build_stamp src_dir build_log wheel_path
  build_stamp="$(date +%Y%m%d-%H%M%S)"
  src_dir="$WORK_DIR/SageAttention-v${SAGE_VERSION}-${build_stamp}"
  build_log="$LOG_DIR/build-sageattention-v${SAGE_VERSION}-${build_stamp}.log"

  log "Clonando SageAttention v${SAGE_VERSION}"
  git clone --depth 1 --branch "v${SAGE_VERSION}" https://github.com/thu-ml/SageAttention.git "$src_dir"

  export TORCH_CUDA_ARCH_LIST
  export CUDAARCHS
  export EXT_PARALLEL
  export MAX_JOBS
  export NVCC_APPEND_FLAGS

  log "Buildando wheel (target ${TORCH_CUDA_ARCH_LIST}, CUDAARCHS=${CUDAARCHS})"
  run_pip wheel "$src_dir" --no-build-isolation --wheel-dir "$WHEELHOUSE_DIR" 2>&1 | tee "$build_log"

  wheel_path="$(get_latest_local_wheel || true)"
  [[ -n "$wheel_path" ]] || die "Wheel não encontrada após build em $WHEELHOUSE_DIR"

  if grep -Eq 'sm_120|compute_120|12.0' "$build_log"; then
    log "Build log contém referências a sm_120/compute_120."
  else
    warn "Não foi possível confirmar sm_120 no log. Verifique: $build_log"
  fi

  install_wheel_file "$wheel_path" || die "Falha ao instalar wheel recém-gerada"
  build_manifest "$wheel_path"

  echo "$wheel_path"
}

validate_runtime() {
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import torch

print(f"[INFO] Validação final: sageattention={md.version('sageattention')}")
print(f"[INFO] Validação final: torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"[INFO] Validação final: cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] Validação final: capability={torch.cuda.get_device_capability(0)}")
PY
}

load_manifest_if_available() {
  local hf_manifest=""
  hf_manifest="$(fetch_hf_manifest || true)"
  parse_manifest_stack "$hf_manifest"
}

main() {
  case "$ACTION" in
    -h|--help|help)
      usage
      exit 0
      ;;
    auto|install|build|publish|init-hf)
      ;;
    *)
      usage
      die "Ação inválida: $ACTION"
      ;;
  esac

  ensure_python
  ensure_pip_ready
  detect_gpu

  case "$ACTION" in
    install)
      load_manifest_if_available
      install_torch_stack
      install_from_any_prebuilt || die "Nenhuma wheel pronta encontrada (HF/URL)."
      validate_runtime
      ;;

    build)
      load_manifest_if_available
      install_torch_stack
      local wheel_path
      wheel_path="$(build_wheel)"
      publish_to_hf "$wheel_path"
      validate_runtime
      ;;

    publish)
      local wheel_path
      wheel_path="$(get_latest_local_wheel || true)"
      [[ -n "$wheel_path" ]] || die "Nenhuma wheel local para publicar em $WHEELHOUSE_DIR"
      build_manifest "$wheel_path"
      publish_to_hf "$wheel_path"
      ;;

    init-hf)
      ensure_hf_repo
      ;;

    auto)
      load_manifest_if_available
      install_torch_stack
      if install_from_any_prebuilt; then
        log "Wheel pronta instalada com sucesso (sem rebuild)."
      else
        log "Wheel no HF não encontrada/falhou. Iniciando build local."
        local wheel_path
        wheel_path="$(build_wheel)"
        if [[ -n "$HF_REPO_ID" && -n "$HF_TOKEN" ]]; then
          publish_to_hf "$wheel_path"
        else
          warn "HF_REPO_ID/HF_TOKEN ausentes: wheel foi compilada e instalada, mas não publicada no HF."
        fi
      fi
      validate_runtime
      ;;
  esac

  log "Concluído."
}

main "$@"
