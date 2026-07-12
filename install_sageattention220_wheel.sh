#!/usr/bin/env bash
set -Eeuo pipefail

# Universal SageAttention installer — multi-GPU, multi-CUDA
# Safe flow:
# 1) Detect GPU (compute capability) and system CUDA
# 2) Only install/update torch+triton if current stack is insufficient
# 3) Look up registry.json on HF for a matching wheel (by SM + Python version)
# 4) If no match, build from source, install, publish wheel + update registry on HF

[[ "${BASH_VERSINFO[0]}" -ge 4 ]] || { printf '[ERROR] Bash 4+ required.\n' >&2; exit 1; }

ACTION="${1:-auto}"
SCRIPT_VERSION="${SCRIPT_VERSION:-2026-07-12-stack-abi-cache-v2}"

# --- Source selection ---
SAGE_VERSION="${SAGE_VERSION:-2.2.0}"
SAGE_SOURCE_REPO="${SAGE_SOURCE_REPO:-https://github.com/thu-ml/SageAttention.git}"
SAGE_SOURCE_REF="${SAGE_SOURCE_REF:-v${SAGE_VERSION}}"
SAGE_EXPECT_VERSION="${SAGE_EXPECT_VERSION:-$SAGE_VERSION}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- Torch stack (empty = auto-detect after GPU detection) ---
TORCH_CHANNEL="${TORCH_CHANNEL:-}"
CUDA_INDEX_VARIANT="${CUDA_INDEX_VARIANT:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TRITON_SPEC="${TRITON_SPEC:-}"

# Optional exact version pins
TORCH_VERSION="${TORCH_VERSION:-}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-}"
TRITON_VERSION="${TRITON_VERSION:-}"

# --- Build target (empty = auto from GPU) ---
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
CUDAARCHS="${CUDAARCHS:-}"
EXT_PARALLEL="${EXT_PARALLEL:-4}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:---threads 8}"

# --- HF artifact storage ---
# Defaults are routed per torch CUDA major (cu12 vs cu13) at runtime by
# route_hf_for_torch_cuda13 — wheels for CUDA 13 live in a separate dataset
# because the prebuilt cu12 wheels link libcudart.so.12 and won't load
# under torch+cu130.
HF_REPO_ID_DEFAULT_CU12="${HF_REPO_ID_DEFAULT_CU12:-adbrasi/sageattention220-wheels}"
HF_REPO_ID_DEFAULT_CU13="${HF_REPO_ID_DEFAULT_CU13:-AdwolfCzar/sageattention220-wheels}"
REMOTE_DIR_DEFAULT_CU12="${REMOTE_DIR_DEFAULT_CU12:-sageattention220}"
REMOTE_DIR_DEFAULT_CU13="${REMOTE_DIR_DEFAULT_CU13:-sageattention220_cu13}"

# Capture explicit user overrides BEFORE applying defaults so the routing
# helper knows whether it's allowed to switch repo.
_HF_REPO_ID_USER_SET=0
_REMOTE_DIR_USER_SET=0
[[ -n "${HF_REPO_ID:-}" ]] && _HF_REPO_ID_USER_SET=1
[[ -n "${REMOTE_DIR:-}" ]] && _REMOTE_DIR_USER_SET=1

HF_REPO_ID="${HF_REPO_ID:-$HF_REPO_ID_DEFAULT_CU12}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_REPO_BRANCH="${HF_REPO_BRANCH:-main}"
HF_PRIVATE="${HF_PRIVATE:-false}"
HF_TOKEN="${HF_TOKEN:-}"
REMOTE_DIR="${REMOTE_DIR:-$REMOTE_DIR_DEFAULT_CU12}"
_HF_ROUTING_DONE=0

# Optional explicit wheel URL (bypasses registry lookup)
WHEEL_URL="${WHEEL_URL:-}"

# Safety
ALLOW_UNSAFE_WHEEL="${ALLOW_UNSAFE_WHEEL:-0}"
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL:-0}"

WORK_DIR="${WORK_DIR:-$PWD/.build-sageattention}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-$PWD/wheelhouse}"
LOG_DIR="${LOG_DIR:-$PWD/logs}"
mkdir -p "$WORK_DIR" "$WHEELHOUSE_DIR" "$LOG_DIR"

# ===== GPU Architecture Map =====
# Format: "arch_name|min_cuda|min_triton|default_torch_channel"
declare -A GPU_ARCH_MAP=(
  ["8.0"]="Ampere|11.1|2.0|stable"
  ["8.6"]="Ampere|11.1|2.0|stable"
  ["8.7"]="Ampere|11.4|2.0|stable"
  ["8.9"]="Ada Lovelace|11.8|2.1|stable"
  ["9.0"]="Hopper|12.0|3.0|stable"
  ["10.0"]="Blackwell DC|12.8|3.3|nightly"
  ["12.0"]="Blackwell|12.8|3.3|nightly"
)

# Known CUDA -> PyTorch index variant (closest lower match is used)
declare -A CUDA_INDEX_MAP=(
  ["11.8"]="cu118"
  ["12.1"]="cu121"
  ["12.4"]="cu124"
  ["12.6"]="cu126"
  ["12.8"]="cu128"
)

# ===== Auto-detected state =====
GPU_NAME=""
GPU_CC=""
GPU_SM=""
GPU_SM_NUM=""
GPU_ARCH_NAME=""
GPU_MIN_CUDA=""
GPU_MIN_TRITON=""
GPU_DEFAULT_CHANNEL=""
SYSTEM_CUDA_VERSION=""
SYSTEM_CUDA_MAJOR=""
SYSTEM_CUDA_MINOR=""
TORCH_INDEX_URL_USED=""
LAST_BUILT_WHEEL=""
SAGE_SOURCE_COMMIT=""

# Registry state
REGISTRY_PATH=""
REGISTRY_IS_LEGACY="false"
REG_TORCH_VERSION=""
REG_TORCH_CUDA_VERSION=""
REG_TORCHVISION_VERSION=""
REG_TORCHAUDIO_VERSION=""
REG_TRITON_VERSION=""
REG_TORCH_INDEX_URL=""
REG_WHEEL_FILE=""
REG_HF_WHEEL_URL=""
REG_SM=""
REG_SAGE_VERSION=""
REG_BUILT_FROM_REPO=""
REG_BUILT_FROM_REF=""
REG_BUILT_FROM_COMMIT=""
REG_ARTIFACT_KEY=""

# ===== Utilities =====

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Comando obrigatório não encontrado: $1"
}

usage() {
  cat <<'USAGE'
Uso:
  ./install_sageattention220_wheel.sh [auto|install|build|publish|init-hf]

Ações:
  auto    -> detecta GPU, instala torch se necessário, tenta wheel do HF,
             se não existir/for inválida compila e publica
  install -> instala torch se necessário, instala somente wheel do HF
  build   -> força build local, instala e publica no HF
  publish -> publica no HF a wheel local mais recente + registry
  init-hf -> cria/valida o repositório HF

GPUs suportadas (auto-detecção via nvidia-smi):
  A100 (sm_80), RTX 4090 (sm_89), L40S (sm_89), RTX 6000 Ada (sm_89),
  H100 (sm_90), B200 (sm_100), RTX 5090 (sm_120), RTX 6000 Pro Blackwell (sm_120)

Variáveis de ambiente (todas opcionais, auto-detectadas quando possível):
  TORCH_CHANNEL        nightly|stable (auto: stable para Ada/Hopper, nightly para Blackwell)
  CUDA_INDEX_VARIANT   cu118|cu121|cu124|cu126|cu128 (auto do CUDA do sistema)
  TORCH_CUDA_ARCH_LIST ex: 8.9, 12.0 (auto da GPU detectada)
  CUDAARCHS            ex: 89, 120 (auto da GPU detectada)
  TRITON_SPEC          ex: triton>=3.0,<4.0 (auto do mínimo para a GPU)
  SKIP_TORCH_INSTALL=1 pula instalação do torch stack (usa o que já está instalado)
  HF_TOKEN             necessário para publish/init-hf
  HF_REPO_ID           auto: adbrasi (CUDA 12) ou AdwolfCzar (CUDA 13)
  SAGE_SOURCE_REF      default: v2.2.0 (use "main" para build da branch principal)
USAGE
}

# ===== Python / pip helpers =====

ensure_python() {
  require_cmd "$PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import sys
print(f"[INFO] Python: {sys.version.split()[0]} ({sys.executable})")
PY
}

run_pip() {
  local cmd="${1:-}"
  shift || true
  case "$cmd" in
    install|uninstall)
      "$PYTHON_BIN" -m pip "$cmd" --root-user-action=ignore --break-system-packages "$@" || \
      "$PYTHON_BIN" -m pip "$cmd" --root-user-action=ignore "$@"
      ;;
    *)
      "$PYTHON_BIN" -m pip "$cmd" "$@"
      ;;
  esac
}

current_package_version() {
  local package_name="$1"
  "$PYTHON_BIN" - "$package_name" <<'PY'
import importlib.metadata as md
import sys

try:
    print(md.version(sys.argv[1]))
except md.PackageNotFoundError:
    raise SystemExit(1)
PY
}

current_torch_cuda_version() {
  "$PYTHON_BIN" - <<'PY'
import torch

if not torch.version.cuda:
    raise SystemExit(1)
print(torch.version.cuda)
PY
}

runtime_artifact_key() {
  local torch_version torch_cuda_version triton_version
  torch_version="$(current_package_version torch)" || return 1
  torch_cuda_version="$(current_torch_cuda_version)" || return 1
  triton_version="$(current_package_version triton)" || return 1
  TORCH_VERSION_CURRENT="$torch_version" \
    TORCH_CUDA_VERSION_CURRENT="$torch_cuda_version" \
    TRITON_VERSION_CURRENT="$triton_version" \
    "$PYTHON_BIN" - <<'PY'
import os
import re


def safe(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")


print(
    f"torch-{safe(os.environ['TORCH_VERSION_CURRENT'])}_"
    f"cuda-{safe(os.environ['TORCH_CUDA_VERSION_CURRENT'])}_"
    f"triton-{safe(os.environ['TRITON_VERSION_CURRENT'])}"
)
PY
}

ensure_pip_ready() {
  run_pip install -U pip setuptools wheel
}

# ===== GPU Detection =====

cc_to_sm_num() {
  # "8.9" -> "89", "12.0" -> "120", "10.0" -> "100"
  local cc="$1"
  local major minor
  major="${cc%%.*}"
  minor="${cc#*.}"
  printf '%s%s' "$major" "$minor"
}

detect_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    if [[ -n "$TORCH_CUDA_ARCH_LIST" && -n "$CUDAARCHS" ]]; then
      warn "nvidia-smi não encontrado; usando TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST manualmente."
      GPU_CC="$TORCH_CUDA_ARCH_LIST"
      GPU_SM_NUM="$CUDAARCHS"
      GPU_SM="sm_${GPU_SM_NUM}"
      GPU_NAME="manual"
    else
      die "nvidia-smi não encontrado e TORCH_CUDA_ARCH_LIST/CUDAARCHS não definidos. Impossível detectar GPU."
    fi
  else
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 | xargs || true)"
    GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | xargs || true)"

    [[ -n "$GPU_CC" ]] || die "Não foi possível detectar compute capability via nvidia-smi."

    GPU_SM_NUM="$(cc_to_sm_num "$GPU_CC")"
    GPU_SM="sm_${GPU_SM_NUM}"
  fi

  log "GPU: ${GPU_NAME} | CC: ${GPU_CC} | SM: ${GPU_SM}"

  # Lookup architecture table
  if [[ -n "${GPU_ARCH_MAP[$GPU_CC]+x}" ]]; then
    IFS='|' read -r GPU_ARCH_NAME GPU_MIN_CUDA GPU_MIN_TRITON GPU_DEFAULT_CHANNEL <<< "${GPU_ARCH_MAP[$GPU_CC]}"
  else
    warn "CC ${GPU_CC} não está na tabela conhecida. Usando defaults conservadores."
    GPU_ARCH_NAME="Desconhecida"
    GPU_MIN_CUDA="12.0"
    GPU_MIN_TRITON="3.0"
    GPU_DEFAULT_CHANNEL="nightly"
  fi

  log "Arquitetura: ${GPU_ARCH_NAME} | Min CUDA: ${GPU_MIN_CUDA} | Min Triton: ${GPU_MIN_TRITON}"

  # Apply dynamic defaults only if user did not set them
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$GPU_CC}"
  CUDAARCHS="${CUDAARCHS:-$GPU_SM_NUM}"
  TORCH_CHANNEL="${TORCH_CHANNEL:-$GPU_DEFAULT_CHANNEL}"
  TRITON_SPEC="${TRITON_SPEC:-triton>=${GPU_MIN_TRITON},<4.0}"
}

# ===== CUDA Detection =====

version_ge() {
  # Returns 0 if $1 >= $2 (major.minor comparison)
  local a_major a_minor b_major b_minor
  a_major="${1%%.*}"; a_minor="${1#*.}"
  b_major="${2%%.*}"; b_minor="${2#*.}"
  (( a_major > b_major )) && return 0
  (( a_major == b_major && a_minor >= b_minor )) && return 0
  return 1
}

resolve_cuda_index_variant() {
  # Find the best (highest) known cu variant <= system CUDA
  local sys_cuda="$1"
  local best="" best_key=""
  local key

  for key in "${!CUDA_INDEX_MAP[@]}"; do
    if version_ge "$sys_cuda" "$key"; then
      if [[ -z "$best" ]] || version_ge "$key" "$best"; then
        best="$key"
        best_key="${CUDA_INDEX_MAP[$key]}"
      fi
    fi
  done

  if [[ -n "$best_key" ]]; then
    printf '%s' "$best_key"
  else
    warn "Nenhum cu variant encontrado para CUDA ${sys_cuda}; usando cu128 como fallback."
    printf 'cu128'
  fi
}

detect_system_cuda() {
  local nvcc_path=""

  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    nvcc_path="${CUDA_HOME}/bin/nvcc"
  elif command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(command -v nvcc)"
  else
    # Scan /usr/local/cuda*
    local d
    for d in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
      if [[ -x "$d" ]]; then
        nvcc_path="$d"
        break
      fi
    done
  fi

  if [[ -z "$nvcc_path" ]]; then
    warn "nvcc não encontrado no sistema. Detecção de CUDA do sistema pulada."
    warn "O build local não vai funcionar sem nvcc, mas instalação de wheel do HF ainda funciona."
    SYSTEM_CUDA_VERSION=""
    CUDA_INDEX_VARIANT="${CUDA_INDEX_VARIANT:-cu128}"
    return 0
  fi

  local nvcc_out
  nvcc_out="$("$nvcc_path" --version 2>&1 || true)"

  SYSTEM_CUDA_VERSION="$("$PYTHON_BIN" -c "
import re, sys
m = re.search(r'release\s+([0-9]+)\.([0-9]+)', '''${nvcc_out}''')
if not m: sys.exit(1)
print(f'{m.group(1)}.{m.group(2)}')
" 2>/dev/null || true)"

  if [[ -z "$SYSTEM_CUDA_VERSION" ]]; then
    warn "Não foi possível parsear versão do nvcc."
    CUDA_INDEX_VARIANT="${CUDA_INDEX_VARIANT:-cu128}"
    return 0
  fi

  SYSTEM_CUDA_MAJOR="${SYSTEM_CUDA_VERSION%%.*}"
  SYSTEM_CUDA_MINOR="${SYSTEM_CUDA_VERSION#*.}"

  log "CUDA do sistema: ${SYSTEM_CUDA_VERSION} (nvcc: ${nvcc_path})"

  # Validate against GPU minimum
  if [[ -n "$GPU_MIN_CUDA" ]] && ! version_ge "$SYSTEM_CUDA_VERSION" "$GPU_MIN_CUDA"; then
    die "CUDA ${SYSTEM_CUDA_VERSION} < ${GPU_MIN_CUDA} (mínimo para ${GPU_ARCH_NAME} / ${GPU_SM})."
  fi

  # Auto-resolve index variant
  CUDA_INDEX_VARIANT="${CUDA_INDEX_VARIANT:-$(resolve_cuda_index_variant "$SYSTEM_CUDA_VERSION")}"
  log "CUDA index variant: ${CUDA_INDEX_VARIANT}"
}

# ===== Torch Index URL =====

default_torch_index_url() {
  if [[ "$TORCH_CHANNEL" == "nightly" ]]; then
    printf 'https://download.pytorch.org/whl/nightly/%s' "$CUDA_INDEX_VARIANT"
  elif [[ "$TORCH_CHANNEL" == "stable" ]]; then
    printf 'https://download.pytorch.org/whl/%s' "$CUDA_INDEX_VARIANT"
  else
    die "TORCH_CHANNEL inválido: $TORCH_CHANNEL (nightly|stable)"
  fi
}

# ===== HF helpers =====

hf_prefix_for_type() {
  case "$1" in
    dataset) printf 'datasets/' ;;
    space) printf 'spaces/' ;;
    *) printf '' ;;
  esac
}

build_hf_resolve_url() {
  local prefix
  prefix="$(hf_prefix_for_type "$2")"
  printf 'https://huggingface.co/%s%s/resolve/%s/%s' "$prefix" "$1" "$3" "$4"
}

hf_curl() {
  local url="$1" out="$2"
  if [[ -n "$HF_TOKEN" ]]; then
    curl -fsSL -H "Authorization: Bearer $HF_TOKEN" "$url" -o "$out" 2>/dev/null
  else
    curl -fsSL "$url" -o "$out" 2>/dev/null
  fi
}

# ===== Registry System =====

# When torch is already installed and built against CUDA 13.x, the prebuilt
# cu12 wheels in adbrasi/sageattention220-wheels link libcudart.so.12 and fail
# to import (RTX PRO 6000 Blackwell scenario). Switch the default HF repo to
# the cu13-built dataset, but only if the user hasn't pinned HF_REPO_ID /
# REMOTE_DIR explicitly. Idempotent — runs at most once per execution.
route_hf_for_torch_cuda13() {
  [[ "$_HF_ROUTING_DONE" == "1" ]] && return 0
  local cuda_major
  cuda_major="$("$PYTHON_BIN" -c "
import torch
v = (torch.version.cuda or '').split('.')
print(v[0] if v and v[0] else '')
" 2>/dev/null || true)"
  if [[ "$cuda_major" == "13" ]]; then
    if [[ "$_HF_REPO_ID_USER_SET" == "0" ]]; then
      HF_REPO_ID="$HF_REPO_ID_DEFAULT_CU13"
    fi
    if [[ "$_REMOTE_DIR_USER_SET" == "0" ]]; then
      REMOTE_DIR="$REMOTE_DIR_DEFAULT_CU13"
    fi
    log "torch CUDA 13.x detectado — usando wheels cu13 de ${HF_REPO_ID}/${REMOTE_DIR}"
  fi
  _HF_ROUTING_DONE=1
}

fetch_registry() {
  route_hf_for_torch_cuda13
  local url out

  # Try registry.json first
  url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" "${REMOTE_DIR}/registry.json")"
  out="$WORK_DIR/hf-registry.json"

  if hf_curl "$url" "$out" && [[ -s "$out" ]]; then
    REGISTRY_PATH="$out"
    REGISTRY_IS_LEGACY="false"
    log "registry.json encontrado no HF."
    return 0
  fi

  # Fallback: try legacy latest_cu13.json (preferred for cu13) then latest.json
  for manifest_name in latest_cu13.json latest.json; do
    url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" "${REMOTE_DIR}/${manifest_name}")"
    out="$WORK_DIR/hf-${manifest_name}"
    if hf_curl "$url" "$out" && [[ -s "$out" ]]; then
      REGISTRY_PATH="$out"
      REGISTRY_IS_LEGACY="true"
      log "${manifest_name} (legado) encontrado no HF."
      return 0
    fi
  done

  return 1
}

parse_registry_entry() {
  [[ -n "$REGISTRY_PATH" && -f "$REGISTRY_PATH" ]] || return 1

  local python_tag current_torch current_torch_cuda current_triton lookup_key
  python_tag="$("$PYTHON_BIN" -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")"
  current_torch="$(current_package_version torch)" || return 1
  current_torch_cuda="$(current_torch_cuda_version)" || return 1
  current_triton="$(current_package_version triton)" || return 1
  lookup_key="${GPU_SM}-${python_tag}-$(runtime_artifact_key)"

  log "Procurando wheel para: ${lookup_key}"

  if [[ "$REGISTRY_IS_LEGACY" == "true" ]]; then
    # Legacy mode: parse latest.json (single wheel)
    readarray -t _vals < <("$PYTHON_BIN" - <<PY
import json, sys
with open("$REGISTRY_PATH", "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("torch_version") or "")
print(d.get("torch_cuda_version") or "")
print(d.get("torchvision_version") or "")
print(d.get("torchaudio_version") or "")
print(d.get("triton_version") or "")
print(d.get("torch_index_url") or "")
print(d.get("wheel_file") or "")
print(d.get("hf_wheel_url") or "")
print(d.get("target_arch") or d.get("sm") or "")
print(d.get("sageattention_version") or "")
print(d.get("built_from_repo") or "")
print(d.get("built_from_ref") or "")
print(d.get("built_from_commit") or "")
print(d.get("artifact_key") or "")
PY
    )
    REG_TORCH_VERSION="${_vals[0]:-}"
    REG_TORCH_CUDA_VERSION="${_vals[1]:-}"
    REG_TORCHVISION_VERSION="${_vals[2]:-}"
    REG_TORCHAUDIO_VERSION="${_vals[3]:-}"
    REG_TRITON_VERSION="${_vals[4]:-}"
    REG_TORCH_INDEX_URL="${_vals[5]:-}"
    REG_WHEEL_FILE="${_vals[6]:-}"
    REG_HF_WHEEL_URL="${_vals[7]:-}"
    REG_SM="${_vals[8]:-}"
    REG_SAGE_VERSION="${_vals[9]:-}"
    REG_BUILT_FROM_REPO="${_vals[10]:-}"
    REG_BUILT_FROM_REF="${_vals[11]:-}"
    REG_BUILT_FROM_COMMIT="${_vals[12]:-}"
    REG_ARTIFACT_KEY="${_vals[13]:-}"

    log "Manifest legado carregado (target_arch=${REG_SM})"
    return 0
  fi

  # Registry mode: match the complete binary/runtime ABI, not only SM + Python.
  readarray -t _vals < <(GPU_SM="$GPU_SM" \
    PYTHON_TAG="$python_tag" \
    CURRENT_TORCH_VERSION="$current_torch" \
    CURRENT_TORCH_CUDA_VERSION="$current_torch_cuda" \
    CURRENT_TRITON_VERSION="$current_triton" \
    "$PYTHON_BIN" - <<PY
import json, os, sys

with open("$REGISTRY_PATH", "r", encoding="utf-8") as f:
    data = json.load(f)

gpu_sm = os.environ["GPU_SM"]
py_tag = os.environ["PYTHON_TAG"]
torch_version = os.environ["CURRENT_TORCH_VERSION"]
torch_cuda_version = os.environ["CURRENT_TORCH_CUDA_VERSION"]
triton_version = os.environ["CURRENT_TRITON_VERSION"]

for entry in data.get("wheels", []):
    if (
        entry.get("sm") == gpu_sm
        and entry.get("python_tag") == py_tag
        and entry.get("torch_version") == torch_version
        and entry.get("torch_cuda_version") == torch_cuda_version
        and entry.get("triton_version") == triton_version
    ):
        print(entry.get("torch_version") or "")
        print(entry.get("torch_cuda_version") or "")
        print(entry.get("torchvision_version") or "")
        print(entry.get("torchaudio_version") or "")
        print(entry.get("triton_version") or "")
        print(entry.get("torch_index_url") or "")
        print(entry.get("wheel_file") or "")
        print(entry.get("hf_wheel_url") or "")
        print(entry.get("sm") or "")
        print(entry.get("sageattention_version") or "")
        print(entry.get("built_from_repo") or "")
        print(entry.get("built_from_ref") or "")
        print(entry.get("built_from_commit") or "")
        print(entry.get("artifact_key") or "")
        sys.exit(0)

# No match
sys.exit(1)
PY
  ) || return 1

  REG_TORCH_VERSION="${_vals[0]:-}"
  REG_TORCH_CUDA_VERSION="${_vals[1]:-}"
  REG_TORCHVISION_VERSION="${_vals[2]:-}"
  REG_TORCHAUDIO_VERSION="${_vals[3]:-}"
  REG_TRITON_VERSION="${_vals[4]:-}"
  REG_TORCH_INDEX_URL="${_vals[5]:-}"
  REG_WHEEL_FILE="${_vals[6]:-}"
  REG_HF_WHEEL_URL="${_vals[7]:-}"
  REG_SM="${_vals[8]:-}"
  REG_SAGE_VERSION="${_vals[9]:-}"
  REG_BUILT_FROM_REPO="${_vals[10]:-}"
  REG_BUILT_FROM_REF="${_vals[11]:-}"
  REG_BUILT_FROM_COMMIT="${_vals[12]:-}"
  REG_ARTIFACT_KEY="${_vals[13]:-}"

  log "Registry entry encontrada para ${lookup_key}"
  return 0
}

validate_registry_entry() {
  if [[ "$ALLOW_UNSAFE_WHEEL" == "1" ]]; then
    warn "ALLOW_UNSAFE_WHEEL=1; pulando validações de registry."
    return 0
  fi

  [[ -n "$REG_WHEEL_FILE" || -n "$REG_HF_WHEEL_URL" ]] || return 1

  local current_torch current_torch_cuda current_triton
  current_torch="$(current_package_version torch)" || return 1
  current_torch_cuda="$(current_torch_cuda_version)" || return 1
  current_triton="$(current_package_version triton)" || return 1

  if [[ -z "$REG_TORCH_VERSION" || "$REG_TORCH_VERSION" != "$current_torch" ]]; then
    warn "Wheel torch=${REG_TORCH_VERSION:-desconhecido} != runtime torch=${current_torch}; build necessário."
    return 1
  fi

  if [[ -z "$REG_TRITON_VERSION" || "$REG_TRITON_VERSION" != "$current_triton" ]]; then
    warn "Wheel triton=${REG_TRITON_VERSION:-desconhecido} != runtime triton=${current_triton}; build necessário."
    return 1
  fi

  if [[ "$REGISTRY_IS_LEGACY" != "true" ]] && \
    [[ -z "$REG_TORCH_CUDA_VERSION" || "$REG_TORCH_CUDA_VERSION" != "$current_torch_cuda" ]]; then
    warn "Wheel torch CUDA=${REG_TORCH_CUDA_VERSION:-desconhecido} != runtime CUDA=${current_torch_cuda}; build necessário."
    return 1
  fi

  # Version check
  if [[ -n "$SAGE_EXPECT_VERSION" && -n "$REG_SAGE_VERSION" && "$REG_SAGE_VERSION" != "$SAGE_EXPECT_VERSION" ]]; then
    warn "Registry sageattention_version=${REG_SAGE_VERSION} != ${SAGE_EXPECT_VERSION}"
    return 1
  fi

  # SM compatibility check (for legacy manifests)
  if [[ -n "$REG_SM" ]]; then
    # Check if the registry entry SM matches current GPU
    # Legacy format may have "12.0" or "120" or "sm_120"
    local entry_sm_num=""
    if [[ "$REG_SM" == sm_* ]]; then
      entry_sm_num="${REG_SM#sm_}"
    elif [[ "$REG_SM" == *"."* ]]; then
      entry_sm_num="$(cc_to_sm_num "$REG_SM")"
    else
      entry_sm_num="$REG_SM"
    fi

    if [[ "$entry_sm_num" != "$GPU_SM_NUM" ]]; then
      warn "Registry SM (${REG_SM} -> ${entry_sm_num}) != GPU atual (${GPU_SM_NUM})"
      return 1
    fi
  fi

  # Source repo check
  if [[ -n "$REG_BUILT_FROM_REPO" && "$REG_BUILT_FROM_REPO" != *"thu-ml/SageAttention"* ]]; then
    warn "Registry built_from_repo inesperado: $REG_BUILT_FROM_REPO"
    return 1
  fi

  # Source ref check
  if [[ -n "$REG_BUILT_FROM_REF" && "$REG_BUILT_FROM_REF" != "$SAGE_SOURCE_REF" ]]; then
    warn "Registry built_from_ref=${REG_BUILT_FROM_REF} != ${SAGE_SOURCE_REF}"
    return 1
  fi

  return 0
}

install_from_registry() {
  [[ -n "$REGISTRY_PATH" ]] || return 1
  parse_registry_entry || return 1
  validate_registry_entry || return 1

  local wheel_url=""

  if [[ -n "$REG_HF_WHEEL_URL" ]]; then
    wheel_url="$REG_HF_WHEEL_URL"
  elif [[ -n "$REG_WHEEL_FILE" ]]; then
    if [[ "$REGISTRY_IS_LEGACY" == "true" ]]; then
      # Legacy: flat path
      wheel_url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" \
        "${REMOTE_DIR}/${REG_WHEEL_FILE}")"
    else
      # New: stack-specific subdirectory so different torch ABIs never collide.
      local artifact_key
      artifact_key="${REG_ARTIFACT_KEY:-$(runtime_artifact_key)}"
      wheel_url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" \
        "${REMOTE_DIR}/${GPU_SM}/${artifact_key}/${REG_WHEEL_FILE}")"
    fi
  else
    return 1
  fi

  install_wheel_url "$wheel_url"
}

load_registry_if_available() {
  REGISTRY_PATH=""
  if fetch_registry; then
    log "Registry carregado."
  else
    log "Registry não encontrado no HF (normal na primeira execução)."
  fi
}

# ===== Torch Stack (smart — only install if needed) =====

check_torch_stack_ok() {
  # Returns 0 if current torch stack is good enough for this GPU
  GPU_MIN_CUDA="$GPU_MIN_CUDA" \
  GPU_MIN_TRITON="$GPU_MIN_TRITON" \
  GPU_SM_NUM="$GPU_SM_NUM" \
  "$PYTHON_BIN" - <<'PY' 2>/dev/null
import importlib.metadata as md
import os
import sys

def parse_mm(v):
    p = (v or "").split(".")
    if len(p) < 2:
        return None
    try:
        return (int(p[0]), int(p[1]))
    except Exception:
        return None

try:
    import torch
except ImportError:
    sys.exit(1)

if not torch.cuda.is_available():
    sys.exit(1)

# Check CUDA version in torch
cuda_v = (torch.version.cuda or "").strip()
min_cuda = parse_mm(os.environ.get("GPU_MIN_CUDA", "11.0"))
cuda_mm = parse_mm(cuda_v)
if not cuda_mm or not min_cuda or cuda_mm < min_cuda:
    sys.exit(1)

# Check triton
try:
    triton_v = md.version("triton")
    min_triton = parse_mm(os.environ.get("GPU_MIN_TRITON", "2.0"))
    triton_mm = parse_mm(triton_v)
    if not triton_mm or not min_triton or triton_mm < min_triton:
        sys.exit(1)
except Exception:
    sys.exit(1)

# Check arch support
sm_num = os.environ.get("GPU_SM_NUM", "")
if sm_num:
    arch_list = torch.cuda.get_arch_list()
    if arch_list and not any(sm_num in a for a in arch_list):
        sys.exit(1)

print(f"[INFO] Stack atual OK: torch={torch.__version__}, cuda={cuda_v}, triton={triton_v}")
sys.exit(0)
PY
}

validate_torch_stack() {
  GPU_MIN_CUDA="$GPU_MIN_CUDA" \
  GPU_MIN_TRITON="$GPU_MIN_TRITON" \
  GPU_SM_NUM="$GPU_SM_NUM" \
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
import torch

def parse_mm(v):
    p = (v or "").split(".")
    if len(p) < 2:
        return None
    try:
        return (int(p[0]), int(p[1]))
    except Exception:
        return None

if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA não disponível após instalação do stack.")

cuda_v = (torch.version.cuda or "").strip()
triton_v = md.version("triton")
min_cuda = parse_mm(os.environ.get("GPU_MIN_CUDA", "11.0"))
min_triton = parse_mm(os.environ.get("GPU_MIN_TRITON", "2.0"))
cuda_mm = parse_mm(cuda_v)
triton_mm = parse_mm(triton_v)

print(f"[INFO] torch: {torch.__version__}")
print(f"[INFO] torchvision: {md.version('torchvision')}")
print(f"[INFO] torchaudio: {md.version('torchaudio')}")
print(f"[INFO] triton: {triton_v}")
print(f"[INFO] torch.version.cuda: {cuda_v}")
print(f"[INFO] device: {torch.cuda.get_device_name(0)}")
print(f"[INFO] capability: {torch.cuda.get_device_capability(0)}")
print(f"[INFO] arch_list: {torch.cuda.get_arch_list()}")

if not cuda_mm or not min_cuda or cuda_mm < min_cuda:
    raise SystemExit(f"[ERROR] CUDA do torch ({cuda_v}) < {os.environ.get('GPU_MIN_CUDA')}")

if not triton_mm or not min_triton or triton_mm < min_triton:
    raise SystemExit(f"[ERROR] Triton ({triton_v}) < {os.environ.get('GPU_MIN_TRITON')}")

sm_num = os.environ.get("GPU_SM_NUM", "")
if sm_num:
    arch_list = torch.cuda.get_arch_list()
    if arch_list and not any(sm_num in a for a in arch_list):
        raise SystemExit(f"[ERROR] Torch não expõe sm_{sm_num} na arch_list.")
PY
}

ensure_triton() {
  if [[ -n "$TRITON_VERSION" ]]; then
    log "Instalando triton fixo: ${TRITON_VERSION}"
    run_pip install --force-reinstall "triton==${TRITON_VERSION}"
    return
  fi

  if GPU_MIN_TRITON="$GPU_MIN_TRITON" "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
import re
import sys

def parse_mm(v):
    m = re.search(r"([0-9]+)\.([0-9]+)", v or "")
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))

min_v = parse_mm(os.environ.get("GPU_MIN_TRITON", "2.0"))
try:
    cur_raw = md.version("triton")
except Exception:
    sys.exit(2)

cur_v = parse_mm(cur_raw)
if not cur_v or not min_v:
    sys.exit(2)

print(f"[INFO] triton atual: {cur_raw}")
if cur_v < min_v:
    sys.exit(3)
PY
  then
    log "Mantendo triton atual (já compatível)."
  else
    case "$?" in
      3) log "Triton abaixo do mínimo para ${GPU_SM}; atualizando para ${TRITON_SPEC}" ;;
      *) log "Triton ausente/inválido; instalando ${TRITON_SPEC}" ;;
    esac
    run_pip install --force-reinstall -U "$TRITON_SPEC"
  fi
}

install_torch_stack() {
  # If user says skip, skip
  if [[ "$SKIP_TORCH_INSTALL" == "1" ]]; then
    log "SKIP_TORCH_INSTALL=1; pulando instalação do torch stack."
    validate_torch_stack
    return
  fi

  # Check if current stack already works
  if check_torch_stack_ok; then
    log "Torch stack atual já é compatível com ${GPU_SM}. Nenhuma instalação necessária."
    return
  fi

  log "Torch stack precisa ser instalado/atualizado para ${GPU_SM}."

  local mode idx_url torch_pkg tv_pkg ta_pkg
  mode="default"
  idx_url="${TORCH_INDEX_URL:-}"

  if [[ -n "$TORCH_VERSION" && -n "$TORCHVISION_VERSION" && -n "$TORCHAUDIO_VERSION" ]]; then
    mode="env-pin"
    torch_pkg="torch==${TORCH_VERSION}"
    tv_pkg="torchvision==${TORCHVISION_VERSION}"
    ta_pkg="torchaudio==${TORCHAUDIO_VERSION}"
  elif [[ -n "$REG_TORCH_VERSION" && -n "$REG_TORCHVISION_VERSION" && -n "$REG_TORCHAUDIO_VERSION" ]]; then
    mode="registry-pin"
    torch_pkg="torch==${REG_TORCH_VERSION}"
    tv_pkg="torchvision==${REG_TORCHVISION_VERSION}"
    ta_pkg="torchaudio==${REG_TORCHAUDIO_VERSION}"
    idx_url="${idx_url:-$REG_TORCH_INDEX_URL}"
  else
    mode="default"
    torch_pkg="torch"
    tv_pkg="torchvision"
    ta_pkg="torchaudio"
  fi

  idx_url="${idx_url:-$(default_torch_index_url)}"
  TORCH_INDEX_URL_USED="$idx_url"
  export TORCH_INDEX_URL_USED

  log "Instalando stack torch/triton (modo=${mode}, channel=${TORCH_CHANNEL})"
  log "Torch index: $idx_url"

  if [[ "$mode" == "default" && "$TORCH_CHANNEL" == "nightly" ]]; then
    run_pip install --force-reinstall --pre "$torch_pkg" "$tv_pkg" "$ta_pkg" --index-url "$idx_url"
  else
    run_pip install --force-reinstall "$torch_pkg" "$tv_pkg" "$ta_pkg" --index-url "$idx_url"
  fi

  ensure_triton
  validate_torch_stack
}

# ===== CUDA Home (for build) =====

ensure_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    :
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
  else
    # Scan common paths
    local d found=""
    for d in /usr/local/cuda /usr/local/cuda-*; do
      if [[ -x "$d/bin/nvcc" ]]; then
        found="$d"
      fi
    done
    if [[ -n "$found" ]]; then
      CUDA_HOME="$found"
      export CUDA_HOME
      export PATH="$CUDA_HOME/bin:$PATH"
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    else
      die "nvcc não encontrado. Precisa de CUDA Toolkit >= ${GPU_MIN_CUDA} para build."
    fi
  fi

  log "CUDA_HOME: ${CUDA_HOME}"
}

# ===== Wheel helpers =====

get_latest_local_wheel() {
  local search_dir="${1:-$WHEELHOUSE_DIR}"
  SEARCH_DIR="$search_dir" SAGE_EXPECT_VERSION="$SAGE_EXPECT_VERSION" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["SEARCH_DIR"])
version = os.environ.get("SAGE_EXPECT_VERSION", "")
pattern = f"sageattention-{version}-*.whl" if version else "sageattention-*.whl"
wheels = list(root.rglob(pattern)) if root.is_dir() else []
if not wheels:
    raise SystemExit(1)
print(max(wheels, key=lambda path: path.stat().st_mtime))
PY
}

validate_sage_version() {
  SAGE_EXPECT_VERSION="$SAGE_EXPECT_VERSION" "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
import sageattention

v = md.version("sageattention")
print(f"[INFO] sageattention instalado e importável: {v}")
expected = (os.environ.get("SAGE_EXPECT_VERSION") or "").strip()
if expected and v != expected:
    raise SystemExit(f"[ERROR] sageattention instalado ({v}) difere do esperado ({expected}).")
PY
}

install_wheel_file() {
  local wheel_path="$1"
  [[ -f "$wheel_path" ]] || return 1
  run_pip install --force-reinstall "$wheel_path" || return 1
  validate_sage_version
}

install_wheel_url() {
  local wheel_url="$1"
  [[ -n "$wheel_url" ]] || return 1
  log "Instalando wheel: $wheel_url"
  run_pip install --force-reinstall "$wheel_url" || return 1
  validate_sage_version
}

wheel_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

# ===== Registry Build & Publish =====

registry_local_path() {
  printf '%s/%s/registry.json' "$WHEELHOUSE_DIR" "$REMOTE_DIR"
}

latest_local_path() {
  printf '%s/%s/latest.json' "$WHEELHOUSE_DIR" "$REMOTE_DIR"
}

build_registry_entry() {
  local wheel_path="$1"
  local wheel_file sha registry_out latest_out hf_wheel_url artifact_key

  wheel_file="$(basename "$wheel_path")"
  sha="$(wheel_sha256 "$wheel_path")"
  artifact_key="$(runtime_artifact_key)"

  registry_out="$(registry_local_path)"
  latest_out="$(latest_local_path)"
  mkdir -p "$(dirname "$registry_out")"

  hf_wheel_url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" \
    "${REMOTE_DIR}/${GPU_SM}/${artifact_key}/${wheel_file}")"

  WHEEL_FILE="$wheel_file" \
  WHEEL_SHA256="$sha" \
  HF_WHEEL_URL="$hf_wheel_url" \
  ARTIFACT_KEY="$artifact_key" \
  TORCH_INDEX_URL_USED="${TORCH_INDEX_URL_USED:-}" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  CUDAARCHS="$CUDAARCHS" \
  GPU_SM="$GPU_SM" \
  GPU_CC="$GPU_CC" \
  GPU_NAME="$GPU_NAME" \
  GPU_ARCH_NAME="$GPU_ARCH_NAME" \
  GPU_MIN_CUDA="$GPU_MIN_CUDA" \
  SAGE_EXPECT_VERSION="$SAGE_EXPECT_VERSION" \
  SAGE_SOURCE_REPO="$SAGE_SOURCE_REPO" \
  SAGE_SOURCE_REF="$SAGE_SOURCE_REF" \
  SAGE_SOURCE_COMMIT="${SAGE_SOURCE_COMMIT:-}" \
  REGISTRY_OUT="$registry_out" \
  LATEST_OUT="$latest_out" \
  "$PYTHON_BIN" - <<'PY'
import datetime
import importlib.metadata as md
import json
import os
import platform
import sys
import torch


def pkg(name):
    try:
        return md.version(name)
    except Exception:
        return None


now_utc = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
gpu_sm = os.environ["GPU_SM"]
artifact_key = os.environ["ARTIFACT_KEY"]
entry_key = f"{gpu_sm}-{python_tag}-{artifact_key}"

entry = {
    "key": entry_key,
    "sm": gpu_sm,
    "compute_capability": os.environ["GPU_CC"],
    "python_tag": python_tag,
    "artifact_key": artifact_key,
    "wheel_file": os.environ["WHEEL_FILE"],
    "wheel_sha256": os.environ["WHEEL_SHA256"],
    "hf_wheel_url": os.environ["HF_WHEEL_URL"],
    "sageattention_version": pkg("sageattention"),
    "torch_version": pkg("torch"),
    "torch_cuda_version": torch.version.cuda,
    "torchvision_version": pkg("torchvision"),
    "torchaudio_version": pkg("torchaudio"),
    "triton_version": pkg("triton"),
    "torch_index_url": os.environ.get("TORCH_INDEX_URL_USED") or None,
    "target_arch": os.environ["TORCH_CUDA_ARCH_LIST"],
    "cudaarchs": os.environ["CUDAARCHS"],
    "required_min_cuda": os.environ.get("GPU_MIN_CUDA") or None,
    "built_for": f"{gpu_sm}-{os.environ.get('GPU_ARCH_NAME', '')}",
    "built_from_repo": os.environ.get("SAGE_SOURCE_REPO") or None,
    "built_from_ref": os.environ.get("SAGE_SOURCE_REF") or None,
    "built_from_commit": os.environ.get("SAGE_SOURCE_COMMIT") or None,
    "builder_gpu_name": os.environ.get("GPU_NAME") or None,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "built_at_utc": now_utc,
}

# --- Update registry.json ---
registry_path = os.environ["REGISTRY_OUT"]
registry = {"schema_version": 2, "updated_at_utc": now_utc, "wheels": []}

# Load existing registry if present
if os.path.isfile(registry_path):
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except Exception:
        pass

# Upsert entry by key
wheels = registry.get("wheels", [])
replaced = False
for i, w in enumerate(wheels):
    if w.get("key") == entry_key:
        wheels[i] = entry
        replaced = True
        break
if not replaced:
    wheels.append(entry)

registry["wheels"] = wheels
registry["schema_version"] = 2
registry["updated_at_utc"] = now_utc

with open(registry_path, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)
print(f"[INFO] Registry atualizado: {registry_path} (key={entry_key})")

# --- Also write latest.json for backward compat ---
latest_path = os.environ["LATEST_OUT"]
latest = dict(entry)
latest["created_at_utc"] = now_utc
with open(latest_path, "w", encoding="utf-8") as f:
    json.dump(latest, f, indent=2)
print(f"[INFO] latest.json gerado: {latest_path}")
PY
}

# ===== HF Publish =====

get_hf_python() {
  local hf_python hf_venv
  hf_python="$PYTHON_BIN"
  hf_venv="$WORK_DIR/.hf_venv"

  if "$hf_python" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "$hf_python"
    return 0
  fi

  if run_pip install -q -U huggingface_hub >/dev/null 2>&1; then
    echo "$hf_python"
    return 0
  fi

  log "Criando venv auxiliar para huggingface_hub em $hf_venv"
  [[ -x "$hf_venv/bin/python" ]] || "$PYTHON_BIN" -m venv "$hf_venv"
  "$hf_venv/bin/python" -m pip install -q -U pip
  "$hf_venv/bin/python" -m pip install -q -U huggingface_hub
  echo "$hf_venv/bin/python"
}

ensure_hf_repo() {
  [[ -n "$HF_REPO_ID" ]] || die "HF_REPO_ID não definido."
  [[ -n "$HF_TOKEN" ]] || die "HF_TOKEN não definido."

  local hf_python
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

merge_remote_registry() {
  # Download remote registry.json and merge with local (local wins for same key)
  local remote_url out_remote local_registry
  remote_url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" "${REMOTE_DIR}/registry.json")"
  out_remote="$WORK_DIR/hf-registry-remote-merge.json"
  local_registry="$(registry_local_path)"

  if hf_curl "$remote_url" "$out_remote" && [[ -s "$out_remote" ]]; then
    REMOTE_REGISTRY="$out_remote" \
    LOCAL_REGISTRY="$local_registry" \
    "$PYTHON_BIN" - <<'PY'
import json
import os

remote_path = os.environ["REMOTE_REGISTRY"]
local_path = os.environ["LOCAL_REGISTRY"]

with open(remote_path, "r", encoding="utf-8") as f:
    remote = json.load(f)
with open(local_path, "r", encoding="utf-8") as f:
    local = json.load(f)

# Build dict from remote, then overlay local entries
merged = {}
for w in remote.get("wheels", []):
    k = w.get("key")
    if k:
        merged[k] = w

for w in local.get("wheels", []):
    k = w.get("key")
    if k:
        merged[k] = w  # local wins

result = {
    "schema_version": 2,
    "updated_at_utc": local.get("updated_at_utc", remote.get("updated_at_utc", "")),
    "wheels": list(merged.values()),
}

with open(local_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"[INFO] Registry merged: {len(merged)} entries")
PY
  else
    log "Nenhum registry remoto para merge (primeiro upload)."
  fi
}

publish_to_hf() {
  local wheel_path="$1"
  [[ -f "$wheel_path" ]] || die "Wheel para upload não encontrada: $wheel_path"
  [[ -n "$HF_TOKEN" ]] || die "HF_TOKEN obrigatório para publicar no HF."

  ensure_hf_repo
  merge_remote_registry

  local hf_python registry_json latest_json wheel_file artifact_key
  hf_python="$(get_hf_python)"
  registry_json="$(registry_local_path)"
  latest_json="$(latest_local_path)"
  wheel_file="$(basename "$wheel_path")"
  artifact_key="$(runtime_artifact_key)"

  [[ -f "$registry_json" ]] || die "Registry não encontrado: $registry_json"
  [[ -f "$latest_json" ]] || die "latest.json não encontrado: $latest_json"

  WHEEL_PATH="$wheel_path" \
  WHEEL_REMOTE="${REMOTE_DIR}/${GPU_SM}/${artifact_key}/${wheel_file}" \
  REGISTRY_PATH="$registry_json" \
  REGISTRY_REMOTE="${REMOTE_DIR}/registry.json" \
  LATEST_PATH="$latest_json" \
  LATEST_REMOTE="${REMOTE_DIR}/latest.json" \
  HF_REPO_ID="$HF_REPO_ID" \
  HF_REPO_TYPE="$HF_REPO_TYPE" \
  HF_TOKEN="$HF_TOKEN" \
  "$hf_python" - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "dataset")

uploads = [
    (os.environ["WHEEL_PATH"], os.environ["WHEEL_REMOTE"]),
    (os.environ["REGISTRY_PATH"], os.environ["REGISTRY_REMOTE"]),
    (os.environ["LATEST_PATH"], os.environ["LATEST_REMOTE"]),
]

for local_path, remote_path in uploads:
    print(f"[INFO] Upload {local_path} -> {remote_path}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

print("[INFO] Upload para HF concluído")
PY
}

# ===== Build =====

build_wheel() {
  require_cmd git
  ensure_cuda_home

  run_pip install -U ninja cmake packaging

  local stamp src_dir build_log wheel_path ref_slug artifact_key build_wheel_dir
  LAST_BUILT_WHEEL=""
  SAGE_SOURCE_COMMIT=""
  stamp="$(date +%Y%m%d-%H%M%S)"
  ref_slug="$(printf '%s' "$SAGE_SOURCE_REF" | tr '/:' '__')"
  src_dir="$WORK_DIR/SageAttention-${ref_slug}-${stamp}"
  build_log="$LOG_DIR/build-sageattention-${ref_slug}-${stamp}.log"
  artifact_key="$(runtime_artifact_key)"
  build_wheel_dir="$WHEELHOUSE_DIR/builds/$artifact_key"
  mkdir -p "$build_wheel_dir"

  log "Clonando SageAttention (${SAGE_SOURCE_REPO} @ ${SAGE_SOURCE_REF})"
  git clone --depth 1 --branch "$SAGE_SOURCE_REF" "$SAGE_SOURCE_REPO" "$src_dir"
  SAGE_SOURCE_COMMIT="$(git -C "$src_dir" rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "$SAGE_SOURCE_COMMIT" ]]; then
    log "Source commit: $SAGE_SOURCE_COMMIT"
  fi

  export TORCH_CUDA_ARCH_LIST
  export CUDAARCHS
  export EXT_PARALLEL
  export MAX_JOBS
  export NVCC_APPEND_FLAGS

  log "Buildando wheel para ${GPU_SM} (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, CUDAARCHS=${CUDAARCHS})"
  run_pip wheel "$src_dir" --no-build-isolation --wheel-dir "$build_wheel_dir" 2>&1 | tee "$build_log"

  # Dynamic build log verification
  local build_pattern="sm_${GPU_SM_NUM}|compute_${GPU_SM_NUM}|${GPU_CC}\\+PTX|arch=compute_${GPU_SM_NUM}|code=sm_${GPU_SM_NUM}"
  if grep -Eq "$build_pattern" "$build_log"; then
    log "Build log indica target ${GPU_SM} (${GPU_CC})."
  else
    warn "Build log não mostrou explicitamente ${GPU_SM}; continuando com validação pós-instalação."
  fi

  wheel_path="$(get_latest_local_wheel "$build_wheel_dir" || true)"
  [[ -n "$wheel_path" ]] || die "Nenhuma wheel encontrada após build."

  install_wheel_file "$wheel_path"
  build_registry_entry "$wheel_path"
  LAST_BUILT_WHEEL="$wheel_path"
}

# ===== Runtime Validation =====

validate_runtime() {
  SAGE_EXPECT_VERSION="$SAGE_EXPECT_VERSION" "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
import sageattention
import torch

sv = md.version("sageattention")
expected = (os.environ.get("SAGE_EXPECT_VERSION") or "").strip()
if expected and sv != expected:
    raise SystemExit(f"[ERROR] sageattention={sv}, esperado={expected}")

print(f"[INFO] Validação final: sageattention={sv}")
print("[INFO] Validação final: import sageattention OK")
print(f"[INFO] Validação final: torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"[INFO] Validação final: cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] Validação final: device={torch.cuda.get_device_name(0)}")
    print(f"[INFO] Validação final: capability={torch.cuda.get_device_capability(0)}")
PY
}

# ===== Main =====

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
  log "Script version: ${SCRIPT_VERSION}"
  ensure_pip_ready
  detect_gpu
  detect_system_cuda

  log "=== Configuração ==="
  log "GPU: ${GPU_NAME} (${GPU_ARCH_NAME}, ${GPU_SM}, CC ${GPU_CC})"
  log "CUDA sistema: ${SYSTEM_CUDA_VERSION:-não detectado}"
  log "Torch channel: ${TORCH_CHANNEL}"
  log "CUDA index: ${CUDA_INDEX_VARIANT}"
  log "Build arch: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, CUDAARCHS=${CUDAARCHS}"
  log "Triton spec: ${TRITON_SPEC}"
  log "==================="

  case "$ACTION" in
    init-hf)
      route_hf_for_torch_cuda13
      ensure_hf_repo
      log "Concluído."
      exit 0
      ;;

    install)
      install_torch_stack
      load_registry_if_available

      if [[ -n "$WHEEL_URL" ]]; then
        install_wheel_url "$WHEEL_URL" || die "Falha ao instalar WHEEL_URL informado."
      else
        install_from_registry || die "Wheel compatível para ${GPU_SM} não encontrada no HF."
      fi

      validate_runtime
      ;;

    build)
      install_torch_stack
      load_registry_if_available
      build_wheel
      [[ -n "$LAST_BUILT_WHEEL" ]] || die "Build concluído sem caminho de wheel."
      if [[ -n "$HF_TOKEN" ]]; then
        publish_to_hf "$LAST_BUILT_WHEEL"
      else
        warn "HF_TOKEN ausente: wheel compilada/instalada, mas não publicada."
      fi
      validate_runtime
      ;;

    publish)
      local wheel_to_publish artifact_key build_wheel_dir
      route_hf_for_torch_cuda13
      artifact_key="$(runtime_artifact_key)"
      build_wheel_dir="$WHEELHOUSE_DIR/builds/$artifact_key"
      wheel_to_publish="$(get_latest_local_wheel "$build_wheel_dir" || true)"
      [[ -n "$wheel_to_publish" ]] || die "Nenhuma wheel local encontrada para publish."
      build_registry_entry "$wheel_to_publish"
      publish_to_hf "$wheel_to_publish"
      ;;

    auto)
      install_torch_stack
      load_registry_if_available

      if [[ -n "$WHEEL_URL" ]]; then
        if install_wheel_url "$WHEEL_URL"; then
          log "Wheel explícita instalada com sucesso."
          validate_runtime
          log "Concluído."
          exit 0
        fi
        warn "WHEEL_URL falhou; seguindo para fluxo HF/build."
      fi

      if install_from_registry; then
        log "Wheel instalada do HF para ${GPU_SM} (sem rebuild)."
      else
        log "Wheel compatível para ${GPU_SM} não encontrada no HF. Build do zero."
        build_wheel
        [[ -n "$LAST_BUILT_WHEEL" ]] || die "Build concluído sem caminho de wheel."

        if [[ -n "$HF_TOKEN" ]]; then
          publish_to_hf "$LAST_BUILT_WHEEL"
        else
          warn "HF_TOKEN ausente: wheel compilada/instalada, mas não publicada no HF."
        fi
      fi

      validate_runtime
      ;;
  esac

  log "Concluído."
}

if [[ "${SAGE_INSTALLER_TEST_MODE:-0}" != "1" ]]; then
  main "$@"
fi
