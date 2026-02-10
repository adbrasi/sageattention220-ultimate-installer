#!/usr/bin/env bash
set -Eeuo pipefail

# Ultimate SageAttention 2.2.0 installer for RTX 5090 (sm_120)
# Safe flow:
# 1) Always install/validate torch+triton stack first
# 2) Install only wheel from your own HF repo manifest
# 3) If unavailable/incompatible, build from source (v2.2.0, sm_120), install, and publish to HF

ACTION="${1:-auto}"
SCRIPT_VERSION="${SCRIPT_VERSION:-2026-02-10-a71f418}"

SAGE_VERSION="${SAGE_VERSION:-2.2.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Torch stack defaults for Blackwell/5090
TORCH_CHANNEL="${TORCH_CHANNEL:-nightly}"          # nightly|stable
CUDA_INDEX_VARIANT="${CUDA_INDEX_VARIANT:-cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TRITON_SPEC="${TRITON_SPEC:-triton>=3.3,<4.0}"

# Optional exact pins (override defaults/manifest)
TORCH_VERSION="${TORCH_VERSION:-}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-}"
TRITON_VERSION="${TRITON_VERSION:-}"

# Build target for RTX 5090
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
CUDAARCHS="${CUDAARCHS:-120}"
EXT_PARALLEL="${EXT_PARALLEL:-4}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:---threads 8}"

# Minimum versions for safety
MIN_CUDA_FOR_SM120="${MIN_CUDA_FOR_SM120:-12.8}"
MIN_TRITON_FOR_50XX="${MIN_TRITON_FOR_50XX:-3.3}"

# HF artifact storage (OWN repo, not third-party prebuilt)
HF_REPO_ID="${HF_REPO_ID:-adbrasi/sageattention220-wheels}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}" # dataset|model|space
HF_REPO_BRANCH="${HF_REPO_BRANCH:-main}"
HF_PRIVATE="${HF_PRIVATE:-false}"
HF_TOKEN="${HF_TOKEN:-}"
REMOTE_DIR="${REMOTE_DIR:-sageattention220}"

# Optional explicit URL (only used if YOU set it)
WHEEL_URL="${WHEEL_URL:-}"

# Manifest safety
ALLOW_UNSAFE_WHEEL="${ALLOW_UNSAFE_WHEEL:-0}"

WORK_DIR="${WORK_DIR:-$PWD/.build-sageattention}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-$PWD/wheelhouse}"
LOG_DIR="${LOG_DIR:-$PWD/logs}"
mkdir -p "$WORK_DIR" "$WHEELHOUSE_DIR" "$LOG_DIR"

GPU_NAME=""
GPU_CC=""
MANIFEST_PATH=""

MANIFEST_TORCH_VERSION=""
MANIFEST_TORCHVISION_VERSION=""
MANIFEST_TORCHAUDIO_VERSION=""
MANIFEST_TRITON_VERSION=""
MANIFEST_TORCH_INDEX_URL=""
MANIFEST_WHEEL_FILE=""
MANIFEST_HF_WHEEL_URL=""
MANIFEST_TARGET_ARCH=""
MANIFEST_SAGE_VERSION=""
MANIFEST_BUILT_FROM_REPO=""
MANIFEST_BUILT_FROM_REF=""

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

Ações:
  auto    -> instala torch/triton, tenta wheel do seu HF; se não existir/for inválida, compila e publica
  install -> instala torch/triton e instala somente wheel do seu HF (falha se não houver)
  build   -> força build local do SageAttention v2.2.0 (sm_120), instala e publica no HF
  publish -> publica no HF a wheel local mais recente + latest.json
  init-hf -> cria/valida o repositório HF

Padrões seguros:
  HF_REPO_ID=adbrasi/sageattention220-wheels
  HF_REPO_TYPE=dataset
  TORCH_CHANNEL=nightly
  CUDA_INDEX_VARIANT=cu128
  TRITON_SPEC=triton>=3.3,<4.0
  TORCH_CUDA_ARCH_LIST=12.0
  CUDAARCHS=120
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
  cmd="${1:-}"
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

ensure_pip_ready() {
  run_pip install -U pip setuptools wheel
}

detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 | xargs || true)"
    GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | xargs || true)"

    log "GPU detectada: ${GPU_NAME:-desconhecida}"
    log "Compute capability: ${GPU_CC:-desconhecida}"

    if [[ "${GPU_NAME}" != *"RTX 5090"* ]]; then
      warn "GPU não parece RTX 5090. Continuando por sua conta."
    fi
    if [[ -n "$GPU_CC" && "$GPU_CC" != 12.0* ]]; then
      warn "Compute capability não é 12.0 (sm_120)."
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
    die "TORCH_CHANNEL inválido: $TORCH_CHANNEL (nightly|stable)"
  fi
}

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

fetch_hf_manifest() {
  local url out
  url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" "${REMOTE_DIR}/latest.json")"
  out="$WORK_DIR/hf-latest.json"

  if [[ -n "$HF_TOKEN" ]]; then
    curl -fsSL -H "Authorization: Bearer $HF_TOKEN" "$url" -o "$out" || return 1
  else
    curl -fsSL "$url" -o "$out" || return 1
  fi

  [[ -s "$out" ]] || return 1
  MANIFEST_PATH="$out"
  return 0
}

parse_manifest_stack() {
  [[ -n "$MANIFEST_PATH" && -f "$MANIFEST_PATH" ]] || return 0

  readarray -t _vals < <("$PYTHON_BIN" - <<PY
import json
with open("$MANIFEST_PATH", "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("torch_version") or "")
print(d.get("torchvision_version") or "")
print(d.get("torchaudio_version") or "")
print(d.get("triton_version") or "")
print(d.get("torch_index_url") or "")
print(d.get("wheel_file") or "")
print(d.get("hf_wheel_url") or "")
print(d.get("target_arch") or "")
print(d.get("sageattention_version") or "")
print(d.get("built_from_repo") or "")
print(d.get("built_from_ref") or "")
PY
)

  MANIFEST_TORCH_VERSION="${_vals[0]:-}"
  MANIFEST_TORCHVISION_VERSION="${_vals[1]:-}"
  MANIFEST_TORCHAUDIO_VERSION="${_vals[2]:-}"
  MANIFEST_TRITON_VERSION="${_vals[3]:-}"
  MANIFEST_TORCH_INDEX_URL="${_vals[4]:-}"
  MANIFEST_WHEEL_FILE="${_vals[5]:-}"
  MANIFEST_HF_WHEEL_URL="${_vals[6]:-}"
  MANIFEST_TARGET_ARCH="${_vals[7]:-}"
  MANIFEST_SAGE_VERSION="${_vals[8]:-}"
  MANIFEST_BUILT_FROM_REPO="${_vals[9]:-}"
  MANIFEST_BUILT_FROM_REF="${_vals[10]:-}"

  log "Manifest carregado de $MANIFEST_PATH"
}

manifest_is_safe_for_5090() {
  if [[ "$ALLOW_UNSAFE_WHEEL" == "1" ]]; then
    warn "ALLOW_UNSAFE_WHEEL=1 ativo; pulando validações estritas de manifest."
    return 0
  fi

  [[ -n "$MANIFEST_WHEEL_FILE" || -n "$MANIFEST_HF_WHEEL_URL" ]] || return 1

  if [[ -n "$MANIFEST_SAGE_VERSION" && "$MANIFEST_SAGE_VERSION" != "$SAGE_VERSION" ]]; then
    warn "Manifest sageattention_version=${MANIFEST_SAGE_VERSION} != ${SAGE_VERSION}"
    return 1
  fi

  if [[ -z "$MANIFEST_TARGET_ARCH" ]]; then
    warn "Manifest sem target_arch; não confiável."
    return 1
  fi
  if [[ "$MANIFEST_TARGET_ARCH" != *"12.0"* && "$MANIFEST_TARGET_ARCH" != *"120"* ]]; then
    warn "Manifest target_arch=${MANIFEST_TARGET_ARCH} não indica sm_120."
    return 1
  fi

  if [[ -n "$MANIFEST_BUILT_FROM_REPO" && "$MANIFEST_BUILT_FROM_REPO" != *"thu-ml/SageAttention"* ]]; then
    warn "Manifest built_from_repo inesperado: $MANIFEST_BUILT_FROM_REPO"
    return 1
  fi

  if [[ -n "$MANIFEST_BUILT_FROM_REF" && "$MANIFEST_BUILT_FROM_REF" != "v${SAGE_VERSION}" ]]; then
    warn "Manifest built_from_ref inesperado: $MANIFEST_BUILT_FROM_REF"
    return 1
  fi

  return 0
}

validate_torch_stack() {
  MIN_CUDA_FOR_SM120="$MIN_CUDA_FOR_SM120" \
  MIN_TRITON_FOR_50XX="$MIN_TRITON_FOR_50XX" \
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
min_cuda = parse_mm(os.environ.get("MIN_CUDA_FOR_SM120", "12.8"))
min_triton = parse_mm(os.environ.get("MIN_TRITON_FOR_50XX", "3.3"))
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
    raise SystemExit(f"[ERROR] CUDA do torch ({cuda_v}) < {os.environ.get('MIN_CUDA_FOR_SM120', '12.8')}")

if not triton_mm or not min_triton or triton_mm < min_triton:
    raise SystemExit(f"[ERROR] Triton ({triton_v}) < {os.environ.get('MIN_TRITON_FOR_50XX', '3.3')}")

cap = torch.cuda.get_device_capability(0)
if cap[0] == 12:
    if not any("120" in a for a in torch.cuda.get_arch_list()):
        raise SystemExit("[ERROR] Torch não expõe sm_120 na arch_list para GPU Blackwell.")
PY
}

install_torch_stack() {
  local mode idx_url torch_pkg tv_pkg ta_pkg triton_pkg
  mode="default"
  idx_url="${TORCH_INDEX_URL:-}"

  if [[ -n "$TORCH_VERSION" && -n "$TORCHVISION_VERSION" && -n "$TORCHAUDIO_VERSION" && -n "$TRITON_VERSION" ]]; then
    mode="env-pin"
    torch_pkg="torch==${TORCH_VERSION}"
    tv_pkg="torchvision==${TORCHVISION_VERSION}"
    ta_pkg="torchaudio==${TORCHAUDIO_VERSION}"
    triton_pkg="triton==${TRITON_VERSION}"
  elif [[ -n "$MANIFEST_TORCH_VERSION" && -n "$MANIFEST_TORCHVISION_VERSION" && -n "$MANIFEST_TORCHAUDIO_VERSION" && -n "$MANIFEST_TRITON_VERSION" ]]; then
    mode="manifest-pin"
    torch_pkg="torch==${MANIFEST_TORCH_VERSION}"
    tv_pkg="torchvision==${MANIFEST_TORCHVISION_VERSION}"
    ta_pkg="torchaudio==${MANIFEST_TORCHAUDIO_VERSION}"
    triton_pkg="triton==${MANIFEST_TRITON_VERSION}"
    idx_url="${idx_url:-$MANIFEST_TORCH_INDEX_URL}"
  else
    mode="default"
    torch_pkg="torch"
    tv_pkg="torchvision"
    ta_pkg="torchaudio"
    triton_pkg="$TRITON_SPEC"
  fi

  idx_url="${idx_url:-$(default_torch_index_url)}"
  TORCH_INDEX_URL_USED="$idx_url"
  export TORCH_INDEX_URL_USED

  log "Instalando stack torch/triton (modo=${mode})"
  log "Torch index: $idx_url"

  if [[ "$mode" == "default" && "$TORCH_CHANNEL" == "nightly" ]]; then
    run_pip install --force-reinstall --pre "$torch_pkg" "$tv_pkg" "$ta_pkg" --index-url "$idx_url"
  else
    run_pip install --force-reinstall "$torch_pkg" "$tv_pkg" "$ta_pkg" --index-url "$idx_url"
  fi

  run_pip install --force-reinstall -U "$triton_pkg"

  validate_torch_stack
}

ensure_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    :
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
  elif [[ -d "/usr/local/cuda-12.8" ]]; then
    CUDA_HOME="/usr/local/cuda-12.8"
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  else
    die "nvcc não encontrado. Precisa de CUDA Toolkit >= ${MIN_CUDA_FOR_SM120} para build."
  fi

  MIN_CUDA_FOR_SM120="$MIN_CUDA_FOR_SM120" "$PYTHON_BIN" - <<'PY'
import os
import re
import subprocess

min_v = tuple(map(int, os.environ.get("MIN_CUDA_FOR_SM120", "12.8").split(".")))
out = subprocess.check_output(["nvcc", "--version"], text=True)
print("[INFO] nvcc --version:")
print(out)
m = re.search(r"release\s+([0-9]+)\.([0-9]+)", out)
if not m:
    raise SystemExit("[ERROR] Não foi possível detectar versão do nvcc.")
cur = (int(m.group(1)), int(m.group(2)))
if cur < min_v:
    raise SystemExit(f"[ERROR] nvcc {cur[0]}.{cur[1]} < {min_v[0]}.{min_v[1]} (mínimo para sm_120).")
PY
}

get_latest_local_wheel() {
  shopt -s nullglob
  local wheels=("$WHEELHOUSE_DIR"/sageattention-"$SAGE_VERSION"-*.whl)
  shopt -u nullglob
  (( ${#wheels[@]} > 0 )) || return 1
  ls -1t "${wheels[@]}" | head -n1
}

install_wheel_file() {
  local wheel_path="$1"
  [[ -f "$wheel_path" ]] || return 1

  run_pip install --force-reinstall "$wheel_path" || return 1

  SAGE_VERSION_EXPECTED="$SAGE_VERSION" "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
v = md.version("sageattention")
print(f"[INFO] sageattention instalado: {v}")
if v != os.environ["SAGE_VERSION_EXPECTED"]:
    raise SystemExit(f"[ERROR] sageattention instalado ({v}) difere do esperado ({os.environ['SAGE_VERSION_EXPECTED']}).")
PY
}

install_wheel_url() {
  local wheel_url="$1"
  [[ -n "$wheel_url" ]] || return 1

  log "Tentando instalação por URL: $wheel_url"
  run_pip install --force-reinstall "$wheel_url" || return 1

  SAGE_VERSION_EXPECTED="$SAGE_VERSION" "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
v = md.version("sageattention")
print(f"[INFO] sageattention instalado: {v}")
if v != os.environ["SAGE_VERSION_EXPECTED"]:
    raise SystemExit(f"[ERROR] sageattention instalado ({v}) difere do esperado ({os.environ['SAGE_VERSION_EXPECTED']}).")
PY
}

install_from_hf_manifest() {
  [[ -n "$MANIFEST_PATH" ]] || return 1
  manifest_is_safe_for_5090 || return 1

  local wheel_url

  if [[ -n "$MANIFEST_HF_WHEEL_URL" ]]; then
    wheel_url="$MANIFEST_HF_WHEEL_URL"
  elif [[ -n "$MANIFEST_WHEEL_FILE" ]]; then
    wheel_url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" "${REMOTE_DIR}/${MANIFEST_WHEEL_FILE}")"
  else
    return 1
  fi

  install_wheel_url "$wheel_url"
}

wheel_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

manifest_latest_path() {
  printf '%s/%s/latest.json' "$WHEELHOUSE_DIR" "$REMOTE_DIR"
}

build_manifest() {
  local wheel_path="$1"
  local wheel_file sha out hf_url

  wheel_file="$(basename "$wheel_path")"
  sha="$(wheel_sha256 "$wheel_path")"
  out="$(manifest_latest_path)"
  mkdir -p "$(dirname "$out")"

  hf_url="$(build_hf_resolve_url "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_REPO_BRANCH" "${REMOTE_DIR}/${wheel_file}")"

  WHEEL_FILE="$wheel_file" \
  WHEEL_SHA256="$sha" \
  HF_WHEEL_URL="$hf_url" \
  TORCH_INDEX_URL_USED="$TORCH_INDEX_URL_USED" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  CUDAARCHS="$CUDAARCHS" \
  OUT_MANIFEST="$out" \
  GPU_NAME="$GPU_NAME" \
  GPU_CC="$GPU_CC" \
  SAGE_VERSION="$SAGE_VERSION" \
  "$PYTHON_BIN" - <<'PY'
import datetime
import importlib.metadata as md
import json
import os
import platform
import sys


def pkg(name):
    try:
        return md.version(name)
    except Exception:
        return None

manifest = {
    "created_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "wheel_file": os.environ["WHEEL_FILE"],
    "wheel_sha256": os.environ["WHEEL_SHA256"],
    "hf_wheel_url": os.environ["HF_WHEEL_URL"],
    "sageattention_version": os.environ["SAGE_VERSION"],
    "torch_version": pkg("torch"),
    "torchvision_version": pkg("torchvision"),
    "torchaudio_version": pkg("torchaudio"),
    "triton_version": pkg("triton"),
    "torch_index_url": os.environ.get("TORCH_INDEX_URL_USED") or None,
    "target_arch": os.environ["TORCH_CUDA_ARCH_LIST"],
    "cudaarchs": os.environ["CUDAARCHS"],
    "required_min_cuda": "12.8",
    "built_for": "RTX5090-sm120",
    "built_from_repo": "https://github.com/thu-ml/SageAttention",
    "built_from_ref": f"v{os.environ['SAGE_VERSION']}",
    "builder_gpu_name": os.environ.get("GPU_NAME") or None,
    "builder_gpu_compute_capability": os.environ.get("GPU_CC") or None,
    "python_version": platform.python_version(),
    "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
    "platform": platform.platform(),
    "machine": platform.machine(),
}

out = os.environ["OUT_MANIFEST"]
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"[INFO] Manifest gerado: {out}")
PY
}

get_hf_python() {
  local hf_python hf_venv
  hf_python="$PYTHON_BIN"
  hf_venv="$WORK_DIR/.hf_venv"

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

publish_to_hf() {
  local wheel_path latest_json hf_python

  wheel_path="$1"
  [[ -f "$wheel_path" ]] || die "Wheel para upload não encontrada: $wheel_path"

  [[ -n "$HF_TOKEN" ]] || die "HF_TOKEN obrigatório para publicar no HF."
  ensure_hf_repo

  latest_json="$(manifest_latest_path)"
  [[ -f "$latest_json" ]] || die "Manifest não encontrado: $latest_json"

  hf_python="$(get_hf_python)"

  WHEEL_PATH="$wheel_path" \
  LATEST_JSON_PATH="$latest_json" \
  HF_REPO_ID="$HF_REPO_ID" \
  HF_REPO_TYPE="$HF_REPO_TYPE" \
  HF_TOKEN="$HF_TOKEN" \
  REMOTE_DIR="$REMOTE_DIR" \
  "$hf_python" - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "dataset")
remote_dir = os.environ.get("REMOTE_DIR", "sageattention220")

for local in (os.environ["WHEEL_PATH"], os.environ["LATEST_JSON_PATH"]):
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

build_wheel() {
  require_cmd git
  ensure_cuda_home

  run_pip install -U ninja cmake packaging

  local stamp src_dir build_log wheel_path
  stamp="$(date +%Y%m%d-%H%M%S)"
  src_dir="$WORK_DIR/SageAttention-v${SAGE_VERSION}-${stamp}"
  build_log="$LOG_DIR/build-sageattention-v${SAGE_VERSION}-${stamp}.log"

  log "Clonando SageAttention v${SAGE_VERSION}"
  git clone --depth 1 --branch "v${SAGE_VERSION}" https://github.com/thu-ml/SageAttention.git "$src_dir"

  export TORCH_CUDA_ARCH_LIST
  export CUDAARCHS
  export EXT_PARALLEL
  export MAX_JOBS
  export NVCC_APPEND_FLAGS

  log "Buildando wheel para sm_120"
  run_pip wheel "$src_dir" --no-build-isolation --wheel-dir "$WHEELHOUSE_DIR" 2>&1 | tee "$build_log"

  if grep -Eq 'sm_120|compute_120|12\\.0\\+PTX|Target compute capabilities:.*12\\.0|arch=compute_120|code=sm_120' "$build_log"; then
    log "Build log indica target Blackwell (12.0/sm_120)."
  else
    warn "Build log não mostrou explicitamente sm_120; continuando com validação pós-instalação."
  fi

  wheel_path="$(get_latest_local_wheel || true)"
  [[ -n "$wheel_path" ]] || die "Nenhuma wheel encontrada após build."

  install_wheel_file "$wheel_path"
  build_manifest "$wheel_path"

  echo "$wheel_path"
}

load_manifest_if_available() {
  MANIFEST_PATH=""
  if fetch_hf_manifest; then
    parse_manifest_stack
  else
    log "Manifest no HF não encontrado ainda (normal na primeira execução)."
  fi
}

validate_runtime() {
  SAGE_VERSION_EXPECTED="$SAGE_VERSION" "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import os
import torch

sv = md.version("sageattention")
if sv != os.environ["SAGE_VERSION_EXPECTED"]:
    raise SystemExit(f"[ERROR] sageattention={sv}, esperado={os.environ['SAGE_VERSION_EXPECTED']}")

print(f"[INFO] Validação final: sageattention={sv}")
print(f"[INFO] Validação final: torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"[INFO] Validação final: capability={torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None}")
PY
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
  log "Script version: ${SCRIPT_VERSION}"
  ensure_pip_ready
  detect_gpu

  case "$ACTION" in
    init-hf)
      ensure_hf_repo
      log "Concluído."
      exit 0
      ;;

    install)
      load_manifest_if_available
      install_torch_stack

      if [[ -n "$WHEEL_URL" ]]; then
        install_wheel_url "$WHEEL_URL" || die "Falha ao instalar WHEEL_URL informado."
      else
        install_from_hf_manifest || die "Wheel/manifest de 5090 não encontrado ou incompatível no HF."
      fi

      validate_runtime
      ;;

    build)
      load_manifest_if_available
      install_torch_stack
      local built_wheel
      built_wheel="$(build_wheel)"
      publish_to_hf "$built_wheel"
      validate_runtime
      ;;

    publish)
      local wheel_to_publish
      wheel_to_publish="$(get_latest_local_wheel || true)"
      [[ -n "$wheel_to_publish" ]] || die "Nenhuma wheel local encontrada para publish."
      build_manifest "$wheel_to_publish"
      publish_to_hf "$wheel_to_publish"
      ;;

    auto)
      load_manifest_if_available
      install_torch_stack

      if [[ -n "$WHEEL_URL" ]]; then
        if install_wheel_url "$WHEEL_URL"; then
          log "Wheel explícita instalada com sucesso."
          validate_runtime
          log "Concluído."
          exit 0
        fi
        warn "WHEEL_URL falhou; seguindo para fluxo HF/build."
      fi

      if install_from_hf_manifest; then
        log "Wheel do seu HF instalada com sucesso (sem rebuild)."
      else
        log "Wheel válida não encontrada no HF. Iniciando build do zero."
        local built_wheel
        built_wheel="$(build_wheel)"

        if [[ -n "$HF_TOKEN" ]]; then
          publish_to_hf "$built_wheel"
        else
          warn "HF_TOKEN ausente: wheel foi compilada/instalada, mas não publicada no HF."
        fi
      fi

      validate_runtime
      ;;
  esac

  log "Concluído."
}

main "$@"
