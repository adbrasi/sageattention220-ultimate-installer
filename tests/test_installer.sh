#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_TMP="$(mktemp -d)"
trap 'rm -rf "$TEST_TMP"' EXIT

export SAGE_INSTALLER_TEST_MODE=1
export WORK_DIR="$TEST_TMP/work"
export WHEELHOUSE_DIR="$TEST_TMP/wheelhouse"
export LOG_DIR="$TEST_TMP/logs"

# shellcheck source=../install_sageattention220_wheel.sh
source "$SCRIPT_DIR/install_sageattention220_wheel.sh"

GPU_SM="sm_120"
GPU_SM_NUM="120"
SAGE_EXPECT_VERSION="2.2.0"
SAGE_SOURCE_REF="v2.2.0"

current_package_version() {
  case "$1" in
    torch) printf '%s\n' '2.13.0+cu130' ;;
    triton) printf '%s\n' '3.7.1' ;;
    *) return 1 ;;
  esac
}

current_torch_cuda_version() {
  printf '%s\n' '13.0'
}

assert_eq() {
  local expected="$1" actual="$2" description="$3"
  if [[ "$actual" != "$expected" ]]; then
    printf '[FAIL] %s: expected=%s actual=%s\n' "$description" "$expected" "$actual" >&2
    return 1
  fi
  printf '[PASS] %s\n' "$description"
}

test_runtime_artifact_key() {
  assert_eq \
    'torch-2.13.0-cu130_cuda-13.0_triton-3.7.1' \
    "$(runtime_artifact_key)" \
    'artifact key includes torch and triton ABI'
}

test_legacy_manifest_rejects_wrong_torch() {
  REG_WHEEL_FILE='sageattention.whl'
  REG_HF_WHEEL_URL='https://example.invalid/sageattention.whl'
  REG_TORCH_VERSION='2.11.0+cu130'
  REG_TORCH_CUDA_VERSION=''
  REG_TRITON_VERSION='3.6.0'
  REG_SM='12.0'
  REG_SAGE_VERSION='2.2.0'
  REG_BUILT_FROM_REPO='https://github.com/thu-ml/SageAttention.git'
  REG_BUILT_FROM_REF='v2.2.0'

  if validate_registry_entry; then
    printf '[FAIL] mismatched torch ABI was accepted\n' >&2
    return 1
  fi
  printf '[PASS] mismatched torch ABI triggers rebuild\n'
}

test_registry_selects_exact_stack() {
  local python_tag
  python_tag="$("$PYTHON_BIN" -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")"
  REGISTRY_PATH="$TEST_TMP/registry.json"
  REGISTRY_IS_LEGACY='false'

  REGISTRY_PATH="$REGISTRY_PATH" PYTHON_TAG="$python_tag" "$PYTHON_BIN" - <<'PY'
import json
import os


def entry(torch_version, triton_version, marker):
    return {
        "key": marker,
        "sm": "sm_120",
        "python_tag": os.environ["PYTHON_TAG"],
        "wheel_file": f"{marker}.whl",
        "hf_wheel_url": f"https://example.invalid/{marker}.whl",
        "sageattention_version": "2.2.0",
        "torch_version": torch_version,
        "torch_cuda_version": "13.0",
        "triton_version": triton_version,
        "built_from_repo": "https://github.com/thu-ml/SageAttention.git",
        "built_from_ref": "v2.2.0",
        "artifact_key": marker,
    }


data = {
    "schema_version": 2,
    "wheels": [
        entry("2.11.0+cu130", "3.6.0", "old-stack"),
        entry("2.13.0+cu130", "3.7.1", "current-stack"),
    ],
}
with open(os.environ["REGISTRY_PATH"], "w", encoding="utf-8") as handle:
    json.dump(data, handle)
PY

  parse_registry_entry
  assert_eq 'current-stack.whl' "$REG_WHEEL_FILE" 'registry selects exact stack wheel'
  validate_registry_entry
  printf '[PASS] exact stack wheel passes validation\n'
}

test_import_validation_rejects_broken_extension() {
  local package_root="$TEST_TMP/pythonpath"
  mkdir -p "$package_root/sageattention" "$package_root/sageattention-2.2.0.dist-info"

  PACKAGE_ROOT="$package_root" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["PACKAGE_ROOT"])
(root / "sageattention" / "__init__.py").write_text(
    "raise ImportError('undefined symbol: simulated_torch_abi')\n",
    encoding="utf-8",
)
(root / "sageattention-2.2.0.dist-info" / "METADATA").write_text(
    "Metadata-Version: 2.1\nName: sageattention\nVersion: 2.2.0\n",
    encoding="utf-8",
)
PY

  if PYTHONPATH="$package_root" validate_sage_version >/dev/null 2>&1; then
    printf '[FAIL] broken sageattention import was accepted\n' >&2
    return 1
  fi
  printf '[PASS] broken sageattention import is rejected\n'
}

test_auto_builds_and_publishes_when_cache_misses() {
  local events=''
  ACTION='auto'
  HF_TOKEN='write-token'
  LAST_BUILT_WHEEL=''

  record() {
    events="${events}${events:+ }$1"
  }
  ensure_python() { record ensure_python; }
  ensure_pip_ready() { record ensure_pip; }
  detect_gpu() {
    GPU_NAME='test-gpu'
    GPU_ARCH_NAME='Blackwell'
    GPU_SM='sm_120'
    GPU_CC='12.0'
    TORCH_CHANNEL='nightly'
    CUDA_INDEX_VARIANT='cu128'
    TORCH_CUDA_ARCH_LIST='12.0'
    CUDAARCHS='120'
    TRITON_SPEC='triton>=3.3,<4.0'
    record detect_gpu
  }
  detect_system_cuda() { SYSTEM_CUDA_VERSION='13.0'; record detect_cuda; }
  install_torch_stack() { record install_torch; }
  load_registry_if_available() { record load_registry; }
  install_from_registry() { record cache_miss; return 1; }
  build_wheel() { LAST_BUILT_WHEEL='/tmp/sageattention.whl'; record build; }
  publish_to_hf() { record publish; }
  validate_runtime() { record validate; }

  main
  assert_eq \
    'ensure_python ensure_pip detect_gpu detect_cuda install_torch load_registry cache_miss build publish validate' \
    "$events" \
    'auto flow builds and publishes an exact-stack cache miss'
}

test_runtime_artifact_key
test_legacy_manifest_rejects_wrong_torch
test_registry_selects_exact_stack
test_import_validation_rejects_broken_extension
test_auto_builds_and_publishes_when_cache_misses
