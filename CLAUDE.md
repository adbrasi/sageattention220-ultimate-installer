# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-script universal installer for **SageAttention 2.2.0** supporting any NVIDIA GPU. Auto-detects GPU compute capability, manages torch/triton stack, and caches built wheels on HuggingFace per GPU architecture.

The sole deliverable is `install_sageattention220_wheel.sh`.

## Running the Script

```bash
# Validate syntax
bash -n install_sageattention220_wheel.sh

# Show help
./install_sageattention220_wheel.sh --help

# Full auto flow
export HF_TOKEN="<token>"
bash install_sageattention220_wheel.sh auto

# Other actions
bash install_sageattention220_wheel.sh install   # install from HF only
bash install_sageattention220_wheel.sh build      # force local build
bash install_sageattention220_wheel.sh publish    # publish local wheel
bash install_sageattention220_wheel.sh init-hf    # create/validate HF repo
```

## Architecture

### GPU Auto-Detection Flow
1. `detect_gpu()` — reads CC from nvidia-smi, derives SM, looks up `GPU_ARCH_MAP` for min CUDA/triton/torch channel
2. `detect_system_cuda()` — parses nvcc version, validates against GPU minimum, resolves `CUDA_INDEX_VARIANT`
3. All build/install variables are derived dynamically (no hardcoded GPU assumptions)

### Smart Torch Stack
`install_torch_stack()` calls `check_torch_stack_ok()` first — if current torch already satisfies the GPU requirements (CUDA version, triton version, arch_list), it skips installation entirely. This avoids unnecessary reinstalls when the user already has a working stack.

### Registry System (HF)
- `registry.json` on HF maps `(sm_XX, python_tag)` → wheel metadata
- Backward compatible: falls back to `latest.json` if registry doesn't exist
- Wheels stored in SM subdirectories: `sageattention220/sm_120/`, `sageattention220/sm_89/`, etc.
- `merge_remote_registry()` handles concurrent builds from different GPU types

### `auto` Action Flow
```
detect_gpu → detect_system_cuda → load_registry_if_available
→ install_torch_stack (skips if already OK)
→ install_from_registry (if matching wheel exists)
→ OR build_wheel + publish_to_hf (if no wheel found)
→ validate_runtime
```

## GPU Architecture Map

Defined as `GPU_ARCH_MAP` associative array (bash 4+):

| CC | Architecture | Min CUDA | Min Triton | Default Channel |
|---|---|---|---|---|
| 8.0 | Ampere | 11.1 | 2.0 | stable |
| 8.9 | Ada Lovelace | 11.8 | 2.1 | stable |
| 9.0 | Hopper | 12.0 | 3.0 | stable |
| 10.0 | Blackwell DC | 12.8 | 3.3 | nightly |
| 12.0 | Blackwell | 12.8 | 3.3 | nightly |

## Key Design Constraints

- **Never use third-party wheels** — only self-built wheels from own HF repo
- **Never reinstall torch unnecessarily** — check first, install only if needed
- Auto-detect everything possible; user env vars override auto-detection
- Registry merge before publish prevents race conditions between different GPU builds
- Primary references: `research.md` and ComfyUI-Easy-Install SageAttention scripts
