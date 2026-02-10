# SageAttention 2.2.0 Ultimate Installer (RTX 5090)

This repo contains one script: `install_sageattention220_wheel.sh`.

It is designed for repeated installs on similar machines (same Docker base, CUDA stack, RTX 5090) with this workflow:
1. Always install/validate PyTorch + Triton first.
2. Try prebuilt `sageattention` wheel (local or Hugging Face).
3. If wheel is missing/fails, build locally for `sm_120`, install it, and upload wheel + manifest to Hugging Face.

## Exact Version Policy

The exact runtime stack is locked by `latest.json` in Hugging Face:
- `torch_version`
- `torchvision_version`
- `torchaudio_version`
- `triton_version`
- `torch_index_url`
- wheel filename + checksum

On new machines, the script reads that manifest first and installs those exact versions before installing the wheel.

## Required Base

- NVIDIA GPU: RTX 5090 (compute capability 12.0 / `sm_120`)
- CUDA Toolkit with `nvcc` available (CUDA 12.8+)
- Python 3.11 or 3.12
- Linux environment

## First Machine (build + publish)

```bash
export HF_REPO_ID="<your_hf_user_or_org>/<your_repo>"
export HF_TOKEN="<your_hf_token>"

curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- auto
```

This first run will:
- install torch/triton,
- try existing wheel from HF,
- build wheel if needed,
- upload wheel + `latest.json` to HF.

## Next Machines (instant install from wheel)

```bash
export HF_REPO_ID="<your_hf_user_or_org>/<your_repo>"
# export HF_TOKEN="<your_hf_token>"   # needed only for private HF repo

curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- auto
```

If wheel + manifest already exist, install is wheel-first (no rebuild).

## Optional: Explicit Pin from Environment

You can force exact versions manually (overrides manifest):

```bash
export TORCH_VERSION="<exact_torch_version>"
export TORCHVISION_VERSION="<exact_torchvision_version>"
export TORCHAUDIO_VERSION="<exact_torchaudio_version>"
export TRITON_VERSION="<exact_triton_version>"
```

## Notes

- Default build target is `TORCH_CUDA_ARCH_LIST=12.0` and `CUDAARCHS=120`.
- If you want PTX fallback, set `TORCH_CUDA_ARCH_LIST=12.0+PTX`.
