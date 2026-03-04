# Activities Log

## 2026-03-04 — Multi-GPU Refactor

Major refactoring of `install_sageattention220_wheel.sh`:

### What changed
- **Multi-GPU support**: Script now auto-detects any GPU via nvidia-smi compute_cap (A100, L40S, RTX 4090, H100, B200, RTX 5090, RTX 6000 Ada/Pro Blackwell, etc.)
- **Smart torch stack**: `check_torch_stack_ok()` verifies if current torch/triton/CUDA already works before attempting reinstall — avoids breaking working setups
- **Registry system**: Replaced single `latest.json` with `registry.json` that maps `(sm_XX, python_tag)` to wheel metadata. Multiple GPU architectures coexist in the same HF repo
- **SM subdirectories**: Wheels stored in `sageattention220/sm_120/`, `sm_89/`, etc. to avoid filename collisions
- **CUDA auto-detection**: `detect_system_cuda()` reads nvcc version, validates against GPU minimum, auto-resolves CUDA index variant (cu118, cu121, cu124, cu126, cu128)
- **Dynamic defaults**: All hardcoded values (sm_120, cu128, nightly, triton>=3.3) replaced with auto-derived values from `GPU_ARCH_MAP` lookup table
- **Backward compat**: Falls back to `latest.json` if `registry.json` not yet published; existing flat wheels for sm_120 continue to work
- **Registry merge**: `merge_remote_registry()` downloads current remote registry before publish, merges entries to prevent overwriting by concurrent builds

### Removed
- `MIN_CUDA_FOR_SM120`, `MIN_TRITON_FOR_50XX` hardcoded vars
- `manifest_is_safe_for_5090()` (replaced by generic `validate_registry_entry()`)
- All RTX 5090-specific checks and warnings

### Files modified
- `install_sageattention220_wheel.sh` — full rewrite
- `README.md` — multi-GPU docs, new env vars, registry system
- `CLAUDE.md` — updated architecture docs
- `activities.md` — this log
- `todo-list.md` — updated checklist

---

## Previous Work (pre-refactor)

- Built original installer targeting RTX 5090 (sm_120) only
- Implemented safe wheel flow: own HF repo, manifest validation, build-from-source fallback
- Fixed wheel publish bug (LAST_BUILT_WHEEL path capture)
- Added triton version preservation (don't force-replace if already >=3.3)
- Added build provenance metadata (built_from_repo, built_from_ref, built_from_commit)
