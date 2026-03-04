# TODO

## Completed

- [x] Remove unsafe default wheel source (`adbrasi/comfywheel`) from installer
- [x] Set default HF artifact repo to `adbrasi/sageattention220-wheels` (dataset)
- [x] Enforce safe flow: own HF wheel only, fallback build-from-source
- [x] Add strict manifest safety checks for compatibility
- [x] Keep first-run behavior to build and publish wheel for future fast installs
- [x] Validate script (`bash -n`, help, `init-hf`) and push to GitHub
- [x] Multi-GPU refactor: auto-detect GPU, dynamic defaults, registry system
- [x] Smart torch stack: skip reinstall if current stack already works
- [x] Registry.json with SM subdirectories on HF
- [x] CUDA auto-detection and index variant resolution
- [x] Backward compat with legacy latest.json
- [x] Registry merge for concurrent build safety
- [x] Update README, CLAUDE.md, activities.md

## Pending

- [ ] Test on actual RunPod/Vast.ai machines with different GPUs
- [ ] First registry.json publish to HF (will happen on first `auto` run)
- [ ] Verify backward compat: old sm_120 wheels still installable via latest.json
