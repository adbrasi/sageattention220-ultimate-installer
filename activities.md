# Activities Log

## Recent Work Summary
- Re-read both references required by user:
  - `research.md` (stack guidance for 5090/Blackwell).
  - `Tavris1/ComfyUI-Easy-Install` branch `MAC-Linux` (scripts `SageAttention-NEXT.sh` and `SageAttention2-2.2.0.sh` from `Helper-CEI-NEXT-unix.zip`).
- Fixed critical publish bug:
  - `build_wheel()` no longer returns mixed stdout logs in command substitution.
  - build path is now stored in `LAST_BUILT_WHEEL`, then uploaded reliably.
- Restored consistent SageAttention version/ref handling:
  - `SAGE_VERSION=2.2.0` default.
  - `SAGE_SOURCE_REF=v2.2.0` default.
  - `SAGE_EXPECT_VERSION=2.2.0` default (can be emptied for `main` builds).
- Improved triton behavior for 5090:
  - script now keeps installed triton if already `>=3.3` (avoids unnecessary replacement of torch-nightly bundled triton).
  - still supports explicit pin via `TRITON_VERSION`.
- Manifest/build metadata improvements:
  - timezone-safe timestamp (removed deprecated `utcnow()` usage warning).
  - manifest now records `built_from_repo`, `built_from_ref`, and `built_from_commit`.
- README updated with:
  - reference alignment note,
  - default stable behavior (`v2.2.0`),
  - optional `main` build mode (`SAGE_SOURCE_REF=main`, `SAGE_EXPECT_VERSION=`).

## User's Latest Instructions
- Always use `research.md` and the `ComfyUI-Easy-Install` repo as primary reference.
- Do not use third-party wheel for other cards (e.g. 6000 Ada build).
- Build from zero for RTX 5090 and save wheel for fast reinstall from HF.

## Current Project State
- Updated local files:
  - `install_sageattention220_wheel.sh`
  - `README.md`
  - `todo-list.md`
  - `activities.md`
- Target GitHub repo for sync:
  - `https://github.com/adbrasi/sageattention220-ultimate-installer`
- Target HF artifact repo (default in script):
  - `https://huggingface.co/datasets/adbrasi/sageattention220-wheels`

## Validation Performed
- `bash -n install_sageattention220_wheel.sh` passed.
- `./install_sageattention220_wheel.sh --help` checked.
- Reference extraction from `ComfyUI-Easy-Install` completed for SageAttention install/build flow.
