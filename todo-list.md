# TODO

- [x] Remove unsafe default wheel source (`adbrasi/comfywheel`) from installer
- [x] Set default HF artifact repo to `adbrasi/sageattention220-wheels` (dataset)
- [x] Enforce safe flow: own HF wheel only, fallback build-from-source for SageAttention v2.2.0
- [x] Add strict manifest safety checks for 5090/sm_120 compatibility
- [x] Keep first-run behavior to build and publish wheel for future fast installs
- [x] Update README to match corrected behavior
- [x] Validate script (`bash -n`, help, `init-hf`) and push to GitHub
- [x] Update `activities.md` with corrected state

- [x] Reconcile script with `research.md` + `ComfyUI-Easy-Install` SageAttention flow
- [x] Fix wheel publish bug (path capture from build output)
- [x] Restore consistent SageAttention version/ref handling (v2.2.0 default)
- [x] Avoid unnecessary triton overwrite when torch stack already ships compatible triton
- [x] Revalidate script and publish updated installer to GitHub
- [x] Update `README.md` and `activities.md` with final behavior
