# SageAttention 2.2.0 Ultimate Installer (RTX 5090)

Script: `install_sageattention220_wheel.sh`

Fluxo implementado:
1. Instala/reinstala PyTorch + Triton primeiro.
2. Tenta instalar wheel por URL direta do Hugging Face (estilo `comfywheel`).
3. Se falhar, tenta wheel indicada por `latest.json` no HF.
4. Se ainda falhar, compila local (`sm_120`), instala e faz upload da wheel + manifest no HF.

## URL direta (prioridade 1)

O script tenta primeiro o formato abaixo (dinâmico por versão do Python):

`https://huggingface.co/adbrasi/comfywheel/resolve/main/sageattention-2.2.0-cpXY-cpXY-linux_x86_64.whl`

Exemplo que você citou:

`https://huggingface.co/adbrasi/comfywheel/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl`

## Repositório HF

- Repo usado por padrão: `adbrasi/comfywheel`
- Tipo padrão: `model`
- Manifest em: `sageattention220/latest.json` dentro do repo

## Versões padrão do instalador

- `SAGE_VERSION=2.2.0`
- `TORCH_CHANNEL=nightly`
- `CUDA_INDEX_VARIANT=cu128`
- `TRITON_SPEC=triton>=3.3`
- `TORCH_CUDA_ARCH_LIST=12.0`
- `CUDAARCHS=120`

## Comando único

```bash
curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- auto
```

## Criar/validar repo HF (se necessário)

```bash
export HF_TOKEN="<seu_token_hf>"
curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- init-hf
```

## Override útil

Se quiser forçar URL exata da wheel:

```bash
export WHEEL_URL="https://huggingface.co/adbrasi/comfywheel/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
```

Se quiser trocar repo HF:

```bash
export HF_DIRECT_REPO_ID="user/repo"
export HF_REPO_ID="user/repo"
```
