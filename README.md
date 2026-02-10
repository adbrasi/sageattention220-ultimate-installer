# SageAttention 2.2.0 Installer (RTX 5090 Safe)

Script: `install_sageattention220_wheel.sh`

## Objetivo

- Nunca usar wheel de terceiros por padrão.
- Priorizar wheel construída por você e armazenada no seu HF repo.
- Se não existir wheel válida para 5090, fazer build local do zero (`sm_120`) e salvar para próximas máquinas.
- Fluxo baseado nas referências: `research.md` e scripts SageAttention do `ComfyUI-Easy-Install` (branch `MAC-Linux`).

## Repositório HF padrão

- `adbrasi/sageattention220-wheels`
- tipo: `dataset`
- arquivos publicados em: `sageattention220/`
  - `latest.json`
  - `sageattention-2.2.0-...whl`

## Fluxo do `auto`

1. Instala stack base (PyTorch/Triton) recomendada para 5090.
2. Tenta instalar wheel do seu próprio HF via `latest.json`.
3. Se não houver wheel/manifest compatível, compila do zero (default `SAGE_SOURCE_REF=v2.2.0`, `TORCH_CUDA_ARCH_LIST=12.0`, `CUDAARCHS=120`).
4. Instala a wheel gerada e publica no HF (se `HF_TOKEN` estiver definido).

## Requisitos mínimos

- GPU: RTX 5090 (`sm_120`)
- CUDA toolkit com `nvcc` >= 12.8
- Python 3.11/3.12

## Defaults técnicos

- `TORCH_CHANNEL=nightly`
- `CUDA_INDEX_VARIANT=cu128`
- `TRITON_SPEC=triton>=3.3,<4.0`
- `SAGE_VERSION=2.2.0`
- `SAGE_SOURCE_REF=v2.2.0`

Obs.: o script não sobrescreve `triton` à força se a versão já for compatível (>= 3.3), evitando troca desnecessária do triton que veio com o torch nightly.

## Primeiro setup (máquina que vai gerar wheel)

```bash
export HF_TOKEN="<seu_token_hf>"
curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- auto
```

## Próximas máquinas (instalação rápida)

```bash
curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- auto
```

Se `latest.json` + wheel já existirem no HF, ele não recompila.

## Opcional: build da branch mais nova do SageAttention

```bash
export HF_TOKEN="<seu_token_hf>"
export SAGE_SOURCE_REF="main"
export SAGE_EXPECT_VERSION=""
curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- auto
```

## Criar/validar repo HF sem build

```bash
export HF_TOKEN="<seu_token_hf>"
curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- init-hf
```

## Importante

- O script valida manifest para evitar wheel incompatível com 5090 (`target_arch` precisa apontar para `12.0/sm_120`).
- `WHEEL_URL` só é usada se você definir explicitamente.
