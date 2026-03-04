# SageAttention 2.2.0 Universal Installer

Script: `install_sageattention220_wheel.sh`

## O que faz

Instala SageAttention 2.2.0 em **qualquer GPU NVIDIA** automaticamente. Detecta a GPU, encontra ou compila a wheel correta, e cacheia no HuggingFace para reutilizar em futuras máquinas.

**Ideal para quem usa RunPod / Vast.ai** e troca de GPU/máquina frequentemente.

## GPUs suportadas (auto-detecção)

| GPU | CC | SM | Arquitetura | Min CUDA |
|---|---|---|---|---|
| A100 | 8.0 | sm_80 | Ampere | 11.1 |
| RTX 4090 | 8.9 | sm_89 | Ada Lovelace | 11.8 |
| L40S | 8.9 | sm_89 | Ada Lovelace | 11.8 |
| RTX 6000 Ada | 8.9 | sm_89 | Ada Lovelace | 11.8 |
| H100 | 9.0 | sm_90 | Hopper | 12.0 |
| B200 | 10.0 | sm_100 | Blackwell DC | 12.8 |
| RTX 5090 | 12.0 | sm_120 | Blackwell | 12.8 |
| RTX 6000 Pro Blackwell | 12.0 | sm_120 | Blackwell | 12.8 |

## Fluxo do `auto`

1. Detecta GPU (compute capability) e CUDA do sistema
2. **Verifica se torch stack atual já funciona** — se sim, não reinstala nada
3. Procura wheel compatível no HF via `registry.json` (por SM + versão do Python)
4. Se encontrou → instala direto do HF (rápido, sem compilação)
5. Se não encontrou → compila do zero, instala, e publica no HF para próximas máquinas

## Uso rápido

### Primeira máquina (gera e publica wheel)

```bash
export HF_TOKEN="<seu_token>"
curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- auto
```

### Próximas máquinas (instalação rápida do HF)

```bash
curl -fsSL https://raw.githubusercontent.com/adbrasi/sageattention220-ultimate-installer/main/install_sageattention220_wheel.sh | bash -s -- auto
```

Se já existe wheel para o SM da sua GPU no HF, não recompila.

## Repositório HF

- `adbrasi/sageattention220-wheels` (tipo: `dataset`)
- Estrutura:
  ```
  sageattention220/
  ├── registry.json         ← índice de todas as wheels
  ├── latest.json           ← última wheel publicada (backward compat)
  ├── sm_120/               ← wheels para Blackwell consumer
  ├── sm_89/                ← wheels para Ada Lovelace
  ├── sm_90/                ← wheels para Hopper
  └── sm_80/                ← wheels para Ampere
  ```

## Variáveis de ambiente

Todas opcionais — o script auto-detecta tudo quando possível.

| Variável | Default | Descrição |
|---|---|---|
| `HF_TOKEN` | *(vazio)* | Necessário para publish |
| `TORCH_CHANNEL` | auto (`stable` ou `nightly` conforme GPU) | Canal do PyTorch |
| `CUDA_INDEX_VARIANT` | auto do CUDA do sistema | Sufixo do index (cu118, cu128...) |
| `TORCH_CUDA_ARCH_LIST` | auto da GPU | Arch de build |
| `CUDAARCHS` | auto da GPU | CMake arch flag |
| `TRITON_SPEC` | auto do mínimo da GPU | Spec do triton |
| `SKIP_TORCH_INSTALL` | `0` | `1` = pula instalação do torch |
| `SAGE_SOURCE_REF` | `v2.2.0` | Tag/branch para build |
| `SAGE_EXPECT_VERSION` | `2.2.0` | Validação pós-install |
| `WHEEL_URL` | *(vazio)* | URL explícita de wheel (bypassa registry) |

## Ações disponíveis

```bash
./install_sageattention220_wheel.sh auto      # fluxo completo
./install_sageattention220_wheel.sh install    # só instala do HF (sem build)
./install_sageattention220_wheel.sh build      # força build local
./install_sageattention220_wheel.sh publish    # publica wheel local no HF
./install_sageattention220_wheel.sh init-hf    # cria/valida repo HF
```

## Build da branch mais nova

```bash
export HF_TOKEN="<seu_token>"
export SAGE_SOURCE_REF="main"
export SAGE_EXPECT_VERSION=""
./install_sageattention220_wheel.sh auto
```

## Torch stack inteligente

O script **não reinstala torch se o stack atual já funciona**. Ele verifica:
- CUDA do torch >= mínimo para a GPU
- Triton >= mínimo para a GPU
- arch_list inclui o SM da GPU

Se tudo OK, pula direto para a instalação do SageAttention.
