# LingBot-Video Dense 1.3B FP8 for ComfyUI

A portable ComfyUI custom-node pack and selective fused FP8 checkpoint for the
official [LingBot-Video Dense 1.3B](https://huggingface.co/robbyant/lingbot-video-dense-1.3b)
model. It includes ready-to-load text-to-video (T2V), first-frame
image-to-video (TI2V), and experimental first/last-frame workflows.

> This is an independent community project, not an official Robbyant release.

## What the FP8 path does

The checkpoint uses native CUDA E4M3 W8A8 scaled matrix multiplication for the
projections that measured faster on the test machine:

- fused attention Q/K/V
- fused MLP gate/up

Attention output, MLP down projection, conditioning, normalization, input, and
output paths remain BF16 to preserve quality where FP8 did not earn a measured
speed advantage. The original BF16 transformer is never modified.

In a matched cached 256×144×5-frame, one-step sampler smoke test on an RTX
5080, FP8 took **3.651 s** versus **4.651 s** for BF16: **1.27× faster**, or
about **21.5% less sampler time**. Fused QKV and gate/up microbenchmarks measured
1.29× and 1.62× respectively. See [BENCHMARKS.md](BENCHMARKS.md) for the exact
scope and limitations.

## Tested system

- NVIDIA RTX 5080, 16 GB VRAM
- 64 GB system RAM
- Windows / WDDM
- Python 3.11.9
- PyTorch 2.12.1+cu130, CUDA 13.0
- Diffusers 0.38.0, Transformers 5.3.0
- SageAttention 2.2.0
- Current ComfyUI source checkout (commit `b910f4fa` during validation)

The FP8 runtime requires a CUDA GPU/runtime where PyTorch provides
`torch._scaled_mm` for E4M3 tensors. Other GPUs and Linux may work but have not
been validated here. Start with a short, low-resolution smoke test.

## Installation

### 1. Install the custom node

From the ComfyUI directory:

```powershell
git clone https://github.com/ALX-CODE/lingbot-video-1.3b-fp8 `
  custom_nodes/ComfyUI-LingBotVideo
.\venv\Scripts\python.exe -m pip install -r `
  custom_nodes\ComfyUI-LingBotVideo\requirements.txt
```

Do not replace your working PyTorch installation with a generic build. The
tested FP8 path depends on a CUDA-enabled PyTorch build appropriate for the GPU.

### 2. Download the official base model components

Install the Hugging Face CLI into the ComfyUI environment if needed, then
download the official model. Keeping the BF16 `transformer` folder gives you a
reference fallback; `--exclude "transformer/*"` saves disk space if you only
want FP8.

```powershell
.\venv\Scripts\python.exe -m pip install -U "huggingface_hub[cli]"
.\venv\Scripts\hf.exe download robbyant/lingbot-video-dense-1.3b `
  --local-dir models\lingbot_video\lingbot-video-dense-1.3b
```

The final model directory must contain `model_index.json`, `processor`,
`scheduler`, `text_encoder`, `vae`, and `transformer_fp8_dense`.

### 3. Download the FP8 transformer

The downloader resumes partial transfers and verifies both files with SHA-256:

```powershell
.\venv\Scripts\python.exe `
  custom_nodes\ComfyUI-LingBotVideo\tools\download_fp8.py `
  --model-dir models\lingbot_video\lingbot-video-dense-1.3b
```

You can also download `config.json` and
`diffusion_pytorch_model.safetensors` manually from the latest GitHub Release
and place them in:

```text
ComfyUI/models/lingbot_video/lingbot-video-dense-1.3b/transformer_fp8_dense/
```

Checksums are in [SHA256SUMS.txt](SHA256SUMS.txt).

### 4. Install workflow node packs

All three UI workflows use:

- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for MP4 output
- [ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) for optional 2× spatial upscaling
- [ComfyUI-VFI](https://github.com/GACLove/ComfyUI-VFI) for optional RIFE interpolation

FlashVSR and RIFE default to bypassed and execute lazily. When enabled, the
workflow order is **FlashVSR first, RIFE second**.

### 5. Launch ComfyUI and load a workflow

Use your normal launcher with SageAttention enabled. On the tested install:

```powershell
.\venv\Scripts\python.exe main.py --use-sage-attention --disable-xformers
```

Drag one of these files into ComfyUI:

- [`workflows/lingbot_fp8_t2v.json`](workflows/lingbot_fp8_t2v.json) — plaintext T2V
- [`workflows/lingbot_fp8_ti2v.json`](workflows/lingbot_fp8_ti2v.json) — first-frame + plaintext TI2V
- [`workflows/lingbot_fp8_flf_experimental.json`](workflows/lingbot_fp8_flf_experimental.json) — experimental first/last-frame bridge

The loader should show model `lingbot-video-dense-1.3b` and transformer
`transformer_fp8_dense`. If the model dropdown does not find an external path,
set `LINGBOT_MODEL_ROOTS` to one or more model-root directories separated by
the platform path separator, then restart ComfyUI.

## Workflow controls

- Enter **FPS** and **total seconds**; frame count is calculated and snapped to
  LingBot's required `4n+1` form.
- The prompt node turns plaintext into validated LingBot structured JSON using
  the base model's local Qwen text encoder. A structured-JSON override remains
  available.
- The sampler displays stage, denoise step, percentage, elapsed time, ETA, CFG
  mode, and any fallback reason.
- `sequential_sage` is the validated default. Batched CFG modes remain opt-in
  because they were slower on the tested 16 GB system.
- The TI2V sampler fixes the first latent frame throughout denoising.
- The FLF sampler pins both endpoint latents. LingBot was not trained for this
  mode, so large endpoint differences can morph, freeze, or jump.

Recommended reference settings are 40 steps, CFG 3.0, shift 3.0, and sequential
Sage CFG. For experiments, keep the seed fixed and change one setting at a time.

## Build your own FP8 checkpoint

The release asset is ready to use. To reproduce it from the official BF16
Dense 1.3B transformer:

```powershell
.\venv\Scripts\python.exe `
  custom_nodes\ComfyUI-LingBotVideo\tools\convert_dense_fp8.py `
  --model-dir models\lingbot_video\lingbot-video-dense-1.3b
```

The converter writes `transformer_fp8_dense.incomplete` first, validates all
120 quantized weights and their scales, then atomically renames the result.

## License and attribution

Apache-2.0. This repository vendors and adapts portions of the official
[Robbyant/LingBot-Video](https://github.com/Robbyant/lingbot-video) project.
See [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE). The FP8 checkpoint is a
quantized derivative of the official Dense 1.3B weights.
