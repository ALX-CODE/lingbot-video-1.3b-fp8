# Attribution and scope

This project vendors and adapts files from
[Robbyant/LingBot-Video](https://github.com/Robbyant/lingbot-video), originally
released under the Apache License 2.0. The vendored baseline was commit
`a2bb04b78edd848500dc27a26e035a95442ae186`.

The FP8 checkpoint is a quantized derivative of
`robbyant/lingbot-video-dense-1.3b`. It is not an official Robbyant release.
LingBot and related names belong to their respective owners.

The ComfyUI integration, selective FP8 runtime path, workflow controls,
progress reporting, prompt helpers, TI2V conditioning, and experimental
first/last-frame bridge were developed independently for this repository.

## Modification manifest

The following files are modified from, or interoperate closely with, the
Robbyant baseline identified above. Their behavior is not an official
Robbyant implementation:

| File | Modification notice |
| --- | --- |
| `lingbot_video/__init__.py` | Adapted package exports for the independently packaged runtime. |
| `lingbot_video/pipeline_lingbot_video.py` | Adapted for ComfyUI execution, explicit attention backends, CFG modes, progress reporting, and local memory lifecycle. |
| `lingbot_video/transformer_lingbot_video.py` | Adapted for selective dense FP8 execution, explicit attention dispatch, batched CFG masking, and optimized timestep modulation. |
| `lingbot_video/prompt_rewriter.py` | Added for local plaintext-to-structured-JSON generation and validation against LingBot prompt structures. |
| `lingbot_video/default_negative_prompt.json` | Packaged and adapted as the default T2V negative-prompt preset. |
| `lingbot_video/default_negative_prompt_image.json` | Packaged and adapted as the default image-conditioned negative-prompt preset. |

Files not listed in this manifest may be original integration code, unmodified
upstream material, or ordinary package metadata. Git history is the
authoritative record of changes in this repository.

## Model asset

`diffusion_pytorch_model.safetensors` in the `v1.0.2` GitHub Release is a
quantized derivative of the official Dense 1.3B transformer. Recipients must
retain the Apache-2.0 license and this attribution notice when redistributing
that derivative. The release asset is intended for inference, and exact visual
equivalence to the official BF16 weights is not claimed.
