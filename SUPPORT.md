# Support

Use GitHub Issues for reproducible defects and Discussions, when enabled, for
general setup or generation-quality questions.

## Supported scope

- LingBot-Video Dense 1.3B with the published selective FP8 transformer
- T2V and TI2V workflows
- The explicitly experimental first/last-frame workflow
- The tested Windows, NVIDIA CUDA, and ComfyUI integration path

Internal MoE/INT8 compatibility code is not a promise that the 30B MoE model is
supported by this Dense 1.3B release. FlashVSR, RIFE, VideoHelperSuite, ComfyUI,
and official Robbyant models are maintained by their respective projects;
issues isolated to those components should be reported upstream.

## Before opening an issue

Start with a short, low-resolution smoke test and verify the release asset
hashes. Search existing issues, then include:

- repository version or commit;
- GPU and VRAM;
- operating system and driver;
- Python, PyTorch, CUDA, ComfyUI, Diffusers, Transformers, and SageAttention
  versions;
- workflow name and non-default settings;
- the complete traceback as text; and
- whether the official BF16 transformer reproduces the problem.

Sanitize usernames, absolute paths, prompts, images, videos, tokens, and other
private data. Performance reports must include dimensions, frames, steps, CFG,
seed, warm-up policy, and whether prompt encoding or post-processing was timed.

This is an independent community project and does not provide an SLA or
official Robbyant support.
