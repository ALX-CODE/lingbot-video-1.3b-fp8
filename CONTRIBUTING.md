# Contributing

Thanks for helping improve the independent LingBot-Video Dense 1.3B FP8
integration. Keep changes focused, reproducible, and safe for an existing
ComfyUI installation.

## Scope

- The supported release target is the Dense 1.3B FP8 checkpoint and its T2V,
  TI2V, and experimental FLF ComfyUI workflows.
- Do not commit model weights, generated media, user prompts, machine-specific
  paths, API keys, logs, or benchmark outputs containing private metadata.
- Preserve the upstream attribution and modification notices in `LICENSE` and
  `NOTICE.md`. New adaptations of upstream files must carry a prominent
  modification notice.
- Avoid broad dependency upgrades or changes to users' PyTorch installations.

## Development setup

Clone this repository as a ComfyUI custom node and use ComfyUI's Python:

```powershell
git clone https://github.com/ALX-CODE/lingbot-video-1.3b-fp8 `
  C:\path\to\ComfyUI\custom_nodes\ComfyUI-LingBotVideo
C:\path\to\ComfyUI\venv\Scripts\python.exe -m pip install -r `
  C:\path\to\ComfyUI\custom_nodes\ComfyUI-LingBotVideo\requirements.txt
```

Run the test suite with the ComfyUI root importable:

```powershell
$env:PYTHONPATH = "C:\path\to\ComfyUI"
C:\path\to\ComfyUI\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Also validate metadata and source syntax:

```powershell
C:\path\to\ComfyUI\venv\Scripts\python.exe -m compileall -q .
C:\path\to\ComfyUI\venv\Scripts\python.exe -c `
  "import json,pathlib,tomllib; tomllib.loads(pathlib.Path('pyproject.toml').read_text()); [json.loads(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.json')]"
```

## Pull requests

Include:

1. The problem and why the change belongs in this Dense 1.3B package.
2. Exact hardware, operating system, Python, PyTorch, CUDA, Diffusers,
   Transformers, SageAttention, and ComfyUI revisions used for validation.
3. Tests run and known gaps.
4. Before/after timings for performance changes, using fixed prompts, seeds,
   dimensions, frame counts, steps, CFG, and warm-up policy.
5. Visual comparisons for any numerical or model-path change. Do not claim
   exact equivalence unless it was measured.

Keep generated media outside Git history; link externally hosted evidence when
needed. By contributing, you agree that your contribution is provided under
the repository's Apache-2.0 license.
