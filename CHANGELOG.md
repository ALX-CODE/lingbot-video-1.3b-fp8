# Changelog

## 1.0.2 — 2026-07-10

- Normalized Qwen-generated `world_knowledge` values into LingBot's required
  array shape when the model returns a string, object, null, or omits the field.
- Preserved strict validation for substantive schema errors while avoiding an
  unnecessary and sometimes unsuccessful second Qwen repair pass.
- Re-published the unchanged FP8 checkpoint under the matching patch tag so
  code and model downloads remain version-aligned.

## 1.0.1 — 2026-07-10

- Corrected all published workflows to safe 640×352, 20 FPS, 3-second,
  28-step defaults with deterministic seed 42.
- Disabled FlashVSR and RIFE by default and repaired current lazy-switch wiring.
- Added early FP8 capability checks and an actionable BF16 fallback message.
- Declared the mandatory `accelerate` dependency and documented SageAttention,
  Windows Portable, external node packs, models, storage, and troubleshooting.
- Made release downloads version-stable and resilient to corrupt or complete
  partial files.
- Added exact public-workflow regression tests, standalone-safe test discovery,
  CI, Registry-ready project metadata, contribution/security/support policies,
  and expanded Apache attribution.
- Added a compressed example output and narrowed performance claims to the one
  RTX 5080 configuration that was measured.

## 1.0.0 — 2026-07-10

- Initial public Dense 1.3B selective FP8 checkpoint, ComfyUI node pack, and
  T2V/TI2V/experimental FLF workflows.
