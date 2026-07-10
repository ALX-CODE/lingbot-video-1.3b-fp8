# Changelog

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
