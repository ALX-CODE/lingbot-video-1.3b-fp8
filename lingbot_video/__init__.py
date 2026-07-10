# Modified by ALX-CODE in 2026 for the independent ComfyUI/FP8 integration.
# Based on Robbyant/LingBot-Video commit a2bb04b78edd848500dc27a26e035a95442ae186.
# See ../NOTICE.md for provenance and the modification manifest.
from __future__ import annotations

import importlib
from typing import Any


_EXPORTS = {
    "FlowUniPCMultistepScheduler": (
        ".scheduling_flow_unipc",
        "FlowUniPCMultistepScheduler",
    ),
    "LingBotVideoPipeline": (
        ".pipeline_lingbot_video",
        "LingBotVideoPipeline",
    ),
    "LingBotVideoImageToVideoPipeline": (
        ".pipeline_lingbot_video_i2v",
        "LingBotVideoImageToVideoPipeline",
    ),
    "LingBotVideoTransformer3DModel": (
        ".transformer_lingbot_video",
        "LingBotVideoTransformer3DModel",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
