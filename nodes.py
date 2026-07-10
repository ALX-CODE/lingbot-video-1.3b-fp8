from __future__ import annotations

import gc
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from PIL import Image

import comfy.model_management as mm
import comfy.utils

from diffusers import AutoencoderKLWan

from .lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
from .lingbot_video.pipeline_lingbot_video_i2v import (
    SPATIAL_MERGE_SIZE,
    _pixel_tensor_to_pil,
    smart_resize,
)
from .lingbot_video.prompt_rewriter import (
    build_expand_prompt,
    build_json_prompt,
    build_repair_prompt,
    extract_json_object,
    normalize_generated_prompt,
    repair_missing_element_actions,
)
from .lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
from .lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
from .lingbot_video.utils import caption_from_sample


PACK_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_ROOT = Path.cwd() / "models" / "lingbot_video"
DEFAULT_DENSE_DIR = DEFAULT_MODELS_ROOT / "lingbot-video-dense-1.3b"
DEFAULT_MOE_DIR = DEFAULT_MODELS_ROOT / "lingbot-video-moe-30b-a3b"
CATEGORY = "LingBot Video"
PROMPT_REWRITE_CACHE_LIMIT = 32
PROMPT_REWRITE_CACHE_VERSION = "2026-07-10-v1"

_PROMPT_REWRITE_CACHE: OrderedDict[str, tuple[str, str]] = OrderedDict()
_PROMPT_REWRITE_CACHE_LOCK = threading.Lock()


def _preflight_fp8_runtime(device: torch.device) -> None:
    """Fail early when the selected CUDA runtime cannot execute LingBot FP8 GEMMs."""
    if device.type != "cuda":
        raise RuntimeError(
            f"LingBot FP8 requires a CUDA device; ComfyUI selected {device}. "
            "Select the BF16 'transformer' checkpoint for a non-FP8 fallback."
        )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "This PyTorch build does not provide torch.float8_e4m3fn. "
            "Install an FP8-capable CUDA PyTorch build or select the BF16 'transformer' checkpoint."
        )
    scaled_mm = getattr(torch, "_scaled_mm", None)
    if not callable(scaled_mm):
        raise RuntimeError(
            "This PyTorch build does not provide the FP8 torch._scaled_mm kernel required by LingBot. "
            "Install an FP8-capable CUDA PyTorch build or select the BF16 'transformer' checkpoint."
        )

    try:
        capability = torch.cuda.get_device_capability(device)
        device_name = torch.cuda.get_device_name(device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect CUDA capability for {device}: {exc}. "
            "Select the BF16 'transformer' checkpoint if this GPU does not support FP8."
        ) from exc

    try:
        # Match the checkpoint's E4M3 x E4M3 -> BF16 path with a tiny aligned GEMM.
        lhs = torch.zeros((16, 16), device=device, dtype=torch.float8_e4m3fn)
        # cuBLAS expects the weight operand in the same transposed/column-major
        # layout used by _fp8_scaled_linear; a contiguous RHS can report a false
        # CUBLAS_STATUS_NOT_SUPPORTED even on a working Blackwell runtime.
        rhs = torch.zeros((16, 16), device=device, dtype=torch.float8_e4m3fn).t()
        lhs_scale = torch.ones(1, device=device, dtype=torch.float32)
        rhs_scale = torch.ones(1, device=device, dtype=torch.float32)
        result = scaled_mm(
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        if result.shape != (16, 16) or result.dtype != torch.bfloat16:
            raise RuntimeError(
                f"unexpected FP8 probe result: shape={tuple(result.shape)}, dtype={result.dtype}"
            )
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"LingBot FP8 is unavailable on {device_name} (CUDA capability "
            f"{capability[0]}.{capability[1]}, PyTorch {torch.__version__}, "
            f"CUDA {torch.version.cuda}): {exc}. Select the BF16 'transformer' checkpoint instead."
        ) from exc


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _frames_from_fps_seconds(fps: float, duration_seconds: float) -> int:
    """Return the closest LingBot-compatible (4n+1) frame count."""
    if fps <= 0:
        raise ValueError("FPS must be greater than zero")
    if duration_seconds <= 0:
        raise ValueError("Duration must be greater than zero")
    target_frames = fps * duration_seconds
    return max(1, 4 * round((target_frames - 1) / 4) + 1)


def _prompt_rewrite_cache_key(
    model_dir: Path,
    plain_prompt: str,
    duration_seconds: float,
    first_frame_pixel: torch.Tensor | None,
    last_frame_pixel: torch.Tensor | None,
) -> str:
    """Hash only inputs that can change Qwen's structured rewrite."""
    digest = hashlib.sha256()
    metadata = {
        "version": PROMPT_REWRITE_CACHE_VERSION,
        "model_dir": str(model_dir.resolve()),
        "plain_prompt": plain_prompt,
        "duration_seconds": float(duration_seconds),
    }
    digest.update(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for pixel in (first_frame_pixel, last_frame_pixel):
        if pixel is None:
            digest.update(b"\x00")
            continue
        value = pixel.detach().to(device="cpu").contiguous()
        digest.update(b"\x01")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(memoryview(value.numpy()))
    return digest.hexdigest()


def _get_cached_prompt_rewrite(cache_key: str) -> tuple[str, str] | None:
    with _PROMPT_REWRITE_CACHE_LOCK:
        cached = _PROMPT_REWRITE_CACHE.get(cache_key)
        if cached is not None:
            _PROMPT_REWRITE_CACHE.move_to_end(cache_key)
        return cached


def _cache_prompt_rewrite(cache_key: str, structured_json: str, expanded_caption: str) -> None:
    with _PROMPT_REWRITE_CACHE_LOCK:
        _PROMPT_REWRITE_CACHE[cache_key] = (structured_json, expanded_caption)
        _PROMPT_REWRITE_CACHE.move_to_end(cache_key)
        while len(_PROMPT_REWRITE_CACHE) > PROMPT_REWRITE_CACHE_LIMIT:
            _PROMPT_REWRITE_CACHE.popitem(last=False)


def _send_progress_text(node_id: str | int | None, text: str) -> None:
    """Show persistent, node-attached status text when the Comfy server is active."""
    if node_id is None:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_progress_text(text, str(node_id))
    except Exception:
        # Unit tests and direct Python calls do not have a running PromptServer.
        pass


def _discover_model_dirs() -> list[str]:
    roots = [DEFAULT_MODELS_ROOT]
    env_roots = os.environ.get("LINGBOT_MODEL_ROOTS", "")
    roots.extend(Path(item) for item in env_roots.split(os.pathsep) if item.strip())
    try:
        import folder_paths

        roots.append(Path(folder_paths.models_dir) / "lingbot_video")
    except Exception:
        pass

    found: list[Path] = []
    preferred = [DEFAULT_DENSE_DIR, DEFAULT_MOE_DIR]
    for model_dir in preferred:
        if (model_dir / "model_index.json").is_file():
            found.append(model_dir)
    for root in roots:
        if not root.is_dir():
            continue
        for model_dir in root.iterdir():
            if model_dir.is_dir() and (model_dir / "model_index.json").is_file():
                found.append(model_dir)

    unique: list[str] = [DEFAULT_DENSE_DIR.name]
    seen: set[str] = {DEFAULT_DENSE_DIR.name.casefold()}
    for model_dir in found:
        resolved = str(model_dir.resolve())
        key = resolved.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _resolve_model_dir(model_dir: str) -> Path:
    """Resolve a portable model name or an explicit external model path."""
    requested = Path(model_dir).expanduser()
    if requested.is_absolute():
        return requested.resolve()

    candidates: list[Path] = []
    if len(requested.parts) == 1:
        try:
            import folder_paths

            candidates.append(Path(folder_paths.models_dir) / "lingbot_video" / requested)
        except Exception:
            pass
        candidates.append(DEFAULT_MODELS_ROOT / requested)
        env_roots = os.environ.get("LINGBOT_MODEL_ROOTS", "")
        candidates.extend(
            Path(item).expanduser() / requested
            for item in env_roots.split(os.pathsep)
            if item.strip()
        )
    candidates.append(Path.cwd() / requested)
    for candidate in candidates:
        if (candidate / "model_index.json").is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _component_dir(
    model_dir: Path,
    name: str,
    config_names: tuple[str, ...] = ("config.json",),
) -> Path:
    path = model_dir / name
    if not path.is_dir():
        raise FileNotFoundError(f"LingBot component directory does not exist: {path}")
    if not any((path / config_name).is_file() for config_name in config_names):
        expected = ", ".join(str(path / config_name) for config_name in config_names)
        raise FileNotFoundError(f"LingBot component config does not exist; expected one of: {expected}")
    return path


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mm.soft_empty_cache()


def _transformer_load_dtype(is_int8: bool) -> torch.dtype | None:
    return None if is_int8 else torch.bfloat16


def _free_comfy_models() -> None:
    mm.unload_all_models()
    _empty_cuda_cache()


def _load_default_negative_prompt() -> str:
    path = PACK_ROOT / "lingbot_video" / "default_negative_prompt.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.dumps(json.load(handle), ensure_ascii=False, separators=(",", ":"))


def _normalize_structured_prompt(prompt: str) -> str:
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError as exc:
        if prompt.lstrip().startswith('"caption":'):
            raise ValueError(
                "ComfyUI stripped the prompt's JSON braces. Dynamic Prompts must be disabled "
                "for LingBot structured JSON; refresh ComfyUI and reload the workflow."
            ) from exc
        raise ValueError(
            "LingBot prompt must be valid structured JSON, normally with a top-level 'caption' object"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("LingBot prompt JSON must be an object")
    normalized = caption_from_sample(payload).strip()
    if not normalized:
        raise ValueError("LingBot prompt JSON produced an empty caption")
    return normalized


def _normalize_negative_prompt(prompt: str) -> str:
    if not prompt.strip():
        return _load_default_negative_prompt()
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError as exc:
        raise ValueError("LingBot negative prompt must be valid JSON") from exc
    if not isinstance(payload, (dict, list)):
        raise ValueError("LingBot negative prompt JSON must be an object or array")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _prepare_ti2v_first_frame(
    image: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Center-crop a Comfy IMAGE to LingBot's requested geometry."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] < 3:
        shape = tuple(image.shape) if isinstance(image, torch.Tensor) else type(image).__name__
        raise ValueError(f"LingBot TI2V expects a Comfy IMAGE [B,H,W,C], got {shape}")
    if image.shape[0] < 1:
        raise ValueError("LingBot TI2V requires at least one input image")
    LingBotVideoPipeline.check_inputs(int(height), int(width), 1)
    frame = image[0, :, :, :3].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    chw = frame.permute(2, 0, 1).unsqueeze(0)
    old_h, old_w = chw.shape[-2:]
    scale = max(int(height) / old_h, int(width) / old_w)
    new_h = max(int(round(old_h * scale)), int(height))
    new_w = max(int(round(old_w * scale)), int(width))
    resized = torch.nn.functional.interpolate(
        chw,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    top = max(0, (new_h - int(height)) // 2)
    left = max(0, (new_w - int(width)) // 2)
    cropped = resized[:, :, top : top + int(height), left : left + int(width)]
    return cropped.clamp_(0.0, 1.0).unsqueeze(2).contiguous()


def _ti2v_vlm_image(
    pixel: torch.Tensor,
    text_encoder: torch.nn.Module,
    processor: Any,
) -> Image.Image:
    """Match the official TI2V visual-token resize without changing DiT geometry."""
    patch_size = 16
    for obj in (
        getattr(getattr(text_encoder, "config", None), "vision_config", None),
        getattr(getattr(processor, "image_processor", None), "config", None),
        getattr(processor, "image_processor", None),
    ):
        value = getattr(obj, "patch_size", None)
        if value is not None:
            patch_size = int(value)
            break
    image = _pixel_tensor_to_pil(pixel)
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * SPATIAL_MERGE_SIZE,
    )
    return image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)


@dataclass
class LingBotModelHandle:
    model_dir: Path
    transformer_subfolder: str
    transformer: LingBotVideoTransformer3DModel
    vae: AutoencoderKLWan
    scheduler: FlowUniPCMultistepScheduler
    device: torch.device
    group_offload: bool
    experts_quant: str | None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class _ComfyLingBotPipeline(LingBotVideoPipeline):
    def set_progress_context(
        self,
        node_id: str | int | None,
        label: str = "Stage 3/5 · Denoising",
    ) -> "_ComfyLingBotPipeline":
        self._comfy_progress_node_id = node_id
        self._comfy_progress_label = label
        return self

    def set_execution_device(self, execution_device: torch.device) -> "_ComfyLingBotPipeline":
        self._comfy_execution_device = torch.device(execution_device)
        return self

    @property
    def _execution_device(self) -> torch.device:
        if hasattr(self, "_comfy_execution_device"):
            return self._comfy_execution_device
        return super()._execution_device

    @property
    def device(self) -> torch.device:
        if hasattr(self, "_comfy_execution_device"):
            return self._comfy_execution_device
        return super().device

    def progress_bar(
        self,
        iterable: Iterable[Any] | None = None,
        total: int | None = None,
    ) -> Iterator[Any]:
        if iterable is None:
            raise ValueError("LingBot progress requires an iterable")
        items = iterable
        item_count = total if total is not None else len(items)  # type: ignore[arg-type]
        item_count = int(item_count)
        node_id = getattr(self, "_comfy_progress_node_id", None)
        label = getattr(self, "_comfy_progress_label", "Denoising")
        progress = comfy.utils.ProgressBar(item_count, node_id=node_id)
        started = time.perf_counter()
        _send_progress_text(
            node_id,
            f"{label}\nStep 0/{item_count} · 0%\nElapsed 0s · ETA calculating…",
        )
        for index, item in enumerate(items, start=1):
            yield item
            progress.update(1)
            elapsed = time.perf_counter() - started
            eta = (elapsed / index) * (item_count - index)
            percent = round(index * 100 / item_count)
            _send_progress_text(
                node_id,
                f"{label}\nStep {index}/{item_count} · {percent}%\n"
                f"Elapsed {_format_duration(elapsed)} · ETA {_format_duration(eta)}",
            )


class LingBotPromptSettings:
    """Prompt-dependent geometry and duration, isolated from sampling controls."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "width": ("INT", {"default": 640, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 352, "min": 16, "max": 4096, "step": 16}),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 120.0,
                        "step": 0.1,
                        "tooltip": "Only these prompt-dependent values invalidate the Qwen rewrite cache.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "duration_seconds")
    FUNCTION = "settings"
    CATEGORY = CATEGORY

    def settings(self, width: int, height: int, duration_seconds: float) -> tuple[int, int, float]:
        if duration_seconds <= 0:
            raise ValueError("Duration must be greater than zero")
        LingBotVideoPipeline.check_inputs(int(height), int(width), 1)
        return int(width), int(height), float(duration_seconds)


class LingBotGenerationSettings:
    """One place to tune all resolution, duration, and sampling controls."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "width": ("INT", {"default": 640, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 352, "min": 16, "max": 4096, "step": 16}),
                "fps": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 120.0,
                        "step": 0.1,
                        "tooltip": "Frame count is calculated from FPS × seconds and snapped to LingBot's required 4n+1 format.",
                    },
                ),
                "steps": ("INT", {"default": 40, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 20.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "INT", "FLOAT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = (
        "width",
        "height",
        "num_frames",
        "fps",
        "steps",
        "cfg",
        "shift",
        "seed",
        "duration_seconds",
    )
    FUNCTION = "settings"
    CATEGORY = CATEGORY

    def settings(
        self,
        width: int,
        height: int,
        fps: float,
        duration_seconds: float,
        steps: int,
        cfg: float,
        shift: float,
        seed: int,
    ) -> tuple[int, int, int, float, int, float, float, int, float]:
        num_frames = _frames_from_fps_seconds(float(fps), float(duration_seconds))
        LingBotVideoPipeline.check_inputs(int(height), int(width), int(num_frames))
        return (
            int(width),
            int(height),
            int(num_frames),
            float(fps),
            int(steps),
            float(cfg),
            float(shift),
            int(seed),
            float(duration_seconds),
        )


class LingBotPostProcessSettings:
    """Central controls for lazy optional interpolation and upscaling branches."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "source_fps": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "enable_rife": ("BOOLEAN", {"default": False}),
                "rife_fps_multiplier": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.25,
                        "tooltip": "2.0 converts 20 FPS to 40 FPS. RIFE is skipped completely when disabled.",
                    },
                ),
                "enable_flashvsr": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "BOOLEAN", "FLOAT")
    RETURN_NAMES = ("rife_enabled", "rife_target_fps", "flashvsr_enabled", "output_fps")
    FUNCTION = "settings"
    CATEGORY = CATEGORY

    def settings(
        self,
        source_fps: float,
        enable_rife: bool,
        rife_fps_multiplier: float,
        enable_flashvsr: bool,
    ) -> tuple[bool, float, bool, float]:
        source_fps = float(source_fps)
        target_fps = source_fps * float(rife_fps_multiplier)
        if target_fps > 240.0:
            raise ValueError(f"RIFE target FPS must be <= 240, got {target_fps:g}")
        output_fps = target_fps if enable_rife else source_fps
        return bool(enable_rife), target_fps, bool(enable_flashvsr), output_fps


class LingBotLazyImageSwitch:
    """Select one IMAGE branch without evaluating the disabled expensive branch."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                "original": ("IMAGE", {"lazy": True}),
                "processed": ("IMAGE", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "select"
    CATEGORY = CATEGORY

    @classmethod
    def check_lazy_status(
        cls,
        enabled: bool,
        original: torch.Tensor | None = None,
        processed: torch.Tensor | None = None,
    ) -> list[str] | None:
        if enabled and processed is None:
            return ["processed"]
        if not enabled and original is None:
            return ["original"]
        return None

    def select(
        self,
        enabled: bool,
        original: torch.Tensor,
        processed: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        return (processed if enabled else original,)


class LingBotLatentSink:
    """Cheap output node for timing denoising without VAE decode or video encode."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"latents": ("LINGBOT_LATENTS",)}}

    RETURN_TYPES = ()
    FUNCTION = "consume"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def consume(self, latents: torch.Tensor) -> dict[str, Any]:
        return {"ui": {"text": [f"LingBot latent shape: {tuple(latents.shape)}"]}}


class LingBotModelLoader:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model_dir": (_discover_model_dirs(),),
                "transformer_subfolder": (
                    [
                        "transformer_fp8_dense",
                        "transformer",
                    ],
                ),
                "group_offload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Leave off for the dense 1.3B FP8 or BF16 transformer.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LINGBOT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        model_dir: str,
        transformer_subfolder: str,
        group_offload: bool,
        unique_id: str | None = None,
    ) -> tuple[LingBotModelHandle]:
        _send_progress_text(unique_id, "Stage 1/5 · Load Model\nPhase 1/3 · Validating local model files")
        model_path = _resolve_model_dir(model_dir)
        if not (model_path / "model_index.json").is_file():
            raise FileNotFoundError(f"Not a LingBot diffusers model directory: {model_path}")
        transformer_path = _component_dir(model_path, transformer_subfolder)
        _component_dir(model_path, "vae")
        _component_dir(model_path, "scheduler", ("scheduler_config.json",))
        _component_dir(model_path, "text_encoder")
        if not (model_path / "processor").is_dir():
            raise FileNotFoundError(f"LingBot processor directory does not exist: {model_path / 'processor'}")

        with (transformer_path / "config.json").open("r", encoding="utf-8") as handle:
            transformer_config = json.load(handle)
        experts_quant = transformer_config.get("experts_quant")
        dense_quant = transformer_config.get("dense_quant")
        dense_fp8 = transformer_config.get("dense_fp8")
        is_moe_int8 = experts_quant == "int8_rowwise"
        is_dense_int8 = dense_quant == "int8_rowwise"
        is_dense_fp8 = dense_fp8 == "e4m3fn_tensorwise_fused"
        is_int8 = is_moe_int8 or is_dense_int8
        is_quantized = is_int8 or is_dense_fp8
        if transformer_subfolder == "transformer_int8" and not is_moe_int8:
            raise ValueError(
                f"{transformer_path / 'config.json'} must declare experts_quant='int8_rowwise'"
            )
        if transformer_subfolder == "transformer_int8_dense" and not is_dense_int8:
            raise ValueError(
                f"{transformer_path / 'config.json'} must declare dense_quant='int8_rowwise'"
            )
        if transformer_subfolder == "transformer_fp8_dense" and not is_dense_fp8:
            raise ValueError(
                f"{transformer_path / 'config.json'} must declare "
                "dense_fp8='e4m3fn_tensorwise_fused'"
            )
        if is_moe_int8 and is_dense_int8:
            raise ValueError("A LingBot transformer cannot use dense and MoE INT8 formats together")
        effective_group_offload = bool(group_offload or is_moe_int8)
        if is_moe_int8 and not group_offload:
            print("[LingBot] group_offload was disabled but is mandatory for INT8 MoE; forcing it on.")

        device = torch.device(mm.get_torch_device())
        if device.type != "cuda":
            raise RuntimeError(f"LingBot Video requires CUDA on this machine; ComfyUI selected {device}")
        if is_dense_fp8:
            _send_progress_text(unique_id, "Stage 1/5 · Load Model\nFP8 runtime preflight")
            _preflight_fp8_runtime(device)

        _free_comfy_models()

        load_dtype = _transformer_load_dtype(is_quantized)
        print(
            f"[LingBot] loading transformer from {transformer_path} "
            f"with torch_dtype={load_dtype}, low_cpu_mem_usage=True"
        )
        _send_progress_text(unique_id, "Stage 1/5 · Load Model\nPhase 2/3 · Loading dense transformer")
        transformer = LingBotVideoTransformer3DModel.from_pretrained(
            str(model_path),
            subfolder=transformer_subfolder,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        if is_dense_int8 or is_dense_fp8:
            # Diffusers builds mixed checkpoints in fp32 when torch_dtype=None.
            # Normalize retained dense weights to bf16 while the model's custom
            # .to() preserves INT8/FP8 storage and fp32 quantization scales.
            transformer.to(dtype=torch.bfloat16)

        if effective_group_offload:
            print("[LingBot] enabling low-memory block group offload")
            transformer.enable_group_offload(
                onload_device=device,
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                non_blocking=True,
                use_stream=True,
                low_cpu_mem_usage=True,
            )
        else:
            # The dense transformer stays on CPU until TextEncode has evicted Qwen.
            transformer.to("cpu")

        print(f"[LingBot] loading VAE and keeping it resident on {device}")
        _send_progress_text(unique_id, "Stage 1/5 · Load Model\nPhase 3/3 · Loading VAE + scheduler")
        vae = AutoencoderKLWan.from_pretrained(
            str(model_path),
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        vae.to(device=device, dtype=torch.bfloat16)

        scheduler = FlowUniPCMultistepScheduler.from_pretrained(
            str(model_path),
            subfolder="scheduler",
            local_files_only=True,
        )
        _empty_cuda_cache()
        _send_progress_text(unique_id, "Stage 1/5 · Load Model\nComplete · Transformer, VAE, and scheduler ready")
        return (
            LingBotModelHandle(
                model_dir=model_path,
                transformer_subfolder=transformer_subfolder,
                transformer=transformer,
                vae=vae,
                scheduler=scheduler,
                device=device,
                group_offload=effective_group_offload,
                experts_quant=experts_quant,
            ),
        )


class LingBotTextEncode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("LINGBOT_MODEL",),
                "prompt": (
                    "STRING",
                    {
                        "default": '{"caption":{"comprehensive_description":{"scene_content_description":"","camera_movement_description":""}},"duration":3}',
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Structured LingBot prompt JSON. Dynamic Prompts is intentionally disabled so ComfyUI does not strip JSON braces.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "JSON only. Leave empty to use the vendored LingBot video negative prompt.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LINGBOT_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    @torch.no_grad()
    def encode(
        self,
        model: LingBotModelHandle,
        prompt: str,
        negative_prompt: str,
        unique_id: str | None = None,
    ) -> tuple[dict[str, torch.Tensor]]:
        _send_progress_text(unique_id, "Stage 2/5 · Encode Prompt\nPhase 1/4 · Validating structured JSON")
        structured_prompt = _normalize_structured_prompt(prompt)
        negative = _normalize_negative_prompt(negative_prompt)

        with model.lock:
            _free_comfy_models()
            if _module_device(model.vae) != model.device:
                model.vae.to(device=model.device, dtype=torch.bfloat16)

            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

            text_encoder_path = model.model_dir / "text_encoder"
            processor_path = model.model_dir / "processor"
            print(f"[LingBot] loading Qwen3VL from {text_encoder_path} with SDPA")
            _send_progress_text(unique_id, "Stage 2/5 · Encode Prompt\nPhase 2/4 · Loading Qwen3VL")
            processor = Qwen3VLProcessor.from_pretrained(
                str(processor_path),
                local_files_only=True,
            )
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                str(text_encoder_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).eval()
            text_encoder.config.use_cache = False
            text_encoder.to(model.device)

            pipe = _ComfyLingBotPipeline(
                transformer=None,
                vae=None,
                text_encoder=text_encoder,
                processor=processor,
                scheduler=None,
            ).set_execution_device(model.device)
            try:
                _send_progress_text(unique_id, "Stage 2/5 · Encode Prompt\nPhase 3/4 · Encoding positive + negative prompts")
                positive_embeds, positive_mask = pipe.encode_prompt(
                    structured_prompt,
                    device=model.device,
                )
                negative_embeds, negative_mask = pipe.encode_prompt(
                    negative,
                    device=model.device,
                )
                conditioning = {
                    "prompt_embeds": positive_embeds.detach().cpu(),
                    "prompt_mask": positive_mask.detach().cpu(),
                    "negative_prompt_embeds": negative_embeds.detach().cpu(),
                    "negative_prompt_mask": negative_mask.detach().cpu(),
                }
            finally:
                # This phase boundary is mandatory on 16GB: Qwen must leave VRAM
                # before the first transformer block is streamed onto the GPU.
                print("[LingBot] evicting Qwen3VL from VRAM before denoising")
                _send_progress_text(unique_id, "Stage 2/5 · Encode Prompt\nPhase 4/4 · Evicting Qwen from VRAM")
                pipe.text_encoder = None
                text_encoder.to("cpu")
                del pipe
                del text_encoder
                del processor
                _empty_cuda_cache()

            _send_progress_text(unique_id, "Stage 2/5 · Encode Prompt\nComplete · Conditioning cached on CPU")
            return (conditioning,)


class _BalancedJSONObjectStoppingCriteria:
    """Stop greedy generation immediately after its first complete JSON object."""

    def __init__(self, processor: Any, input_length: int) -> None:
        self.processor = processor
        self.input_length = int(input_length)
        self.processed_lengths: list[int] = []
        self.depths: list[int] = []
        self.started: list[bool] = []
        self.in_strings: list[bool] = []
        self.escaped: list[bool] = []
        self.finished: list[bool] = []

    def _ensure_batch(self, batch_size: int) -> None:
        while len(self.finished) < batch_size:
            self.processed_lengths.append(self.input_length)
            self.depths.append(0)
            self.started.append(False)
            self.in_strings.append(False)
            self.escaped.append(False)
            self.finished.append(False)

    def _consume(self, row: int, fragment: str) -> None:
        for char in fragment:
            if self.finished[row]:
                return
            if self.in_strings[row]:
                if self.escaped[row]:
                    self.escaped[row] = False
                elif char == "\\":
                    self.escaped[row] = True
                elif char == '"':
                    self.in_strings[row] = False
                continue
            if char == '"' and self.started[row]:
                self.in_strings[row] = True
            elif char == "{":
                self.started[row] = True
                self.depths[row] += 1
            elif char == "}" and self.started[row]:
                self.depths[row] -= 1
                if self.depths[row] == 0:
                    self.finished[row] = True

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor | None,
        **_: Any,
    ) -> torch.BoolTensor:
        batch_size = int(input_ids.shape[0])
        self._ensure_batch(batch_size)
        for row in range(batch_size):
            start = self.processed_lengths[row]
            end = int(input_ids.shape[1])
            if not self.finished[row] and end > start:
                fragment = self.processor.decode(
                    input_ids[row, start:end].tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                self._consume(row, fragment)
            self.processed_lengths[row] = end
        return torch.tensor(self.finished, dtype=torch.bool, device=input_ids.device)


def _qwen_generate_text(
    text_encoder: torch.nn.Module,
    processor: Any,
    instruction: str,
    *,
    max_new_tokens: int,
    image: Image.Image | None = None,
    additional_images: list[Image.Image] | None = None,
    stop_at_json_object: bool = False,
) -> tuple[str, int]:
    visual_images = ([image] if image is not None else []) + list(additional_images or [])
    content: list[dict[str, Any]] = []
    for visual_image in visual_images:
        content.append({"type": "image", "image": visual_image})
    content.append({"type": "text", "text": instruction})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    chat = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[chat],
        images=(visual_images or None),
        return_tensors="pt",
    ).to(_module_device(text_encoder))
    input_tokens = int(inputs["input_ids"].shape[1])
    generate_options: dict[str, Any] = {}
    if stop_at_json_object:
        from transformers import StoppingCriteriaList

        generate_options["stopping_criteria"] = StoppingCriteriaList(
            [_BalancedJSONObjectStoppingCriteria(processor, input_tokens)]
        )
    generated = text_encoder.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
        **generate_options,
    )
    new_tokens = generated[:, input_tokens:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return text, int(new_tokens.shape[1])


def _ground_rewrite_to_first_frame(instruction: str, has_image: bool) -> str:
    if not has_image:
        return instruction
    return f"""An image is attached and is the exact first frame of the requested video.
Treat its visible identity, subject count, faces, anatomy, clothing, objects,
colors, lighting, viewpoint, and scene layout as ground truth. Describe a
temporally coherent continuation from that frame. Do not invent details that
contradict it, and do not describe a cut to a different opening composition.

{instruction}"""


def _ground_rewrite_to_visual_frames(instruction: str, image_count: int) -> str:
    if image_count < 2:
        return _ground_rewrite_to_first_frame(instruction, image_count == 1)
    return f"""Two ordered images are attached. The first image is the exact first
frame and the second image is the exact last frame of one continuous video.
Treat both endpoints' visible identities, subject counts, faces, anatomy,
clothing, objects, colors, lighting, viewpoints, and scene layouts as ground
truth. Describe a physically plausible continuous transition from the first
image to the second over the full requested duration. Preserve identity and
object continuity; do not introduce a cut, teleportation, reversal, or a new
composition that is absent from both endpoints.

{instruction}"""


class LingBotPromptEncode:
    """Rewrite plain prose to LingBot JSON, then encode it with the same Qwen load."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("LINGBOT_MODEL",),
                "plaintext_prompt": (
                    "STRING",
                    {
                        "default": "A cinematic medium shot of a woman walking through a sunlit apartment, then turning toward the camera with a subtle smile.",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Write normal prose here. Qwen expands it and creates the complete LingBot structured JSON.",
                    },
                ),
                "structured_json_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Optional advanced override. When non-empty, this JSON is encoded directly and plaintext rewriting is skipped.",
                    },
                ),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 120.0,
                        "step": 0.1,
                        "tooltip": "Connect this to Generation Settings so Qwen's action timestamps match the requested video duration.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "JSON only. Leave empty to use LingBot's vendored default negative prompt.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LINGBOT_CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "structured_json", "expanded_caption")
    FUNCTION = "rewrite_and_encode"
    CATEGORY = CATEGORY

    @torch.no_grad()
    def rewrite_and_encode(
        self,
        model: LingBotModelHandle,
        plaintext_prompt: str,
        structured_json_override: str,
        duration_seconds: float,
        negative_prompt: str,
        unique_id: str | None = None,
        first_frame: torch.Tensor | None = None,
        last_frame: torch.Tensor | None = None,
        width: int = 640,
        height: int = 352,
    ) -> tuple[dict[str, torch.Tensor], str, str]:
        override = structured_json_override.strip()
        plain = plaintext_prompt.strip()
        if not override and not plain:
            raise ValueError("Enter a plaintext prompt or a structured JSON override")
        if duration_seconds <= 0:
            raise ValueError("Duration must be greater than zero")

        _send_progress_text(unique_id, "Stage 2/5 · Prompt + Qwen\nPhase 1/7 · Validating inputs")
        negative = _normalize_negative_prompt(negative_prompt)
        expanded_caption = ""
        structured_json = ""
        condition_pixel = (
            _prepare_ti2v_first_frame(first_frame, int(height), int(width))
            if first_frame is not None
            else None
        )
        last_condition_pixel = (
            _prepare_ti2v_first_frame(last_frame, int(height), int(width))
            if last_frame is not None
            else None
        )
        if last_condition_pixel is not None and condition_pixel is None:
            raise ValueError("LingBot FLF requires both a first frame and a last frame")

        rewrite_cache_key: str | None = None
        cached_rewrite: tuple[str, str] | None = None
        if not override:
            rewrite_cache_key = _prompt_rewrite_cache_key(
                model.model_dir,
                plain,
                duration_seconds,
                condition_pixel,
                last_condition_pixel,
            )
            cached_rewrite = _get_cached_prompt_rewrite(rewrite_cache_key)

        with model.lock:
            _free_comfy_models()
            if _module_device(model.vae) != model.device:
                model.vae.to(device=model.device, dtype=torch.bfloat16)

            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

            text_encoder_path = model.model_dir / "text_encoder"
            processor_path = model.model_dir / "processor"
            print(f"[LingBot] loading one Qwen3VL instance for rewrite + encode from {text_encoder_path}")
            _send_progress_text(unique_id, "Stage 2/5 · Prompt + Qwen\nPhase 2/7 · Loading Qwen3VL once")
            processor = Qwen3VLProcessor.from_pretrained(
                str(processor_path),
                local_files_only=True,
            )
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                str(text_encoder_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).eval()
            text_encoder.to(model.device)

            vlm_image = (
                _ti2v_vlm_image(condition_pixel, text_encoder, processor)
                if condition_pixel is not None
                else None
            )
            vlm_last_image = (
                _ti2v_vlm_image(last_condition_pixel, text_encoder, processor)
                if last_condition_pixel is not None
                else None
            )
            vlm_images = [item for item in (vlm_image, vlm_last_image) if item is not None]

            pipe = _ComfyLingBotPipeline(
                transformer=None,
                vae=None,
                text_encoder=text_encoder,
                processor=processor,
                scheduler=None,
            ).set_execution_device(model.device)
            try:
                if override:
                    # Keep the legacy direct-JSON path deliberately permissive.
                    _normalize_structured_prompt(override)
                    structured_json = json.dumps(
                        json.loads(override),
                        ensure_ascii=False,
                        indent=2,
                    )
                    expanded_caption = "Direct structured JSON override"
                    _send_progress_text(
                        unique_id,
                        "Stage 2/5 · Prompt + Qwen\nPhase 3-5/7 · Direct JSON override; rewrite skipped",
                    )
                elif cached_rewrite is not None:
                    structured_json, expanded_caption = cached_rewrite
                    print("[LingBot] structured rewrite cache hit; Qwen generation skipped")
                    _send_progress_text(
                        unique_id,
                        "Stage 2/5 · Prompt + Qwen\nPhase 3-5/7 · Cached structured JSON; rewrite skipped",
                    )
                else:
                    started = time.perf_counter()
                    _send_progress_text(
                        unique_id,
                        "Stage 2/5 · Prompt + Qwen\nPhase 3/7 · Expanding plaintext with timestamps",
                    )
                    expanded_caption, expand_tokens = _qwen_generate_text(
                        text_encoder,
                        processor,
                        _ground_rewrite_to_visual_frames(
                            build_expand_prompt(plain, duration_seconds),
                            len(vlm_images),
                        ),
                        max_new_tokens=1024,
                        image=vlm_image,
                        additional_images=([vlm_last_image] if vlm_last_image is not None else None),
                    )
                    print(
                        f"[LingBot] plaintext expansion: {expand_tokens} tokens in "
                        f"{time.perf_counter() - started:.1f}s"
                    )

                    _send_progress_text(
                        unique_id,
                        "Stage 2/5 · Prompt + Qwen\nPhase 4/7 · Mapping caption to structured JSON",
                    )
                    raw_json, map_tokens = _qwen_generate_text(
                        text_encoder,
                        processor,
                        _ground_rewrite_to_visual_frames(
                            build_json_prompt(expanded_caption, duration_seconds),
                            len(vlm_images),
                        ),
                        max_new_tokens=4096,
                        image=vlm_image,
                        additional_images=([vlm_last_image] if vlm_last_image is not None else None),
                        stop_at_json_object=True,
                    )
                    print(f"[LingBot] JSON mapping generated {map_tokens} tokens")

                    _send_progress_text(
                        unique_id,
                        "Stage 2/5 · Prompt + Qwen\nPhase 5/7 · Validating structured JSON",
                    )
                    errors: list[str]
                    try:
                        payload = extract_json_object(raw_json)
                        structured_json, errors = normalize_generated_prompt(
                            payload,
                            duration_seconds,
                        )
                        if errors:
                            payload, action_repairs = repair_missing_element_actions(
                                payload,
                                duration_seconds,
                                expanded_caption,
                            )
                            if action_repairs:
                                structured_json, errors = normalize_generated_prompt(
                                    payload,
                                    duration_seconds,
                                )
                                if not errors:
                                    print(
                                        f"[LingBot] deterministically restored {action_repairs} "
                                        "missing action array(s); Qwen repair skipped"
                                    )
                    except ValueError as exc:
                        errors = [str(exc)]

                    if errors:
                        print(f"[LingBot] Qwen JSON needs one repair pass: {'; '.join(errors)}")
                        _send_progress_text(
                            unique_id,
                            "Stage 2/5 · Prompt + Qwen\nPhase 5/7 · Repairing invalid JSON once",
                        )
                        repaired, _ = _qwen_generate_text(
                            text_encoder,
                            processor,
                            _ground_rewrite_to_visual_frames(
                                build_repair_prompt(
                                    raw_json,
                                    errors,
                                    expanded_caption,
                                    duration_seconds,
                                ),
                                len(vlm_images),
                            ),
                            max_new_tokens=4096,
                            image=vlm_image,
                            additional_images=([vlm_last_image] if vlm_last_image is not None else None),
                            stop_at_json_object=True,
                        )
                        try:
                            repaired_payload = extract_json_object(repaired)
                            structured_json, errors = normalize_generated_prompt(
                                repaired_payload,
                                duration_seconds,
                            )
                            if errors:
                                repaired_payload, action_repairs = repair_missing_element_actions(
                                    repaired_payload,
                                    duration_seconds,
                                    expanded_caption,
                                )
                                if action_repairs:
                                    structured_json, errors = normalize_generated_prompt(
                                        repaired_payload,
                                        duration_seconds,
                                    )
                                    if not errors:
                                        print(
                                            f"[LingBot] deterministically restored {action_repairs} "
                                            "missing action array(s) after Qwen repair"
                                        )
                        except ValueError as exc:
                            errors = [str(exc)]
                        if errors:
                            raise ValueError(
                                "Qwen could not produce valid LingBot structured JSON after one repair: "
                                + "; ".join(errors)
                            )

                    if rewrite_cache_key is not None:
                        _cache_prompt_rewrite(
                            rewrite_cache_key,
                            structured_json,
                            expanded_caption,
                        )
                        print("[LingBot] cached structured rewrite independently of prompt embeddings")

                structured_prompt = _normalize_structured_prompt(structured_json)
                text_encoder.config.use_cache = False
                _send_progress_text(
                    unique_id,
                    "Stage 2/5 · Prompt + Qwen\nPhase 6/7 · Encoding positive + negative prompts",
                )
                positive_embeds, positive_mask = pipe.encode_prompt(
                    structured_prompt,
                    images=(vlm_images or None),
                    device=model.device,
                )
                negative_embeds, negative_mask = pipe.encode_prompt(
                    negative,
                    images=(vlm_images or None),
                    device=model.device,
                )
                conditioning = {
                    "prompt_embeds": positive_embeds.detach().cpu(),
                    "prompt_mask": positive_mask.detach().cpu(),
                    "negative_prompt_embeds": negative_embeds.detach().cpu(),
                    "negative_prompt_mask": negative_mask.detach().cpu(),
                }
                if condition_pixel is not None:
                    conditioning["first_frame_pixel"] = condition_pixel.detach().cpu()
                if last_condition_pixel is not None:
                    conditioning["last_frame_pixel"] = last_condition_pixel.detach().cpu()
            finally:
                print("[LingBot] evicting rewrite/encode Qwen3VL before denoising")
                _send_progress_text(
                    unique_id,
                    "Stage 2/5 · Prompt + Qwen\nPhase 7/7 · Evicting Qwen from VRAM",
                )
                pipe.text_encoder = None
                text_encoder.to("cpu")
                del pipe
                del text_encoder
                del processor
                _empty_cuda_cache()

            _send_progress_text(
                unique_id,
                "Stage 2/5 · Prompt + Qwen\nComplete · JSON ready; conditioning cached on CPU",
            )
            return conditioning, structured_json, expanded_caption


class LingBotTI2VPromptEncode(LingBotPromptEncode):
    """Use the same first frame for rewriting, Qwen-VL conditioning, and VAE conditioning."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("LINGBOT_MODEL",),
                "first_frame": (
                    "IMAGE",
                    {
                        "tooltip": "The first image in the batch is center-cropped to the selected video aspect ratio and fixed as frame 0.",
                    },
                ),
                "plaintext_prompt": (
                    "STRING",
                    {
                        "default": "The subject begins from the supplied first frame, then moves naturally while preserving identity, clothing, lighting, and scene layout. Stable cinematic camera motion.",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Describe what should happen after the supplied first frame. Qwen sees the image during both rewrite stages.",
                    },
                ),
                "structured_json_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Optional advanced override. The supplied image is still used by Qwen-VL and as frame-0 conditioning.",
                    },
                ),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.1, "max": 120.0, "step": 0.1},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "JSON only. Leave empty to use LingBot's vendored default negative prompt.",
                    },
                ),
                "width": ("INT", {"default": 640, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 352, "min": 16, "max": 4096, "step": 16}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LINGBOT_CONDITIONING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = (
        "conditioning",
        "structured_json",
        "expanded_caption",
        "conditioned_first_frame",
    )
    FUNCTION = "rewrite_and_encode_ti2v"

    @torch.no_grad()
    def rewrite_and_encode_ti2v(
        self,
        model: LingBotModelHandle,
        first_frame: torch.Tensor,
        plaintext_prompt: str,
        structured_json_override: str,
        duration_seconds: float,
        negative_prompt: str,
        width: int,
        height: int,
        unique_id: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], str, str, torch.Tensor]:
        conditioning, structured_json, expanded_caption = self.rewrite_and_encode(
            model=model,
            plaintext_prompt=plaintext_prompt,
            structured_json_override=structured_json_override,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            unique_id=unique_id,
            first_frame=first_frame,
            width=int(width),
            height=int(height),
        )
        pixel = conditioning["first_frame_pixel"]
        preview = pixel[:, :, 0].permute(0, 2, 3, 1).contiguous()
        return conditioning, structured_json, expanded_caption, preview


class LingBotFLFPromptEncode(LingBotPromptEncode):
    """Experimental two-image prompt path for first/last latent endpoint pinning."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("LINGBOT_MODEL",),
                "first_frame": ("IMAGE", {"tooltip": "Exact opening frame."}),
                "last_frame": ("IMAGE", {"tooltip": "Experimental exact ending-frame target."}),
                "plaintext_prompt": (
                    "STRING",
                    {
                        "default": "Create one continuous, physically plausible transition from the supplied first frame to the supplied last frame. Preserve subject identity, anatomy, clothing, objects, lighting, and scene continuity with no cuts or teleportation.",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Describe the motion connecting the first image to the last image. Qwen sees both ordered endpoints.",
                    },
                ),
                "structured_json_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Optional structured JSON override. Both endpoint images still condition Qwen and the latent sampler.",
                    },
                ),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.2, "max": 120.0, "step": 0.1},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "width": ("INT", {"default": 640, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 352, "min": 16, "max": 4096, "step": 16}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LINGBOT_CONDITIONING", "STRING", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "conditioning",
        "structured_json",
        "expanded_caption",
        "conditioned_first_frame",
        "conditioned_last_frame",
    )
    FUNCTION = "rewrite_and_encode_flf"

    @torch.no_grad()
    def rewrite_and_encode_flf(
        self,
        model: LingBotModelHandle,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        plaintext_prompt: str,
        structured_json_override: str,
        duration_seconds: float,
        negative_prompt: str,
        width: int,
        height: int,
        unique_id: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], str, str, torch.Tensor, torch.Tensor]:
        conditioning, structured_json, expanded_caption = self.rewrite_and_encode(
            model=model,
            plaintext_prompt=plaintext_prompt,
            structured_json_override=structured_json_override,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            unique_id=unique_id,
            first_frame=first_frame,
            last_frame=last_frame,
            width=int(width),
            height=int(height),
        )
        first = conditioning["first_frame_pixel"][:, :, 0].permute(0, 2, 3, 1).contiguous()
        last = conditioning["last_frame_pixel"][:, :, 0].permute(0, 2, 3, 1).contiguous()
        return conditioning, structured_json, expanded_caption, first, last


class LingBotPromptPreview:
    """Display Qwen's generated JSON and expanded caption directly in ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "structured_json": ("STRING", {"forceInput": True}),
                "expanded_caption": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("structured_json",)
    FUNCTION = "preview"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def preview(self, structured_json: str, expanded_caption: str) -> dict[str, Any]:
        text = f"EXPANDED CAPTION\n{expanded_caption}\n\nSTRUCTURED JSON\n{structured_json}"
        return {"ui": {"text": [text]}, "result": (structured_json,)}


class LingBotSampler:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("LINGBOT_MODEL",),
                "conditioning": ("LINGBOT_CONDITIONING",),
                "width": ("INT", {"default": 640, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 352, "min": 16, "max": 4096, "step": 16}),
                "num_frames": ("INT", {"default": 73, "min": 1, "max": 1001, "step": 4}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 20.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "cfg_execution": (
                    [
                        "sequential_sage",
                        "batched_native",
                        "batched_sage_varlen",
                        "hybrid_sage_safe",
                    ],
                    {
                        "default": "sequential_sage",
                        "tooltip": "Sequential Sage is the conservative baseline. Batched native computes exact masked B=2 CFG in one pass. Sage-varlen is experimental. All batched modes auto-fallback above 16,384 video tokens or on CUDA OOM.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LINGBOT_LATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    @torch.no_grad()
    def sample(
        self,
        model: LingBotModelHandle,
        conditioning: dict[str, torch.Tensor],
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        cfg: float,
        shift: float,
        seed: int,
        cfg_execution: str = "sequential_sage",
        unique_id: str | None = None,
    ) -> tuple[torch.Tensor]:
        LingBotVideoPipeline.check_inputs(height, width, num_frames)
        required = {
            "prompt_embeds",
            "prompt_mask",
            "negative_prompt_embeds",
            "negative_prompt_mask",
        }
        missing = sorted(required.difference(conditioning))
        if missing:
            raise ValueError(f"LingBot conditioning is missing: {', '.join(missing)}")
        first_frame_pixel = conditioning.get("first_frame_pixel")
        last_frame_pixel = conditioning.get("last_frame_pixel")
        if first_frame_pixel is not None:
            if not isinstance(first_frame_pixel, torch.Tensor) or first_frame_pixel.ndim != 5:
                raise ValueError("LingBot TI2V first-frame conditioning must be [B,C,1,H,W]")
            if first_frame_pixel.shape[:3] != (1, 3, 1):
                raise ValueError(
                    "LingBot TI2V currently supports one RGB first frame; got "
                    f"{tuple(first_frame_pixel.shape)}"
                )
            if tuple(first_frame_pixel.shape[-2:]) != (int(height), int(width)):
                raise ValueError(
                    "TI2V resolution changed after prompt encoding: conditioned frame is "
                    f"{first_frame_pixel.shape[-1]}x{first_frame_pixel.shape[-2]}, sampler is "
                    f"{width}x{height}. Keep the same Generation Settings connections on both nodes."
                )
        if last_frame_pixel is not None:
            if first_frame_pixel is None:
                raise ValueError("LingBot FLF requires both first-frame and last-frame conditioning")
            if not isinstance(last_frame_pixel, torch.Tensor) or last_frame_pixel.ndim != 5:
                raise ValueError("LingBot FLF last-frame conditioning must be [B,C,1,H,W]")
            if last_frame_pixel.shape[:3] != (1, 3, 1):
                raise ValueError(
                    "LingBot FLF currently supports one RGB last frame; got "
                    f"{tuple(last_frame_pixel.shape)}"
                )
            if tuple(last_frame_pixel.shape[-2:]) != (int(height), int(width)):
                raise ValueError(
                    "FLF resolution changed after prompt encoding: conditioned last frame is "
                    f"{last_frame_pixel.shape[-1]}x{last_frame_pixel.shape[-2]}, sampler is "
                    f"{width}x{height}. Keep the same Generation Settings connections on both nodes."
                )
            if int(num_frames) < 5:
                raise ValueError("LingBot FLF requires at least 5 output frames")

        with model.lock:
            _free_comfy_models()
            if _module_device(model.vae) != model.device:
                raise RuntimeError(
                    "LingBot VAE left the CUDA device before denoising; this integration keeps it GPU-resident."
                )
            if not model.group_offload:
                model.transformer.to(model.device)

            pipe = _ComfyLingBotPipeline(
                transformer=model.transformer,
                vae=model.vae,
                text_encoder=None,
                processor=None,
                scheduler=model.scheduler,
            ).set_execution_device(model.device).set_progress_context(unique_id)
            generator = torch.Generator(device=model.device).manual_seed(int(seed))
            cond_latent = None
            cond_latent_last = None
            if first_frame_pixel is not None:
                is_flf = last_frame_pixel is not None
                _send_progress_text(
                    unique_id,
                    f"Stage 3/5 · {'FLF' if is_flf else 'TI2V'} Denoising\n"
                    f"Phase 1/2 · Encoding fixed {'endpoint' if is_flf else 'first-frame'} latent(s)",
                )
                cond_latent = pipe.encode_video_latent(first_frame_pixel, generator=generator)
                if last_frame_pixel is not None:
                    cond_latent_last = pipe.encode_video_latent(
                        last_frame_pixel,
                        generator=generator,
                    )
                pipe.set_progress_context(
                    unique_id,
                    f"Stage 3/5 · {'FLF' if is_flf else 'TI2V'} Denoising",
                )
                _send_progress_text(
                    unique_id,
                    f"Stage 3/5 · {'FLF' if is_flf else 'TI2V'} Denoising\n"
                    f"Phase 2/2 · {'Both endpoints' if is_flf else 'First frame'} fixed; starting diffusion",
                )
            # Accept the former values so saved workflows remain loadable.
            is_sequential = cfg_execution in {"sequential", "sequential_sage"}
            is_hybrid = cfg_execution == "hybrid_sage_safe"
            batch_backend = (
                "sage_varlen"
                if cfg_execution in {"batched_sage_full", "batched_sage_varlen", "hybrid_sage_safe"}
                else "native"
            )
            effective_cfg_mode = "sequential_sage"
            cfg_fallback_reason: str | None = None
            try:
                result = pipe(
                    prompt="",
                    negative_prompt="",
                    height=int(height),
                    width=int(width),
                    num_frames=int(num_frames),
                    num_inference_steps=int(steps),
                    guidance_scale=float(cfg),
                    shift=float(shift),
                    generator=generator,
                    prompt_embeds=conditioning["prompt_embeds"],
                    prompt_mask=conditioning["prompt_mask"],
                    negative_prompt_embeds=conditioning["negative_prompt_embeds"],
                    negative_prompt_mask=conditioning["negative_prompt_mask"],
                    cond_latent=cond_latent,
                    cond_latent_last=cond_latent_last,
                    output_type="latent",
                    batch_cfg=not is_sequential,
                    batch_cfg_fraction=0.25 if is_hybrid else 1.0,
                    batch_cfg_tail=is_hybrid,
                    attention_backend="sage",
                    batch_cfg_backend=batch_backend,
                    return_dict=True,
                )
                if pipe._last_batch_cfg_fallback_reason:
                    print(f"[LingBot] batched CFG fallback: {pipe._last_batch_cfg_fallback_reason}")
                if pipe._last_effective_batch_cfg:
                    effective_cfg_mode = (
                        f"hybrid_sage_tail {pipe._last_batch_cfg_steps}/{pipe._last_total_denoise_steps} steps"
                        if pipe._last_batch_cfg_steps < pipe._last_total_denoise_steps
                        else f"batched_{pipe._last_batch_cfg_backend}"
                    )
                else:
                    effective_cfg_mode = "sequential_sage"
                cfg_fallback_reason = pipe._last_batch_cfg_fallback_reason
                latents = result.frames.detach().to(device="cpu", dtype=torch.float32)
            finally:
                del pipe
                if not model.group_offload:
                    model.transformer.to("cpu")
                _empty_cuda_cache()
            cfg_status = f"CFG mode · {effective_cfg_mode}"
            if cfg_fallback_reason:
                cfg_status += f" (fallback: {cfg_fallback_reason})"
            _send_progress_text(
                unique_id,
                f"Stage 3/5 · Denoising\nComplete · {steps}/{steps} steps\n{cfg_status}",
            )
            return (latents,)


class LingBotTI2VSampler(LingBotSampler):
    """TI2V-only sampler surface; the proven base loop performs the actual denoising."""

    @torch.no_grad()
    def sample(
        self,
        model: LingBotModelHandle,
        conditioning: dict[str, torch.Tensor],
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        cfg: float,
        shift: float,
        seed: int,
        cfg_execution: str = "sequential_sage",
        unique_id: str | None = None,
    ) -> tuple[torch.Tensor]:
        if "first_frame_pixel" not in conditioning:
            raise ValueError(
                "LingBot TI2V Sampler requires conditioning from LingBot TI2V Plaintext + Image + Qwen Encode"
            )
        return super().sample(
            model=model,
            conditioning=conditioning,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            shift=shift,
            seed=seed,
            cfg_execution=cfg_execution,
            unique_id=unique_id,
        )


class LingBotFLFSampler(LingBotSampler):
    """Experimental sampler that hard-pins the first and last temporal latents."""

    @torch.no_grad()
    def sample(
        self,
        model: LingBotModelHandle,
        conditioning: dict[str, torch.Tensor],
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        cfg: float,
        shift: float,
        seed: int,
        cfg_execution: str = "sequential_sage",
        unique_id: str | None = None,
    ) -> tuple[torch.Tensor]:
        missing = [
            name
            for name in ("first_frame_pixel", "last_frame_pixel")
            if name not in conditioning
        ]
        if missing:
            raise ValueError(
                "LingBot Experimental FLF Sampler requires conditioning from the "
                "LingBot Experimental First + Last Frame Prompt node"
            )
        return super().sample(
            model=model,
            conditioning=conditioning,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            shift=shift,
            seed=seed,
            cfg_execution=cfg_execution,
            unique_id=unique_id,
        )


class LingBotVAEDecode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("LINGBOT_MODEL",),
                "latents": ("LINGBOT_LATENTS",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    @torch.no_grad()
    def decode(
        self,
        model: LingBotModelHandle,
        latents: torch.Tensor,
        unique_id: str | None = None,
    ) -> tuple[torch.Tensor]:
        _send_progress_text(unique_id, "Stage 4/5 · VAE Decode\nDecoding latent video into RGB frames")
        with model.lock:
            if _module_device(model.vae) != model.device:
                raise RuntimeError(
                    "LingBot VAE is on CPU at decode time; it must stay on CUDA for bf16 decode."
                )
            pipe = _ComfyLingBotPipeline(
                transformer=None,
                vae=model.vae,
                text_encoder=None,
                processor=None,
                scheduler=None,
            ).set_execution_device(model.device)
            try:
                videos = pipe._decode_latents(latents)
            finally:
                del pipe
                _empty_cuda_cache()
            if len(videos) != 1:
                raise ValueError(f"Expected one LingBot video, got batch size {len(videos)}")
            frames = torch.from_numpy(videos[0]).to(dtype=torch.float32).contiguous().clamp_(0.0, 1.0)
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"Expected decoded [T,H,W,3] frames, got {tuple(frames.shape)}")
            _send_progress_text(unique_id, f"Stage 4/5 · VAE Decode\nComplete · {frames.shape[0]} frames ready")
            return (frames,)


NODE_CLASS_MAPPINGS = {
    "LingBotModelLoader": LingBotModelLoader,
    "LingBotTextEncode": LingBotTextEncode,
    "LingBotPromptEncode": LingBotPromptEncode,
    "LingBotTI2VPromptEncode": LingBotTI2VPromptEncode,
    "LingBotFLFPromptEncode": LingBotFLFPromptEncode,
    "LingBotPromptPreview": LingBotPromptPreview,
    "LingBotPromptSettings": LingBotPromptSettings,
    "LingBotGenerationSettings": LingBotGenerationSettings,
    "LingBotPostProcessSettings": LingBotPostProcessSettings,
    "LingBotLazyImageSwitch": LingBotLazyImageSwitch,
    "LingBotLatentSink": LingBotLatentSink,
    "LingBotSampler": LingBotSampler,
    "LingBotTI2VSampler": LingBotTI2VSampler,
    "LingBotFLFSampler": LingBotFLFSampler,
    "LingBotVAEDecode": LingBotVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LingBotModelLoader": "LingBot Model Loader",
    "LingBotTextEncode": "LingBot Text Encode",
    "LingBotPromptEncode": "LingBot Plaintext Prompt + Qwen Encode",
    "LingBotTI2VPromptEncode": "LingBot TI2V Plaintext + Image + Qwen Encode",
    "LingBotFLFPromptEncode": "LingBot EXPERIMENTAL First + Last Frame Prompt",
    "LingBotPromptPreview": "LingBot Generated Prompt Preview",
    "LingBotPromptSettings": "LingBot Prompt Geometry + Duration",
    "LingBotGenerationSettings": "LingBot Generation Settings",
    "LingBotPostProcessSettings": "LingBot Post-Process Settings",
    "LingBotLazyImageSwitch": "LingBot Lazy Image Switch",
    "LingBotLatentSink": "LingBot Latent Benchmark Sink",
    "LingBotSampler": "LingBot Sampler",
    "LingBotTI2VSampler": "LingBot TI2V Sampler (First Frame Fixed)",
    "LingBotFLFSampler": "LingBot EXPERIMENTAL FLF Sampler (Both Endpoints Fixed)",
    "LingBotVAEDecode": "LingBot VAE Decode",
}
