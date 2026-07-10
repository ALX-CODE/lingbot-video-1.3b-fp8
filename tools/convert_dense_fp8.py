"""Create the speed/quality balanced FP8 checkpoint for LingBot-Video 1.3B.

Q/K/V and MLP gate/up projections use E4M3 weights with a shared tensor-wise
scale per fused group.  The runtime quantizes activations dynamically and uses
native ``torch._scaled_mm`` W8A8 Tensor Core execution.  Attention output,
MLP down projection, conditioning, normalization, and model I/O remain BF16.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the complete official LingBot-Video Dense 1.3B model directory",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def quantize_group(
    tensors: dict[str, torch.Tensor], keys: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    scale = max(float(tensors[key].float().abs().amax()) for key in keys) / FP8_MAX
    scale_tensor = torch.tensor([max(scale, 1e-12)], dtype=torch.float32)
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        output[key] = (
            (tensors[key].float() / scale_tensor)
            .clamp(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        output[f"{key}_scale"] = scale_tensor.clone()
    return output


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    source = model_dir / "transformer"
    destination = model_dir / "transformer_fp8_dense"
    temporary = model_dir / "transformer_fp8_dense.incomplete"
    if destination.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {destination}; pass --force to replace it")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    with (source / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config.get("num_experts", 0)) != 0:
        raise ValueError("This converter accepts only the dense LingBot model")

    weights_path = source / "diffusion_pytorch_model.safetensors"
    print(f"Loading {weights_path}", flush=True)
    tensors = load_file(weights_path, device="cpu")
    output = dict(tensors)
    quantized_keys: set[str] = set()
    for index in range(int(config["depth"])):
        groups = (
            tuple(f"blocks.{index}.attn.to_{name}.weight" for name in ("q", "k", "v")),
            tuple(f"blocks.{index}.ffn.{name}_proj.weight" for name in ("gate", "up")),
        )
        for keys in groups:
            output.update(quantize_group(tensors, keys))
            quantized_keys.update(keys)

    config["dense_fp8"] = "e4m3fn_tensorwise_fused"
    with (temporary / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    output_path = temporary / weights_path.name
    save_file(output, output_path)

    with safe_open(output_path, framework="pt", device="cpu") as handle:
        saved_keys = set(handle.keys())
        fp8_count = sum(
            handle.get_slice(key).get_dtype() == "F8_E4M3" for key in quantized_keys
        )
    expected_scales = {f"{key}_scale" for key in quantized_keys}
    if len(quantized_keys) != 120 or fp8_count != 120 or not expected_scales.issubset(saved_keys):
        raise RuntimeError(
            f"Validation failed: keys={len(quantized_keys)}, fp8={fp8_count}, "
            f"missing_scales={len(expected_scales - saved_keys)}"
        )

    del tensors, output
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)
    size_gb = sum(path.stat().st_size for path in destination.iterdir()) / 1e9
    print(f"Done: 120 fused-path weights converted; output={destination}; size={size_gb:.2f} GB")


if __name__ == "__main__":
    main()
