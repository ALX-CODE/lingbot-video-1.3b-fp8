# Benchmarks

All numbers below are local measurements on one RTX 5080, not universal
performance claims. No result in this file establishes performance for the RTX
50-series as a whole.

## Matched sampler smoke test

Hardware and runtime:

- NVIDIA GeForce RTX 5080, 16 GB VRAM
- Windows / WDDM
- Python 3.11.9
- PyTorch 2.12.1+cu130
- CUDA 13.0 runtime
- Diffusers 0.38.0
- Transformers 5.3.0
- SageAttention 2.2.0

Workload: cached text conditioning, 256×144, 5 frames, 1 denoise step, CFG 1.
This deliberately tiny test isolates sampler overhead; it is not representative
of a normal 28- or 40-step video and should not be used to estimate end-to-end
generation time.

| Transformer | Sampler time | Relative |
| --- | ---: | ---: |
| BF16 official dense | 4.651 s | 1.00× |
| Selective fused FP8 | 3.651 s | **1.27×** |

That is about 21.5% less sampler time in this deliberately small matched test.
End-to-end speed varies with resolution, frame count, prompt encoding, VAE
decode, offloading, and post-processing.

## Projection microbenchmarks

| Fused projection | FP8 speedup |
| --- | ---: |
| Attention Q/K/V | 1.29× |
| MLP gate/up | 1.62× |

## Quantization shape

- 120 fused-path weights stored as E4M3 FP8
- 905,969,664 FP8 parameters
- 451,201,720 retained floating-point parameters
- Representative weight relative RMSE: 0.026491
- Checkpoint size: 1,886,009,888 bytes for the safetensors file

Attention output, MLP down projection, conditioning, normalization, input, and
output paths stay BF16. This selective layout was chosen because the measured
FP8 speedup was concentrated in QKV and gate/up projections.

Visual output was successfully validated on the test machine. Exact visual
equivalence to BF16 is not claimed; use the official BF16 transformer when
reference precision matters more than speed.

## Reproducibility limitations

The original measurement did not publish a standalone benchmark runner, raw
timing JSON, or paired BF16/FP8 media. Until those artifacts and a full-workload
comparison are published, quote this result only as a local one-step smoke test.
