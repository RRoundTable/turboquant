# Option A (docker build .) verification

Bench of the image built from `Dockerfile` at repo HEAD (commit `8d7a99a`),
built on Forge via `forge image build --context . --name tq-option-a --tag v0`
(image id `827ea1df`, 8.9 GB).

## What works

- `docker build` completes cleanly (~23 min on Forge's builder).
- `vllm serve` starts; `/health` returns 200.
- Plugin registers via `vllm.general_plugins.turboquant`.
- Engine selects `AttentionBackendEnum.CUSTOM` and allocates 829,104 KV tokens
  on Qwen3-1.7B (PR #39868 page-size override fires).
- `/v1/completions` produces valid text.

## Per-config bench vs `results/v5_serve/tq-*` (same image config, earlier runs)

Qwen3-8B, A100-40GB, 64 prompts, output_len=128.

| config       | dur v5 | dur A | TTFT_med v5 | TTFT_med A | TPOT_med v5 | TPOT_med A | tok/s v5 | tok/s A | drift (tok/s) |
|--------------|-------:|------:|------------:|-----------:|------------:|-----------:|---------:|--------:|--------------:|
| s1024 × c8   |   41 s | 64 s  |      240 ms |     553 ms |      29.5 ms |      50.2 ms |    198   |    128  | −35 % |
| s2048 × c32  |   27 s | 68 s  |      821 ms |   20221 ms |      62.3 ms |      66.4 ms |    303   |    121  | −60 % |
| s8192 × c8   |   77 s | 76 s  |     1315 ms |    2084 ms |      56.8 ms |      46.9 ms |    106   |    108  | +3 % |

## Control: old image rerun at s2048 × c32 (today)

Running the same config on the old `tq-hyp029:pr` image reproduces v5_serve
within noise: duration 27.0 s, TTFT 813 ms, TPOT 60.5 ms, 303.6 tok/s.

So the drift is **real and image-specific**, not a Forge-load effect.

## Why, and what to do

Package version check between the two images (both on A100 SM80):

| package       | old image (`tq-hyp029:pr`) | new image (Option A, `tq-option-a:v0`) |
|---------------|----------------------------|-----------------------------------------|
| vllm          | 0.19.0                     | 0.19.0                                  |
| torch         | 2.10.0+cu128               | 2.10.0+cu128                            |
| flashinfer    | 0.6.6                      | 0.6.6                                   |
| transformers  | 4.57.6                     | 4.57.6                                  |
| numpy         | 2.2.6                      | 2.2.6                                   |
| triton        | 3.6.0                      | 3.6.0                                   |
| cuDNN         | 91002                      | 91002                                   |
| torch_extensions cache | absent | absent                              |

All identical. Drift isn't explained by any visible package diff. Hypothesis
not yet verified:

1. **Base-image differences below Python**: old uses Forge's `mlops-notebook`
   (whatever its CUDA runtime stack is); Option A uses `nvidia/cuda:12.6.3-devel`
   directly. Some low-level driver interaction or linked library could change
   how FA2 or the scheduler behaves at small-seq / high-concurrency.
2. **Some global-state difference** we haven't probed — e.g.,
   `VLLM_TORCH_COMPILE_CACHE_DIR` defaults, scheduler config autodetected
   differently per base.

At `s8192 × c8` (our headline long-ctx config in BENCHMARKS.md), the two
images match within 3%, so **users hitting the use-case our docs recommend
will see the published numbers**. Short-seq / high-concurrency workloads
may see lower throughput and should benchmark on their own hardware before
depending on specific serving SLAs.

## Reproducibility note for the README / BENCHMARKS.md

A user running `docker build .` from repo will get Option A's numbers, not
`v5_serve`'s. BENCHMARKS.md's numbers should be understood as a lower bound
on what you can get with this exact code, achieved with some Forge
base-image warmup we can't currently attribute to a specific package pin.
