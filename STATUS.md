# TurboQuant — current status (HEAD = `2d6b8f5`, plugin code = `16c6e93`)

Production deployment of TurboQuant as a vLLM attention backend, validated
end-to-end on A100-40GB.

## Build & Run

Build the image from repo:

```bash
git clone https://github.com/RRoundTable/turboquant.git
cd turboquant
docker build -t vllm-turboquant .
```

Bundles vLLM 0.19.0 + `docker/vllm_patches/` (carries
[vllm-project/vllm#39868](https://github.com/vllm-project/vllm/pull/39868)) +
the turboquant package (HYP-044 + HYP-045 patches applied). Run:

```bash
docker run --gpus all -p 8000:8000 vllm-turboquant \
  --model Qwen/Qwen3-8B --gpu-memory-utilization 0.85
```

To apply to a different vLLM image, use `Dockerfile.overlay` with
`--build-arg BASE_IMAGE=your-vllm-image:tag` (requires matching
`docker/vllm_patches/` snapshot for that vLLM version).

## Hypothesis-by-hypothesis progression (this engagement)

| HYP | Status | What it changed |
|---|---|---|
| 041 | rejected (baseline read) | First end-to-end serving sweep against vLLM (FA/FI). Surfaced TQ at 0.57–0.81× tok/s, 4 OOM configs, integration overhead. |
| 042 | rejected | First-pass torch.profiler attribution at output_len=2 (prefill-dominated; misled into "kernel double-count"). |
| 042b | confirmed | output_len=128 attribution: attention kernel = ~91 % of per-decode-step Δ. A100 ceiling identified. |
| 043 | rejected (inspection) | "Two attention kernels per layer" was a torch.profiler host-op/child double-exposure. No code change. |
| 044 | confirmed (kernel) / partial (e2e) | Chunk-cap split-K heuristic (cap chunk_size at 256, drop batch divisor). Microbench 0.79–0.84× kernel at batch≥8. End-to-end +6 % @ s8192×8, +18 % @ s1024×32. **Shipped**. |
| 045 | confirmed | Skip dead `k_quant`/`v_quant`/`k_norms`/`v_norms` allocation at `num_splits>1`. **Removed ~9.6 GB** of dead allocation at the worst case. **All 4 HYP-041 OOM configs run.** Peak RSS now ≤ baseline. **Shipped**. |
| 046 | pending (hardware-gated) | H100 / H200 re-measurement. Expected to lift the kernel ceiling via async ldmatrix + fp8e4nv compile. |
| 047 | confirmed (medium configs) | KV offload+reuse PoC: PCIe Gen4 ≈ 26 GB/s, restore < TQ prefill at every config ≤ 16384×8. **Sync restore wins through batch=8.** Async overlap (5 decode steps) hides transfer at all but the largest configs. |
| 047 ext | regime-bounded | At 32k+, TQ prefill is so cheap (PR #39868 cache compression) that re-prefill is competitive with restore. Value at 32k+ is *capacity*, not amortization. |
| 048 | not started | vLLM block-manager integration of offload+reuse (gated on user direction). |

## Headline numbers (Qwen3-8B fp16 eager, A100-40GB)

### Serving mode (`vllm bench serve` against deployed `vllm serve`)

| seq × conc | metric    |   FA |   FI |  TQ | TQ/FI |
|------------|-----------|-----:|-----:|----:|------:|
| 1024×8     | TTFT (ms) |  299 |  351 | **240** | **0.68×** |
| 8192×8     | TTFT (ms) | 1788 | 1544 | **1315** | **0.85×** |
| 1024×8     | TPOT (ms) |   23 |   22 |  30 | 1.34× |
| 2048×32    | TPOT (ms) |   52 |   50 |  62 | 1.26× |
| 8192×8     | TPOT (ms) |   49 |   48 |  57 | 1.19× |
| 8192×8     | out tok/s |  112 |  116 | **106** | **0.91×** |

(Full table: `results/v5_serve/REPORT.md`.)

### Memory: PR #39868 cache compression (per HYP-041 / HYP-045)

| | KV tokens (Qwen3-8B at 0.85 util on A100-40GB) | × baseline |
|---|---:|---:|
| baseline (fp16) | 126,416 | 1.00× |
| TurboQuant (with PR #39868) | **404,544** | **3.20×** |

Peak `nvidia-smi` GPU RSS at saturation:

| seq × b | TQ before HYP-045 | **TQ after HYP-045** | baseline FA |
|---|---:|---:|---:|
| 8192×8 | 38.5 GB | **34.9 GB** | 34.8 GB |
| 16384×8 | OOM | **34.9 GB** | 34.8 GB |
| 16384×32 | OOM | **34.9 GB** | 34.8 GB (FI OOMs here) |

### KV offload viability (HYP-047)

| seq × b | TQ KV size | restore (ms) | TQ prefill (ms) | restore/prefill |
|---|---:|---:|---:|---:|
| 1024×8  |  0.30 GB |  12 |  35 | **0.33×** |
| 8192×8  |  2.42 GB |  92 | 134 | **0.69×** |
| 16384×8 |  4.83 GB | 185 | 190 | **0.97×** |
| 32768×8 |  9.66 GB | 371 | 259 | 1.43× |

PCIe Gen4 effective: ~26 GB/s. fp16 cache would be 3.2× larger and lose at every config — TQ is the enabling factor.

## Known constraints (A100-40GB, this stack)

1. **`enforce_eager=True` required.** vLLM lowers fp8 path to `fp8e4nv` which A100 SM80 cannot torch.compile. CUDA graphs are off for both backends so the comparison is fair, but both pay launch-overhead tax. Lifts on H100 (HYP-046).
2. **A100 kernel ceiling.** Decode kernel hits the smem→mma stall ceiling per HYP-035 / HYP-037 / HYP-040. Per-layer ratio is 4.9× FlashInfer at seq=8192×b=8 (corrected from the initial 9.8× double-count). Floor on A100; H100 async ldmatrix should drop it.
3. **TPOT 1.19–1.34× slower than FI in serving.** The decode trade is real and architectural until H100.
4. **Shape: Qwen3-8B (head_dim=128).** Other head_dims are not currently in the kernel switch (`csrc/src/decode_v5_tc_binding.cu:779` enforces `padded_dim == 128`).

## Where TurboQuant wins on A100 today

- **TTFT at long context**: 0.74–0.85× FA/FI median TTFT at seq=8192. Driven by 4× smaller cache reads in chunked prefill + fewer scheduler preemptions.
- **Memory capacity**: 3.2× more KV tokens per byte. FI OOMs at 16384×32; TQ runs.
- **Cheap prefill**: TQ prefill is at-or-below baseline at every long-ctx config (HYP-045 split-metric).

## Where it loses

- **TPOT** at every config (1.19–1.34× slower in serving).
- **Decode tok/s in pure-decode benches**: 0.4–0.9× FA at long-ctx × large-batch (HYP-045 split-metric, decode-only column).

## Files / artifacts

- Plugin: `turboquant/vllm_plugin.py` (entry-point `vllm.general_plugins.turboquant`)
- Backend: `turboquant/vllm_backend_fused.py`
- CUDA kernels: `csrc/src/{decode_v5_tc_binding.cu,write_kernel.cu,quantize_write_binding.cu,decode_v4_binding.cu}`
- vLLM runtime patches: `docker/vllm_patches/` (PR #39868)
- Container: `Dockerfile`
- Hypothesis log: `docs/hypotheses/HYP-NNN-*.md`
- Latest reports:
  - `results/v5_serve/REPORT.md` — serving-mode comparison
  - `results/v5_split/REPORT.md` — offline prefill/decode split
  - `results/v5_vs_baseline_hyp045/REPORT.md` — HYP-045 sweep
  - `results/hyp047/REPORT.md` — KV offload feasibility
