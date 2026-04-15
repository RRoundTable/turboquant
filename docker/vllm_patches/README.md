# vLLM patches

These Python files overlay the vLLM pip install to add the
`custom_page_size` seam needed for TurboQuant's 4-bit quantized KV
cache. The change is proposed upstream — see
[vllm-project/vllm PR branch `turboquant-custom-cache`](https://github.com/vllm-project/vllm/compare/main...turboquant-custom-cache).

Once the PR is merged, these patches can be deleted and the Dockerfile
can depend on the vLLM version that contains the change.

## What it adds

Three seams on `vllm/v1/...` — all no-op for existing backends:

1. `AttentionBackend.get_kv_cache_page_size(...)` optional classmethod,
   defaults to `None`.
2. `AttentionSpec.custom_page_size` optional dataclass field; when set,
   `real_page_size_bytes` returns it verbatim.
3. `GPUModelRunner._reshape_kv_cache_tensors` views a prefix of the
   raw buffer when the backend's shape packs fewer elements than
   `head_size * dtype_size` per entry.

## Files

| file | role |
|---|---|
| `v1/attention/backend.py` | Adds default `get_kv_cache_page_size` classmethod |
| `v1/kv_cache_interface.py` | Adds `custom_page_size` field + merge preservation |
| `v1/worker/gpu_model_runner.py` | Queries backend + sub-view reshape |
| `v1/worker/gpu/attn_utils.py` | Same query for the other `get_kv_cache_spec` path |
| `model_executor/layers/attention/attention.py` | Unchanged vs vLLM main (kept as a marker for rebase sanity) |

## Keeping these in sync with the upstream PR

See `~/workdir/vllm-upstream`, branch `turboquant-custom-cache`. After
rebasing the branch onto a new vLLM main, regenerate with:

```bash
cd ~/workdir/vllm-upstream
for f in v1/attention/backend.py v1/kv_cache_interface.py \
         v1/worker/gpu_model_runner.py v1/worker/gpu/attn_utils.py \
         model_executor/layers/attention/attention.py; do
  cp "vllm/$f" "~/workdir/turboquant/docker/vllm_patches/$f"
done
```
