"""Patched Triton kernels for upstream vLLM v0.20.0 TurboQuant.

Each module in this package replaces the launch wrapper for a specific
upstream kernel in `vllm.v1.attention.ops.triton_turboquant_*` while
keeping the kernel body verbatim. Monkey-patches are wired in
`turboquant.vllm_plugin:register()` under env-gated toggles.
"""
