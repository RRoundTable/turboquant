"""Example: TurboQuant KV cache quantization with vLLM.

Two approaches are shown:
  1. Using TurboQuantCache directly with HuggingFace models (works anywhere)
  2. Monkey-patching a vLLM LLM instance
"""

import torch

# ── Approach 1: HuggingFace transformers + TurboQuantCache ──────────────
#
# This works with any HuggingFace model and doesn't require vLLM.
# Use this for prototyping or when you want maximum control.


def huggingface_example():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant import TurboQuantCache

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Create quantized KV cache
    # 3-bit MSE quantization → ~5.3x compression vs fp16
    cache = TurboQuantCache(
        head_dim=128,       # Llama-3.1 uses 128-dim heads
        bit_width=3,        # 3 bits per coordinate
        mode="mse",         # MSE-optimized (Algorithm 1)
    )

    # For 2.5-bit with outlier handling (as in the paper):
    # cache = TurboQuantCache(
    #     head_dim=128,
    #     bit_width=2,
    #     num_outlier_channels=32,   # 32 outlier channels at 3 bits
    #     outlier_bit_width=3,       # remaining 96 channels at 2 bits
    #     mode="mse",                # effective: (32*3 + 96*2)/128 = 2.5 bits
    # )

    inputs = tokenizer("The key insight of TurboQuant is", return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            past_key_values=cache,
            use_cache=True,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(f"\nCompression ratio: {cache.compression_ratio():.1f}x")
    print(f"Effective bit-width: {cache.effective_bit_width():.1f} bits")


# ── Approach 2: vLLM monkey-patching ────────────────────────────────────
#
# This patches the vLLM model in-place so that KV cache entries are
# quantized/dequantized transparently during inference.


def vllm_example():
    from vllm import LLM, SamplingParams
    from turboquant.vllm_integration import apply_turboquant_to_vllm

    llm = LLM(
        "meta-llama/Llama-3.1-8B-Instruct",
        gpu_memory_utilization=0.90,
    )

    # Apply TurboQuant — patches attention layers in-place
    apply_turboquant_to_vllm(llm, bit_width=3)

    sampling = SamplingParams(temperature=0.7, max_tokens=100)
    outputs = llm.generate(
        ["The key insight of TurboQuant is"],
        sampling,
    )

    for output in outputs:
        print(output.outputs[0].text)


# ── Approach 3: Standalone quantizer for custom pipelines ───────────────


def standalone_example():
    from turboquant import TurboQuantMSE, TurboQuantProd

    head_dim = 128
    batch, num_heads, seq_len = 1, 32, 512

    # Simulate KV cache vectors
    keys = torch.randn(batch, num_heads, seq_len, head_dim, device="cpu")
    values = torch.randn(batch, num_heads, seq_len, head_dim, device="cpu")

    # MSE quantizer (Algorithm 1) — 3 bits
    q_mse = TurboQuantMSE(head_dim, bit_width=3, device="cpu")
    k_idx, k_norms = q_mse.quantize(keys)
    keys_hat = q_mse.dequantize(k_idx, k_norms)

    mse = (keys - keys_hat).pow(2).mean().item()
    print(f"MSE quantizer (3-bit): MSE = {mse:.6f}")

    # Inner-product quantizer (Algorithm 2) — 3 bits
    q_prod = TurboQuantProd(head_dim, bit_width=3, device="cpu")
    idx, norms, signs, rnorms = q_prod.quantize(keys)
    keys_hat_prod = q_prod.dequantize(idx, norms, signs, rnorms)

    # Check unbiasedness: E[<y, x_hat>] ≈ <y, x>
    queries = torch.randn_like(keys)
    true_ip = (queries * keys).sum(dim=-1).mean()
    est_ip = (queries * keys_hat_prod).sum(dim=-1).mean()
    print(f"IP quantizer (3-bit): true IP = {true_ip:.6f}, estimated = {est_ip:.6f}")

    # Memory comparison
    orig_bytes = keys.nelement() * 2  # fp16
    quant_bytes = k_idx.nelement() * 1 + k_norms.nelement() * 4  # uint8 + float32
    print(f"Compression: {orig_bytes} → {quant_bytes} bytes ({orig_bytes/quant_bytes:.1f}x)")


if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant Standalone Example")
    print("=" * 60)
    standalone_example()
