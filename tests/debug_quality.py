"""Deep analysis: why 3.5-bit mixed degrades more than expected."""

import torch, math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

inputs = tokenizer("The capital of France is Paris and the capital of Germany is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs, use_cache=True)

kv = out.past_key_values

# Codebooks
C4 = torch.tensor([-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
     0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368], device="cuda")
B4 = torch.tensor([-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
     0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919], device="cuda")
C3 = torch.tensor([-2.1519775,-1.3439709,-0.7560052,-0.2451210, 0.2451210, 0.7560052, 1.3439709, 2.1519775], device="cuda")
B3 = torch.tensor([-1.7479742,-1.0499881,-0.5005631, 0.0, 0.5005631, 1.0499881, 1.7479742], device="cuda")

pd = 128
scale = 1.0 / math.sqrt(pd)
b4 = B4 * scale; c4 = C4 * scale
b3 = B3 * scale; c3 = C3 * scale

gen = torch.Generator(device="cpu"); gen.manual_seed(42)
signs = torch.sign(torch.randn(pd, generator=gen))
signs[signs == 0] = 1.0
signs = signs.to("cuda")

def fwht(x):
    d = x.shape[-1]; x = x.clone(); shape = x.shape; h = 1
    while h < d:
        x = x.view(*shape[:-1], d//(2*h), 2, h)
        a = x[...,0,:].clone(); b = x[...,1,:].clone()
        x[...,0,:] = a+b; x[...,1,:] = a-b
        x = x.view(shape); h *= 2
    return x * (1.0/math.sqrt(d))

def quant_deq(x, mode="4bit"):
    xf = x.float()
    norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normed = xf / norms
    norms = norms.to(torch.float16).float()
    rotated = fwht(normed * signs)
    result = torch.zeros_like(rotated)
    for chunk in range(pd // 64):
        ds = chunk * 64
        cd = rotated[..., ds:ds+64]
        if mode == "4bit":
            idx = torch.bucketize(cd.contiguous(), b4)
            result[..., ds:ds+64] = c4[idx]
        elif mode == "3.5bit":
            hi_idx = torch.bucketize(cd[..., :32].contiguous(), b4)
            lo_idx = torch.bucketize(cd[..., 32:].contiguous(), b3)
            result[..., ds:ds+32] = c4[hi_idx]
            result[..., ds+32:ds+64] = c3[lo_idx]
        elif mode == "3bit":
            idx = torch.bucketize(cd.contiguous(), b3)
            result[..., ds:ds+64] = c3[idx]
    inv = fwht(result) * signs
    return (inv * norms).to(x.dtype)

print("=== Per-Layer KV Cosine Similarity ===")
print(f"{'Layer':>6} {'4-bit cos':>12} {'3.5-bit cos':>12} {'3-bit cos':>12}")
print("-" * 48)

for layer_idx in [0, 7, 14, 21, 27]:
    k = kv[layer_idx][0].float()
    cos4 = F.cosine_similarity(k.flatten(0,2), quant_deq(k, "4bit").float().flatten(0,2), dim=-1).mean().item()
    cos35 = F.cosine_similarity(k.flatten(0,2), quant_deq(k, "3.5bit").float().flatten(0,2), dim=-1).mean().item()
    cos3 = F.cosine_similarity(k.flatten(0,2), quant_deq(k, "3bit").float().flatten(0,2), dim=-1).mean().item()
    print(f"{layer_idx:>6} {cos4:>12.6f} {cos35:>12.6f} {cos3:>12.6f}")

print()
print("=== Rotated Distribution (Layer 0) ===")
k0 = kv[0][0].float()
norms = k0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
rotated = fwht(k0 / norms * signs)
print(f"mean={rotated.mean():.6f}  std={rotated.std():.6f}  expected_std={1.0/math.sqrt(128):.6f}")
print(f"range: [{rotated.min():.4f}, {rotated.max():.4f}]")
print(f"3-bit outer boundary: +/-{1.7479742 * scale:.4f}")
print(f"4-bit outer boundary: +/-{2.4009919 * scale:.4f}")
print(f"% values clipped by 3-bit: {(rotated.abs() > 1.7479742 * scale).float().mean()*100:.1f}%")
print(f"% values clipped by 4-bit: {(rotated.abs() > 2.4009919 * scale).float().mean()*100:.1f}%")

print()
print("=== Attention Score Error Analysis ===")
k0_orig = kv[0][0].float()
q0 = k0_orig[:,:,-1:,:]  # use last K as query

for mode in ["4bit", "3.5bit", "3bit"]:
    k_hat = quant_deq(k0_orig, mode).float()
    scores_orig = (q0 @ k0_orig.transpose(-2,-1)) / math.sqrt(128)
    scores_quant = (q0 @ k_hat.transpose(-2,-1)) / math.sqrt(128)

    w_orig = torch.softmax(scores_orig, dim=-1)
    w_quant = torch.softmax(scores_quant, dim=-1)

    kl = (w_orig * (w_orig.clamp(min=1e-10).log() - w_quant.clamp(min=1e-10).log())).sum(-1).mean().item()
    tv = (w_orig - w_quant).abs().sum(-1).mean().item() / 2
    score_err = (scores_orig - scores_quant).abs().mean().item()

    print(f"{mode:>8}: score_err={score_err:.4f}  KL={kl:.6f}  TV_distance={tv:.4f}")

print()
print("=== Cumulative Error Across Layers ===")
print("Simulating how errors compound through 28 layers")
k_cos_products = {"4bit": 1.0, "3.5bit": 1.0, "3bit": 1.0}
for layer_idx in range(28):
    k = kv[layer_idx][0].float()
    v = kv[layer_idx][1].float()
    for mode in ["4bit", "3.5bit", "3bit"]:
        k_hat = quant_deq(k, mode).float()
        cos = F.cosine_similarity(k.flatten(0,2), k_hat.flatten(0,2), dim=-1).mean().item()
        k_cos_products[mode] *= cos

for mode, prod in k_cos_products.items():
    print(f"{mode:>8}: cumulative cos product = {prod:.6f}")
