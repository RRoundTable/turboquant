"""Minimal test for FlashInfer-style kernel: 1 token, 1 head, head_dim=64."""
import torch
import math
import sys
sys.path.insert(0, "/workspace/turboquant/tests")
from test_flashinfer_turbo import _build_module

module = _build_module()
print("Module loaded")

# head_dim=64, 1 kv head, 1 qo head, 1 token
k_q = torch.full((1, 1, 16, 32), 0x77, dtype=torch.uint8, device="cuda")
v_q = torch.full((1, 1, 16, 32), 0x88, dtype=torch.uint8, device="cuda")
k_n = torch.ones(1, 1, 16, 1, dtype=torch.float16, device="cuda")
v_n = torch.ones(1, 1, 16, 1, dtype=torch.float16, device="cuda")
q = torch.ones(1, 1, 64, dtype=torch.float16, device="cuda")

ki = torch.tensor([0], dtype=torch.int32, device="cuda")
kp = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
kl = torch.tensor([1], dtype=torch.int32, device="cuda")

out = module.decode(q, k_q.view(-1), v_q.view(-1),
    k_n.view(-1).view(torch.uint8).view(torch.float16),
    v_n.view(-1).view(torch.uint8).view(torch.float16),
    ki, kp, kl, 1, 1, 16, 64, 64, 1.0/math.sqrt(64))

print("Output shape:", out.shape)
print("Output[:5]:", out[0, 0, :5].float().tolist())

# Expected: index 8 centroid = 0.1284 * (1/sqrt(64)) = 0.01605
expected = 0.1284154 / math.sqrt(64)
print(f"Expected: {expected:.6f}")
print(f"Got: {out[0,0,0].float().item():.6f}")
print("PASS" if abs(out[0,0,0].float().item() - expected) < 0.01 else "FAIL")
