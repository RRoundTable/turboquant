"""Minimal test: 1 token, 1 head, head_dim=128 with real quantized data."""
import torch, math
import torch.nn.functional as F
from turboquant.decode_kernel import _get_module

hd = 128; pd = 128; s = 1.0 / math.sqrt(pd)
C4 = torch.tensor([-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
     0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368], device="cuda")
B4 = torch.tensor([-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
     0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919], device="cuda")
c4 = C4 * s; b4 = B4 * s

gen = torch.Generator(device="cpu"); gen.manual_seed(42)
signs = torch.sign(torch.randn(pd, generator=gen)); signs[signs == 0] = 1.0; signs = signs.to("cuda")

def fwht(x):
    d = x.shape[-1]; x = x.clone(); shape = x.shape; h = 1
    while h < d:
        x = x.view(*shape[:-1], d // (2 * h), 2, h)
        a = x[..., 0, :].clone(); b = x[..., 1, :].clone()
        x[..., 0, :] = a + b; x[..., 1, :] = a - b
        x = x.view(shape); h *= 2
    return x * (1.0 / math.sqrt(d))

def rotate(x): return fwht(x * signs)
def inv_rotate(y): return fwht(y) * signs
def pack_4bit(idx): return ((idx[..., 0::2] << 4) | (idx[..., 1::2] & 0x0F)).to(torch.uint8)

torch.manual_seed(42)
K = torch.randn(1, 128, device="cuda")
V = torch.randn(1, 128, device="cuda")
Q = torch.randn(1, 128, device="cuda")

# Quantize K and V
norm_K = K.norm(dim=-1, keepdim=True).clamp(min=1e-8)
RK = rotate(K / norm_K)
idx_K = torch.bucketize(RK, b4).to(torch.uint8)
packed_K = pack_4bit(idx_K)

norm_V = V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
RV = rotate(V / norm_V)
idx_V = torch.bucketize(RV, b4).to(torch.uint8)
packed_V = pack_4bit(idx_V)

# Store HND: [1 page, 1 head, 16 entries, 64 bytes]
k_q = torch.zeros(1, 1, 16, 64, dtype=torch.uint8, device="cuda")
k_q[0, 0, 0, :32] = packed_K[0, :32]
k_q[0, 0, 0, 32:] = packed_K[0, 32:]
v_q = torch.zeros(1, 1, 16, 64, dtype=torch.uint8, device="cuda")
v_q[0, 0, 0, :32] = packed_V[0, :32]
v_q[0, 0, 0, 32:] = packed_V[0, 32:]
k_n = torch.zeros(1, 1, 16, 2, dtype=torch.float16, device="cuda")
k_n[0, 0, 0, :] = norm_K.squeeze().half()
v_n = torch.zeros(1, 1, 16, 2, dtype=torch.float16, device="cuda")
v_n[0, 0, 0, :] = norm_V.squeeze().half()

RQ = rotate(Q)
sm = 1.0 / math.sqrt(128)

# Python reference
deq_RK = c4[idx_K.long()] * norm_K.half().float()
deq_RV = c4[idx_V.long()] * norm_V.half().float()
score = (RQ @ deq_RK.t()) * sm
w = torch.softmax(score, dim=-1)
py_out = inv_rotate(w @ deq_RV)
print("Py ref[:5]:", py_out[0, :5].tolist())

# CUDA kernel
module = _get_module()
ki = torch.tensor([0], dtype=torch.int32, device="cuda")
kp = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
kl = torch.tensor([1], dtype=torch.int32, device="cuda")

fused = module.decode_attention(
    RQ.half().unsqueeze(0).unsqueeze(0),
    k_q.view(-1), v_q.view(-1),
    k_n.view(-1).view(torch.uint8).view(torch.float16),
    v_n.view(-1).view(torch.uint8).view(torch.float16),
    ki, kp, kl, 1, 1, 16, 128, 128, sm)

fused_rot = fused.float().squeeze()
fused_unrot = inv_rotate(fused_rot)
print("Kernel unrot[:5]:", fused_unrot[:5].tolist())
print("Kernel rot[:5]:", fused_rot[:5].tolist())
cos = F.cosine_similarity(py_out.flatten(), fused_unrot.flatten(), dim=0)
print(f"Cosine (unrotated): {cos.item():.6f}")

# Also compare in rotated space directly
py_rot_out = w @ deq_RV
print("Py rot out[:5]:", py_rot_out[0, :5].tolist())
cos_rot = F.cosine_similarity(py_rot_out.flatten(), fused_rot.flatten(), dim=0)
print(f"Cosine (rotated): {cos_rot.item():.6f}")

# Check: with 1 token, softmax = [1.0], so output = V_dequantized
# Kernel output should equal the dequanted V directly
print("\nV dequant rotated[:5]:", deq_RV[0, :5].tolist())
print("Kernel rotated[:5]:", fused_rot[:5].tolist())
print("Match dim0:", abs(deq_RV[0, 0].item() - fused_rot[0].item()) < 0.01)

print("\nDim 64 check:")
print("V dequant rot[64]:", deq_RV[0, 64].item())
print("Kernel rot[64]:", fused_rot[64].item())
print("Match dim64:", abs(deq_RV[0, 64].item() - fused_rot[64].item()) < 0.01)

# Check all chunk boundaries
for d in [0, 32, 63, 64, 96, 127]:
    py_val = deq_RV[0, d].item()
    kern_val = fused_rot[d].item()
    print(f"  dim {d}: py={py_val:.6f} kern={kern_val:.6f} diff={abs(py_val-kern_val):.6f}")

print("\nPASS" if cos.item() > 0.99 else "FAIL")
