"""Debug: check which chunk is correct in the bdz>1 merge."""
import torch, math
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/turboquant/tests")
from test_flashinfer_turbo import _build_module, setup_data

DEVICE = "cuda"

module = _build_module()
data = setup_data(1)  # 1 token

ki = torch.tensor([0], dtype=torch.int32, device=DEVICE)
kp = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
kl = torch.tensor([1], dtype=torch.int32, device=DEVICE)

fused = module.decode(
    data["RQ"].half().unsqueeze(0),
    data["k_q"].view(-1), data["v_q"].view(-1),
    data["k_n"].view(-1).view(torch.uint8).view(torch.float16),
    data["v_n"].view(-1).view(torch.uint8).view(torch.float16),
    ki, kp, kl, 16, 8, 16, 128, 128, 1.0/math.sqrt(128))

fused_unrot = data["inv_rotate"](fused.float().squeeze(0))
ref = data["ref"]

# Check per-chunk cosine
cos_chunk0 = F.cosine_similarity(ref[..., :64].flatten(), fused_unrot[..., :64].flatten(), dim=0)
cos_chunk1 = F.cosine_similarity(ref[..., 64:].flatten(), fused_unrot[..., 64:].flatten(), dim=0)
cos_full = F.cosine_similarity(ref.flatten(), fused_unrot.flatten(), dim=0)

print(f"Chunk 0 (dims 0-63):   cos = {cos_chunk0:.6f}")
print(f"Chunk 1 (dims 64-127): cos = {cos_chunk1:.6f}")
print(f"Full:                  cos = {cos_full:.6f}")

# Check rotated output (before un-rotation)
fused_rot = fused.float().squeeze(0)
# Ref rotated
RQ = data["RQ"]
pd = 128; s = 1.0/math.sqrt(pd)
C4 = torch.tensor([-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
     0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368], device=DEVICE)
c4 = C4 * s

# For 1 token, output = V_dequant (softmax weight = 1.0)
# So rotated output should = dequant(packed_V) * norm
print(f"\nRotated output dims 0,64:")
print(f"  Kernel dim 0:  {fused_rot[0,0,0].item():.6f}")
print(f"  Kernel dim 64: {fused_rot[0,0,64].item():.6f}")

# Dims 0-63 and 64-127 in rotated space
cos_rot0 = F.cosine_similarity(fused_rot[..., :64].flatten(), fused_rot[..., :64].flatten(), dim=0)
print(f"\nRotated chunk 0 norm: {fused_rot[..., :64].norm():.4f}")
print(f"Rotated chunk 1 norm: {fused_rot[..., 64:].norm():.4f}")

# Are they the same?
if torch.allclose(fused_rot[..., :64], fused_rot[..., 64:], atol=1e-3):
    print("BUG: chunk 0 == chunk 1 (copied instead of computed separately)")
