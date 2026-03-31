"""Test FlashInfer-style kernel: head_dim=128, multi-head, GQA."""
import torch, math
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/turboquant/tests")
from test_flashinfer_turbo import _build_module, setup_data, pack_4bit, unpack_4bit, C4, B4, _next_pow2

DEVICE = "cuda"
NUM_KV_HEADS = 8
NUM_QO_HEADS = 16
PAGE_SIZE = 16
HEAD_DIM = 128

module = _build_module()
print("Module loaded")

for seq_len in [1, 4, 16]:
    data = setup_data(seq_len)
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    ki = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    kp = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    kl = torch.tensor([seq_len - (num_pages - 1) * PAGE_SIZE], dtype=torch.int32, device=DEVICE)

    fused = module.decode(
        data["RQ"].half().unsqueeze(0),
        data["k_q"].view(-1), data["v_q"].view(-1),
        data["k_n"].view(-1).view(torch.uint8).view(torch.float16),
        data["v_n"].view(-1).view(torch.uint8).view(torch.float16),
        ki, kp, kl,
        NUM_QO_HEADS, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, 128,
        1.0 / math.sqrt(HEAD_DIM))

    fused_unrot = data["inv_rotate"](fused.float().squeeze(0))

    cos = F.cosine_similarity(data["ref"].flatten(), fused_unrot.flatten(), dim=0).item()
    status = "PASS" if cos > 0.99 else "FAIL"
    print(f"  seq_len={seq_len:>4}  cos={cos:.6f}  {status}")
