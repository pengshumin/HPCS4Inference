"""验证 fused_qk_norm_rope_kvcache 的 QK-Norm + KV Cache 写入。

运行：
    python tests/test_qk_norm.py
"""
import sys
from pathlib import Path

# 未安装包时，允许直接运行 tests/、benchmarks/、examples/ 下的脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hpcs4inference import fused_qk_norm_rope_kvcache

sys.stdout.reconfigure(errors="replace")  # Windows GBK 控制台无法编码 emoji

torch.manual_seed(0)
device = "cuda"

B, S, H, KVH, D = 1, 4, 4, 2, 64

# ★ GQA 布局：(H + 2*KVH) * D
total_dim = (H + 2 * KVH) * D
qkv = torch.randn(B, S, total_dim, dtype=torch.float16, device=device)
q_out = torch.zeros(B, S, H * D, dtype=torch.float16, device=device)
k_cache = torch.zeros(B, KVH, S, D, dtype=torch.float16, device=device)
v_cache = torch.zeros(B, KVH, S, D, dtype=torch.float16, device=device)
q_weight = torch.randn(D, dtype=torch.float16, device=device)
k_weight = torch.randn(D, dtype=torch.float16, device=device)
rope_cos = torch.ones(S, D, dtype=torch.float16, device=device)
rope_sin = torch.zeros(S, D, dtype=torch.float16, device=device)

fused_qk_norm_rope_kvcache(
    qkv, q_out, k_cache, v_cache,
    q_weight, k_weight, rope_cos, rope_sin,
    H, KVH, D, 0, 1e-6
)
torch.cuda.synchronize()

# PyTorch 参考：按 GQA 布局拆分
qkv_f = qkv.float()
q_ref = qkv_f[:, :, :H*D].reshape(B, S, H, D)
k_ref = qkv_f[:, :, H*D:(H+KVH)*D].reshape(B, S, KVH, D)
v_ref = qkv_f[:, :, (H+KVH)*D:].reshape(B, S, KVH, D)

# Q RMSNorm（cos=1, sin=0 → RoPE 是恒等）
q_rms = torch.rsqrt(q_ref.pow(2).mean(-1, keepdim=True) + 1e-6)
q_expected = (q_ref * q_rms * q_weight.float()).half()
q_expected = q_expected.reshape(B, S, H * D)
diff_q = (q_out - q_expected).abs().max().item()
print(f"Q diff: {diff_q:.6f}")

# K RMSNorm + 写 k_cache
k_rms = torch.rsqrt(k_ref.pow(2).mean(-1, keepdim=True) + 1e-6)
k_expected = (k_ref * k_rms * k_weight.float()).half()
k_expected = k_expected.permute(0, 2, 1, 3)  # [B, KVH, S, D]
diff_k = (k_cache - k_expected).abs().max().item()
print(f"K cache diff: {diff_k:.6f}")

# V 直接 permute
v_expected = v_ref.half().permute(0, 2, 1, 3)  # [B, KVH, S, D]
diff_v = (v_cache - v_expected).abs().max().item()
print(f"V cache diff: {diff_v:.6f}")

print()
if max(diff_q, diff_k, diff_v) < 1e-2:
    print("✅ PASS — 所有 kernel 数值正确")
else:
    print("❌ FAIL — 还有索引或精度问题")
