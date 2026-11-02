"""用真实 RoPE 频率验证 fused_qk_norm_rope_kvcache。

运行：
    python tests/test_rope_real.py
"""
import sys
from pathlib import Path

# 未安装包时，允许直接运行 tests/、benchmarks/、examples/ 下的脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hpcs4inference import fused_qk_norm_rope_kvcache

torch.manual_seed(1)
device = "cuda"
B, S, H, KVH, D = 1, 4, 4, 2, 64
total_dim = (H + 2 * KVH) * D

qkv = torch.randn(B, S, total_dim, dtype=torch.float16, device=device)
q_out = torch.zeros(B, S, H * D, dtype=torch.float16, device=device)
k_cache = torch.zeros(B, KVH, S, D, dtype=torch.float16, device=device)
v_cache = torch.zeros(B, KVH, S, D, dtype=torch.float16, device=device)
q_weight = torch.randn(D, dtype=torch.float16, device=device)
k_weight = torch.randn(D, dtype=torch.float16, device=device)

# ★ 真实 RoPE：每个位置一组角度
inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device=device).float() / D))
pos = torch.arange(S, device=device).float().unsqueeze(1)
freqs = pos * inv_freq.unsqueeze(0)          # [S, D/2]
emb = torch.cat([freqs, freqs], dim=-1)      # [S, D]
rope_cos = emb.cos().half()
rope_sin = emb.sin().half()

fused_qk_norm_rope_kvcache(
    qkv, q_out, k_cache, v_cache,
    q_weight, k_weight, rope_cos, rope_sin,
    H, KVH, D, 0, 1e-6
)
torch.cuda.synchronize()

# PyTorch 参考
qkv_f = qkv.float()
q_ref = qkv_f[:, :, :H*D].reshape(B, S, H, D)
k_ref = qkv_f[:, :, H*D:(H+KVH)*D].reshape(B, S, KVH, D)

def apply_rope(x, cos, sin):
    x_e = x[..., 0::2]
    x_o = x[..., 1::2]
    c = cos[:, 0::2].unsqueeze(0).unsqueeze(2)   # [1, S, 1, D/2]
    s = sin[:, 0::2].unsqueeze(0).unsqueeze(2)
    out_e = x_e * c - x_o * s
    out_o = x_e * s + x_o * c
    out = torch.stack([out_e, out_o], dim=-1).flatten(-2)
    return out

q_rms = torch.rsqrt(q_ref.pow(2).mean(-1, keepdim=True) + 1e-6)
q_norm = q_ref * q_rms * q_weight.float()
q_rope = apply_rope(q_norm, rope_cos, rope_sin).half()
diff_q = (q_out.reshape(B, S, H, D) - q_rope).abs().max().item()
print(f"Q (with RoPE) diff: {diff_q:.6f}")

k_rms = torch.rsqrt(k_ref.pow(2).mean(-1, keepdim=True) + 1e-6)
k_norm = k_ref * k_rms * k_weight.float()
k_rope = apply_rope(k_norm, rope_cos, rope_sin).permute(0, 2, 1, 3).half()
diff_k = (k_cache - k_rope).abs().max().item()
print(f"K (with RoPE) diff: {diff_k:.6f}")
