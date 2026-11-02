# benchmark.py
import sys
from pathlib import Path

# 未安装包时，允许直接运行 tests/、benchmarks/、examples/ 下的脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hpcs4inference import fused_attention_decode, fused_qk_norm_rope_kvcache


def bench(fn, warmup=10, iters=100):
    """用 CUDA Event 计时，避免墙钟分辨率不足导致结果为 0。"""
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters  # ms

device = "cuda"

# ---- Attention Decode 基准 ----
print("=" * 60)
print("Attention Decode Benchmark")
print("=" * 60)
for S in [64, 256, 1024, 2048]:
    B, H, D = 1, 4, 64
    q = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    out = torch.zeros(B, H, 1, D, dtype=torch.float16, device=device)

    def custom():
        fused_attention_decode(q, k, v, out, S)

    def pytorch():
        scale = 1.0 / (D ** 0.5)
        s = torch.matmul(q, k.transpose(-2, -1)) * scale
        a = torch.softmax(s, dim=-1)
        return torch.matmul(a, v)

    t_c = bench(custom)
    t_p = bench(pytorch)
    print(f"S={S:5d}  custom={t_c:7.3f}ms  pytorch={t_p:7.3f}ms  ratio={t_p/t_c:.2f}x")

# ---- QK-Norm + RoPE + KV Cache 基准 ----
print()
print("=" * 60)
print("QK-Norm + RoPE + KV Cache Benchmark")
print("=" * 60)
for S in [64, 256, 1024]:
    B, H, KVH, D = 1, 4, 2, 64
    total = (H + 2 * KVH) * D
    qkv = torch.randn(B, S, total, dtype=torch.float16, device=device)
    q_out = torch.zeros(B, S, H * D, dtype=torch.float16, device=device)
    k_c = torch.zeros(B, KVH, S, D, dtype=torch.float16, device=device)
    v_c = torch.zeros(B, KVH, S, D, dtype=torch.float16, device=device)
    qw = torch.randn(D, dtype=torch.float16, device=device)
    kw = torch.randn(D, dtype=torch.float16, device=device)
    cos = torch.randn(S, D, dtype=torch.float16, device=device)
    sin = torch.randn(S, D, dtype=torch.float16, device=device)

    def custom():
        fused_qk_norm_rope_kvcache(
            qkv, q_out, k_c, v_c, qw, kw, cos, sin, H, KVH, D, 0, 1e-6)

    t_c = bench(custom)
    print(f"S={S:5d}  custom={t_c:7.3f}ms")
