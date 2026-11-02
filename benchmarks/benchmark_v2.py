# benchmark_v2.py
import sys
from pathlib import Path

# 未安装包时，允许直接运行 tests/、benchmarks/、examples/ 下的脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hpcs4inference import fused_attention_decode


def bench_cuda(fn, warmup=20, iters=200):
    """用 CUDA Event 精确计时，只测 GPU 时间"""
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

# ---- Attention Decode ----
print("Attention Decode (CUDA Event)")
for S in [64, 256, 512, 1024, 2048]:
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

    t_c = bench_cuda(custom)
    t_p = bench_cuda(pytorch)
    print(f"S={S:5d}  custom={t_c:7.3f}ms  pytorch={t_p:7.3f}ms  ratio={t_p/t_c:.2f}x")

'''
x64 Native Tools Command Prompt for VS 2019

cd D:\cppproject\HPCS4Inference

conda activate env3.10

ncu --set full -k "fused_attention_decode_split_kernel" --launch-count 1 --target-processes all python benchmarks/benchmark_v2.py
'''
