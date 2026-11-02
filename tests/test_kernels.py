"""三个自定义 CUDA kernel 的基础冒烟测试。

运行：
    python tests/test_kernels.py
"""
import sys
from pathlib import Path

# 未安装包时，允许直接运行 tests/、benchmarks/、examples/ 下的脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hpcs4inference import (
    fused_attention_decode,
    fused_qk_norm_rope_kvcache,
    fused_silu_mul,
)

sys.stdout.reconfigure(errors="replace")  # Windows GBK 控制台无法编码 emoji

torch.manual_seed(42)
device = "cuda"

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================
# 测试 1: fused_silu_mul（最简单，先跑通）
# ============================================================
print("=" * 60)
print("Test 1: fused_silu_mul")
print("=" * 60)

B, I = 4, 128
gate_t   = torch.randn(B, I, dtype=torch.float16, device=device)
up_t     = torch.randn(B, I, dtype=torch.float16, device=device)
output_t = torch.zeros(B, I, dtype=torch.float16, device=device)

try:
    fused_silu_mul(gate_t, up_t, output_t)
    torch.cuda.synchronize()
    print("  Kernel 执行成功，无 CUDA error")
    print(f"  Output shape: {output_t.shape}, dtype: {output_t.dtype}")
    print(f"  Output stats: min={output_t.min().item():.4f}, max={output_t.max().item():.4f}")

    # PyTorch 参考实现
    silu_gate = gate_t * torch.sigmoid(gate_t)
    ref = silu_gate * up_t
    diff = (output_t - ref).abs().max().item()
    print(f"  Max diff vs PyTorch: {diff:.6f}")
    if diff < 1e-2:
        print("  ✅ PASS")
    else:
        print("  ⚠️  输出与预期不一致，需要检查 kernel 索引")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

print()

# ============================================================
# 测试 2: fused_attention_decode
# ============================================================
print("=" * 60)
print("Test 2: fused_attention_decode")
print("=" * 60)

B, H, S, D = 1, 4, 16, 64
q = torch.randn(B, H, 1, D, dtype=torch.float16, device=device)
k_cache = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
v_cache = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
output = torch.zeros(B, H, 1, D, dtype=torch.float16, device=device)

try:
    fused_attention_decode(q, k_cache, v_cache, output, S)
    torch.cuda.synchronize()
    print("  Kernel 执行成功，无 CUDA error")
    print(f"  Output shape: {output.shape}")
    print(f"  Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}")

    # PyTorch 参考实现
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q.float(), k_cache.float().transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    ref = torch.matmul(attn, v_cache.float()).half()

    diff = (output - ref).abs().max().item()
    print(f"  Max diff vs PyTorch: {diff:.6f}")
    if diff < 1e-2:
        print("  ✅ PASS")
    else:
        print("  ⚠️  输出与预期不一致，需要检查 kernel 索引")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

print()

# ============================================================
# 测试 3: fused_qk_norm_rope_kvcache
# ============================================================
print("=" * 60)
print("Test 3: fused_qk_norm_rope_kvcache")
print("=" * 60)

B, S, num_heads, num_kv_heads, head_dim = 1, 4, 4, 2, 64
total_dim = (num_heads + 2 * num_kv_heads) * head_dim  # Q + K + V 拼接

qkv = torch.randn(B, S, total_dim, dtype=torch.float16, device=device)
q_out = torch.zeros(B, S, num_heads * head_dim, dtype=torch.float16, device=device)
k_cache = torch.zeros(B, num_kv_heads, S, head_dim, dtype=torch.float16, device=device)
v_cache = torch.zeros(B, num_kv_heads, S, head_dim, dtype=torch.float16, device=device)
q_weight = torch.randn(head_dim, dtype=torch.float16, device=device)
k_weight = torch.randn(head_dim, dtype=torch.float16, device=device)
rope_cos = torch.randn(S, head_dim, dtype=torch.float16, device=device)
rope_sin = torch.randn(S, head_dim, dtype=torch.float16, device=device)

try:
    fused_qk_norm_rope_kvcache(
        qkv, q_out, k_cache, v_cache,
        q_weight, k_weight, rope_cos, rope_sin,
        num_heads, num_kv_heads, head_dim,
        position_offset=0, eps=1e-6
    )
    torch.cuda.synchronize()
    print("  Kernel 执行成功，无 CUDA error")
    print(f"  q_out shape: {q_out.shape}")
    print(f"  k_cache shape: {k_cache.shape}")
    print(f"  v_cache shape: {v_cache.shape}")
    print(f"  q_out stats: min={q_out.min().item():.4f}, max={q_out.max().item():.4f}")
    print(f"  k_cache non-zero: {(k_cache != 0).sum().item()} / {k_cache.numel()}")
    print(f"  v_cache non-zero: {(v_cache != 0).sum().item()} / {v_cache.numel()}")

    if (k_cache != 0).sum().item() > 0 and (v_cache != 0).sum().item() > 0:
        print("  ✅ KV Cache 有数据写入")
    else:
        print("  ⚠️  KV Cache 为空，可能索引有误")
except Exception as e:
    print(f"  ❌ FAIL: {e}")

print()
print("=" * 60)
print("测试完成")
print("=" * 60)
