"""
模拟 Qwen3-0.6B 的 decode 阶段，对比自定义 kernel 与 PyTorch 原生的性能。

Qwen3-0.6B 参数（从 config.json 读取）：
  - hidden_size: 1024
  - num_attention_heads: 16
  - num_key_value_heads: 8
  - head_dim: 128
  - num_hidden_layers: 28

为了准确模拟，我们：
  1. 用 Qwen3 的真实维度构造 K/V cache
  2. 生成 128 个 token，每个 token 都调用一次 attention decode
  3. 累加时间，对比自定义 kernel 和 PyTorch 原生
"""

import sys
from pathlib import Path

# 未安装包时，允许直接运行 tests/、benchmarks/、examples/ 下的脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch

from hpcs4inference import fused_attention_decode

sys.stdout.reconfigure(errors="replace")  # Windows GBK 控制台无法编码 emoji

# ============================================================
# Qwen3-0.6B 配置
# ============================================================
B = 1
NUM_LAYERS = 28
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
SEQ_LEN = 512          # 模拟的上下文长度
GEN_TOKENS = 128       # 要生成多少个 token

device = "cuda"

print("=" * 70)
print(f"Qwen3-0.6B Decode 模拟")
print(f"  层数: {NUM_LAYERS}, Q heads: {NUM_HEADS}, KV heads: {NUM_KV_HEADS}")
print(f"  head_dim: {HEAD_DIM}, 上下文: {SEQ_LEN}, 生成: {GEN_TOKENS} tokens")
print("=" * 70)
print()


# ============================================================
# 构造 K/V Cache（模拟真实推理的中间状态）
# ============================================================
def make_kv_cache(seq_len):
    """构造 [B, NUM_HEADS, S, D] 的 K/V cache"""
    k = torch.randn(B, NUM_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(B, NUM_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=device)
    return k, v


# ============================================================
# PyTorch 原生 attention decode
# ============================================================
def pytorch_attn_decode(q, k_cache, v_cache, seq_len):
    """
    q: [B, H, 1, D]
    k_cache: [B, H, S, D]
    v_cache: [B, H, S, D]
    """
    scale = 1.0 / (HEAD_DIM ** 0.5)
    scores = torch.matmul(q, k_cache.transpose(-2, -1)) * scale   # [B, H, 1, S]
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_cache)                              # [B, H, 1, D]
    return out


# ============================================================
# 自定义 kernel 的 attention decode
# ============================================================
def custom_attn_decode(q, k_cache, v_cache, seq_len):
    out = torch.zeros(B, NUM_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=device)
    fused_attention_decode(q, k_cache, v_cache, out, seq_len)
    return out


# ============================================================
# 正确性检查（确保两者输出一致）
# ============================================================
print("Step 1: 正确性检查")
q_test = torch.randn(B, NUM_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=device) * 0.1
k_test, v_test = make_kv_cache(SEQ_LEN)

out_pytorch = pytorch_attn_decode(q_test, k_test, v_test, SEQ_LEN)
out_custom = custom_attn_decode(q_test, k_test, v_test, SEQ_LEN)
diff = (out_pytorch - out_custom).abs().max().item()
print(f"  Max diff: {diff:.6f}")
print(f"  {'✅ 一致' if diff < 1e-2 else '❌ 不一致'}")
print()


# ============================================================
# Warmup
# ============================================================
print("Step 2: Warmup")
q = torch.randn(B, NUM_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=device) * 0.1
k_cache, v_cache = make_kv_cache(SEQ_LEN)

for _ in range(10):
    pytorch_attn_decode(q, k_cache, v_cache, SEQ_LEN)
    custom_attn_decode(q, k_cache, v_cache, SEQ_LEN)
torch.cuda.synchronize()
print("  Warmup 完成")
print()


# ============================================================
# 模拟逐 token 生成
# ============================================================
def simulate(model_type="pytorch"):
    """
    模拟生成 GEN_TOKENS 个 token 的过程。
    每个 token 都调用一次 attention decode（28 层累加）。
    实际推理中，每个 token 都要跑完整个模型的 28 层。
    """
    q = torch.randn(B, NUM_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=device) * 0.1
    k_cache, v_cache = make_kv_cache(SEQ_LEN)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(GEN_TOKENS):
        for _ in range(NUM_LAYERS):
            if model_type == "pytorch":
                pytorch_attn_decode(q, k_cache, v_cache, SEQ_LEN)
            else:
                custom_attn_decode(q, k_cache, v_cache, SEQ_LEN)

    torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed


print(f"Step 3: 模拟生成 {GEN_TOKENS} 个 token（每 token 跑 {NUM_LAYERS} 层）")
print("-" * 70)

t_pytorch = simulate("pytorch")
t_custom = simulate("custom")

total_ops = GEN_TOKENS * NUM_LAYERS
speedup = t_pytorch / t_custom

print(f"  PyTorch 原生:  {t_pytorch:.3f}s  ({t_pytorch/GEN_TOKENS*1000:.1f} ms/token, "
      f"{GEN_TOKENS/t_pytorch:.1f} tokens/s)")
print(f"  自定义 kernel: {t_custom:.3f}s  ({t_custom/GEN_TOKENS*1000:.1f} ms/token, "
      f"{GEN_TOKENS/t_custom:.1f} tokens/s)")
print(f"  加速比:        {speedup:.2f}x")
print()


# ============================================================
# 分解到单次 attention 调用
# ============================================================
print(f"Step 4: 单次 attention decode 延迟（{total_ops} 次调用的平均值）")
print("-" * 70)

# 精确测量单次延迟（用 CUDA Event）
def measure_single(model_type, iters=1000):
    q = torch.randn(B, NUM_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=device) * 0.1
    k_cache, v_cache = make_kv_cache(SEQ_LEN)

    # Warmup
    for _ in range(50):
        if model_type == "pytorch":
            pytorch_attn_decode(q, k_cache, v_cache, SEQ_LEN)
        else:
            custom_attn_decode(q, k_cache, v_cache, SEQ_LEN)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        if model_type == "pytorch":
            pytorch_attn_decode(q, k_cache, v_cache, SEQ_LEN)
        else:
            custom_attn_decode(q, k_cache, v_cache, SEQ_LEN)
    stop.record()
    torch.cuda.synchronize()

    return start.elapsed_time(stop) / iters * 1000  # us


t_single_pytorch = measure_single("pytorch")
t_single_custom = measure_single("custom")

print(f"  PyTorch 原生:  {t_single_pytorch:.2f} us")
print(f"  自定义 kernel: {t_single_custom:.2f} us")
print(f"  单次加速比:    {t_single_pytorch/t_single_custom:.2f}x")
print()


# ============================================================
# 换算成实际推理的收益
# ============================================================
print("=" * 70)
print("结论")
print("=" * 70)

# 每个 token 的 attention 总耗时
per_token_pytorch_ms = t_single_pytorch * NUM_LAYERS / 1000
per_token_custom_ms = t_single_custom * NUM_LAYERS / 1000

print(f"每生成一个 token，attention 部分耗时：")
print(f"  PyTorch 原生:  {per_token_pytorch_ms:.2f} ms")
print(f"  自定义 kernel: {per_token_custom_ms:.2f} ms")
print(f"  每个 token 节省: {per_token_pytorch_ms - per_token_custom_ms:.2f} ms")
print()
print(f"生成 {GEN_TOKENS} 个 token：")
print(f"  PyTorch 原生:  {t_pytorch*1000:.0f} ms")
print(f"  自定义 kernel: {t_custom*1000:.0f} ms")
print(f"  总共节省:      {(t_pytorch - t_custom)*1000:.0f} ms  ({(t_pytorch-t_custom)*1000:.0f} ms)")
print()
print(f"⚠️ 注意：这只是 attention 部分的加速。真实推理还要加上")
print(f"   MLP、LayerNorm、LM head 等其他层的耗时。")
print(f"   在 Qwen3-0.6B 上，attention 大约占整个 decode 的 20-30%，")
print(f"   所以端到端的实际加速比会小于这里的 {speedup:.1f}x。")
