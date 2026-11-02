"""
Qwen3-0.6B 连续对话推理系统
用自定义 CUDA kernel 加速 attention decode 阶段

运行：
    python examples/chat.py              # 使用自定义 kernel
    python examples/chat.py --no-kernel  # 使用 PyTorch 原生实现（对比用）
"""
import sys
from pathlib import Path

# 未安装包时，允许直接运行 tests/、benchmarks/、examples/ 下的脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from hpcs4inference import fused_attention_decode, get_model_path

sys.stdout.reconfigure(errors="replace")  # Windows GBK 控制台无法编码 emoji

# ============================================================
# 配置
# ============================================================
MODEL_PATH = str(get_model_path())
DEVICE = "cuda"
DTYPE = torch.float16
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.8

# 命令行参数：--no-kernel 用 PyTorch 原生
parser = argparse.ArgumentParser()
parser.add_argument("--no-kernel", action="store_true",
                    help="禁用自定义 kernel，用 PyTorch 原生实现（对比用）")
args = parser.parse_args()
USE_CUSTOM_KERNEL = not args.no_kernel


# ============================================================
# 加载模型
# ============================================================
print("正在加载模型...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    quantization_config=bnb_config,
    device_map=DEVICE,
    # 自定义 forward 依赖显式的因果 mask；sdpa 在 prefill 时不传 mask
    attn_implementation="eager",
)

model.eval()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"✅ 模型加载完成，参数量 {n_params:.1f}M")
print()


# ============================================================
# 性能统计
# ============================================================
STATS = {
    "decode_calls": 0,
    "kernel_time_us": 0.0,
    "prefill_calls": 0,
    "prefill_time_ms": 0.0,
}

# ============================================================
# 替换 Qwen3Attention.forward
# ============================================================
def custom_attn_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,      # ★ 复数，与 HF 一致
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # ---- 1. QKV 投影 + QK-Norm ----
    query_states = self.q_norm(
        self.q_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    key_states = self.k_norm(
        self.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # ---- 2. RoPE ----
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    # ---- 3. 更新 KV cache（新版 API 只传 3 个参数）----
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    B, H, Q, D = query_states.shape
    S = key_states.shape[2]

    # ---- 4. Attention 计算 ----
    if USE_CUSTOM_KERNEL and Q == 1 and S > 1:
        # ★ Decode 阶段：用自定义 kernel
        key_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_rep = repeat_kv(value_states, self.num_key_value_groups)

        # 自定义 kernel 只支持 FP16；量化模型可能输出 BF16，这里显式转换
        orig_dtype = query_states.dtype
        q_c = query_states.contiguous().to(torch.float16)
        k_c = key_rep.contiguous().to(torch.float16)
        v_c = value_rep.contiguous().to(torch.float16)
        out = torch.zeros_like(q_c)

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        fused_attention_decode(q_c, k_c, v_c, out, S)
        end_ev.record()
        torch.cuda.synchronize()
        out = out.to(orig_dtype)

        STATS["decode_calls"] += 1
        STATS["kernel_time_us"] += start_ev.elapsed_time(end_ev) * 1000
        attn_weights = None
    else:
        # Prefill 阶段：PyTorch eager
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

        key_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_rep = repeat_kv(value_states, self.num_key_value_groups)

        scaling = getattr(self, "scaling", 1.0 / (D ** 0.5))
        scores = torch.matmul(query_states, key_rep.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :S]
            scores = scores + causal_mask
        attn_weights_full = torch.softmax(
            scores, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        out = torch.matmul(attn_weights_full, value_rep)

        end_ev.record()
        torch.cuda.synchronize()
        STATS["prefill_calls"] += 1
        STATS["prefill_time_ms"] += start_ev.elapsed_time(end_ev)
        attn_weights = attn_weights_full

    # ---- 5. 输出投影 ----
    attn_output = out.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


# 应用替换
if USE_CUSTOM_KERNEL:
    for layer in model.model.layers:
        layer.self_attn.forward = custom_attn_forward.__get__(
            layer.self_attn, Qwen3Attention
        )
    print("⚡ 已启用自定义 CUDA kernel（decode 阶段）")
else:
    print("🐢 使用 PyTorch 原生实现")
print()


# ============================================================
# 采样
# ============================================================
def sample_token(logits, temperature=0.7, top_p=0.8):
    logits = logits.float() / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()

    idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx[0, idx[0]].item()


# ============================================================
# 自定义 generate 循环
# ============================================================
def generate(input_ids, max_new_tokens=MAX_NEW_TOKENS):
    past_kv = None
    current_ids = input_ids
    generated = []

    start_time = time.time()
    ttft = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
        past_kv = outputs.past_key_values

        logits = outputs.logits[:, -1, :]
        next_id = sample_token(logits, TEMPERATURE, TOP_P)

        if ttft is None:
            ttft = (time.time() - start_time) * 1000

        generated.append(next_id)
        current_ids = torch.tensor([[next_id]], device=DEVICE, dtype=torch.long)

        if next_id == tokenizer.eos_token_id:
            break

    total = time.time() - start_time
    return generated, ttft, total


# ============================================================
# 主循环
# ============================================================
def main():
    print("=" * 70)
    print("Qwen3-0.6B 连续对话")
    if USE_CUSTOM_KERNEL:
        print("  ⚡ 使用自定义 CUDA kernel 加速 decode")
    else:
        print("  🐢 使用 PyTorch 原生实现")
    print("=" * 70)
    print("命令: exit/quit/q 退出 | stats 统计 | clear 清空历史")
    print()

    history = []

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in ["exit", "quit", "q"]:
            print("再见！")
            break

        if user_input == "stats":
            print(f"\n📊 性能统计:")
            print(f"  Decode 阶段:")
            print(f"    调用次数: {STATS['decode_calls']}")
            if STATS["decode_calls"] > 0:
                avg = STATS["kernel_time_us"] / STATS["decode_calls"]
                print(f"    总耗时: {STATS['kernel_time_us']/1000:.2f} ms")
                print(f"    平均: {avg:.2f} us/call")
            print(f"  Prefill 阶段:")
            print(f"    调用次数: {STATS['prefill_calls']}")
            if STATS["prefill_calls"] > 0:
                print(f"    总耗时: {STATS['prefill_time_ms']:.2f} ms")
            continue

        if user_input == "clear":
            history = []
            print("🔄 对话历史已清空")
            continue

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        print("Qwen: ", end="", flush=True)
        gen_ids, ttft, total = generate(inputs.input_ids)

        reply = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(reply)

        n = len(gen_ids)
        tps = n / total if total > 0 else 0
        print(f"\n[TTFT: {ttft:.0f}ms | {n} tokens | {total:.2f}s | {tps:.1f} tokens/s]")

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
