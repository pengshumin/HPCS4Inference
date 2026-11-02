import sys
from pathlib import Path

# 未安装包时，允许直接运行 tests/、benchmarks/、examples/ 下的脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from hpcs4inference import get_model_path

model_path = str(get_model_path())

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
model.eval()

# ============================================================
# 关键优化 1: torch.compile 自动算子融合
# ============================================================
model = torch.compile(model, mode="max-autotune")

# ============================================================
# 关键优化 2: 批处理推理（提升算术强度）
# ============================================================
def benchmark_batch(model, tokenizer, prompts, batch_size, max_new_tokens=128):
    """批处理推理基准测试"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = outputs.shape[0] * max_new_tokens
    tps = total_tokens / elapsed
    print(f"Batch size: {batch_size} | 总生成: {total_tokens} tokens | "
          f"耗时: {elapsed:.2f}s | 速度: {tps:.2f} tokens/s")
    return tps

# ============================================================
# 关键优化 3: CUDA Graphs 消除内核启动开销
# ============================================================
def benchmark_cuda_graphs(model, tokenizer, prompt, max_new_tokens=128):
    """使用 CUDA Graphs 加速 Decode 阶段"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Warmup：捕获前的预热（必须）
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # 准备静态输入（CUDA Graph 要求固定形状）
    static_input_ids = input_ids.clone()
    static_attention_mask = attention_mask.clone()

    # 捕获 CUDA Graph
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        # 第一次运行，建立内存池
        model(input_ids=static_input_ids, attention_mask=static_attention_mask)
        torch.cuda.synchronize()

        with torch.cuda.graph(g):
            static_output = model(
                input_ids=static_input_ids,
                attention_mask=static_attention_mask
            )

    # 用 CUDA Graph 重放进行生成
    torch.cuda.synchronize()
    start = time.time()

    generated = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            g.replay()
        # 采样（简化：取 argmax）
        next_token = static_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

    torch.cuda.synchronize()
    elapsed = time.time() - start
    tps = len(generated) / elapsed
    print(f"CUDA Graphs | 生成: {len(generated)} tokens | "
          f"耗时: {elapsed:.2f}s | 速度: {tps:.2f} tokens/s")
    return tps


if __name__ == "__main__":
    # ---- 测试 1: 单请求基线 ----
    print("=" * 60)
    print("测试 1: 单请求基线")
    print("=" * 60)

    prompt = "介绍一下爱因斯坦"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        start = time.time()
        outputs = model.generate(
            **inputs, max_new_tokens=128, do_sample=False
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"单请求 | 速度: {128/elapsed:.2f} tokens/s")

    # ---- 测试 2: 批处理 ----
    print("\n" + "=" * 60)
    print("测试 2: 批处理推理")
    print("=" * 60)

    prompts = ["介绍一下爱因斯坦"] * 8  # batch size = 8
    benchmark_batch(model, tokenizer, prompts, batch_size=8, max_new_tokens=128)

    # ---- 测试 3: CUDA Graphs ----
    print("\n" + "=" * 60)
    print("测试 3: CUDA Graphs")
    print("=" * 60)

    benchmark_cuda_graphs(model, tokenizer, text, max_new_tokens=128)

'''
# 运行（在项目根目录）
python benchmarks/optimized_inference.py
'''
