# HPCS4Inference

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch&logoColor=white)
![GPU](https://img.shields.io/badge/Verified%20on-GTX%201650-lightgrey)

面向大模型推理的高性能 CUDA 算子与可插拔加速框架（**原型阶段**）。

当前仓库已实现一组面向 **Qwen3-0.6B decode 阶段** 的融合 CUDA kernel，并通过 PyTorch C++ 扩展以 `hpcs4inference._C` 的形式暴露给 Python。示例通过 **猴子补丁** 替换 Hugging Face 的 `Qwen3Attention.forward`，在不修改 transformers 源码的前提下接入自定义算子。

> 这是一个持续演进中的系统原型，而非完整的推理引擎。下文所有性能数据均标注了具体硬件、形状与统计范围。

---

## 目录

- [当前已实现](#当前已实现)
- [基准环境](#基准环境)
- [基准结果](#基准结果)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [设计要点](#设计要点)
- [路线图](#路线图)
- [已知限制](#已知限制)

---

## 当前已实现

### 融合算子

| 算子 | 融合内容 | 输入 / 输出 | 状态 |
|---|---|---|---|
| `fused_qk_norm_rope_kvcache` | Q/K RMSNorm + RoPE + KV Cache 写入 | `qkv [B,S,(H+2·KVH)·D]`；输出 `q_out`、`k_cache`、`v_cache` | ✅ 已实现并验证 |
| `fused_attention_decode` | Decode Attention + Online Softmax + V 加权，自适应 Split-K | `q [B,H,1,D]`、`k/v_cache [B,H,S,D]` | ✅ 已实现并验证 |
| `fused_silu_mul` | SiLU(gate) × up | `gate/up/output [B·S,I]` | ✅ 已实现并验证 |

- **FP16** 计算，目标架构 `sm_75`（Turing）。
- 支持 **GQA** 的 QKV 打包布局：`(H + 2·KVH) · D`。
- Python 包：`hpcs4inference`；扩展模块：`hpcs4inference._C`。
- 示例已通过猴子补丁接入 `Qwen3Attention`，作为后续“可插拔框架”的雏形。

> 当前**仅验证了 Qwen3-0.6B 的 decode 路径**；prefill、分页 KV Cache、连续批处理、BF16/FP8 等尚未实现。

---

## 基准环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce GTX 1650（4 GB，sm_75） |
| 操作系统 | Windows 11 |
| Python | 3.10.18 |
| PyTorch | 2.7.1 + CUDA 11.8 |
| 编译器 | MSVC 19.29 |
| 被测模型 | Qwen3-0.6B（FP16 kernel；端到端对话使用 4-bit 量化） |
| 计时方式 | CUDA Event（`torch.cuda.Event`） |

> 所有数据均来自上述单机环境的一次或多次实跑，**不代表其他显卡或模型规模下的表现**。

---

## 基准结果

### 1. Attention Decode（真实 Qwen3-0.6B 形状）

测试条件：`B=1, H=16, KVH=8, D=128`，上下文 `S=512`，逐 token 调用，统计 3584 次（128 token × 28 层）的平均值。

| 实现 | 单次调用延迟 | 每 token attention 耗时（28 层） | 128 token 总耗时 |
|---|---:|---:|---:|
| PyTorch 原生 | 148.2 µs | 4.15 ms | 516 ms |
| 自定义 CUDA kernel | **64.9 µs** | **1.82 ms** | **230 ms** |
| 加速比 | **≈ 2.28×** | **≈ 2.28×** | **≈ 2.24×** |

> 该数据**只统计 attention 子模块**，不含 MLP、RMSNorm、LM Head 等其余开销。因此端到端加速比会明显低于这一数字。

### 2. Attention Decode（合成形状扫描）

在 `B=1, H=4, D=64` 的合成张量上扫描 `S=64…2048`，多次实跑得到的 kernel 级加速比大致落在 **2.6×–8×** 区间，随 `S` 与运行批次波动较明显（小显存 GPU 上计时抖动较大）。该结果**仅用于观察趋势**，不作为精确性能结论。

### 3. 端到端对话（4-bit 量化）

测试条件：`examples/chat.py`，Qwen3-0.6B 4-bit（bitsandbytes），prompt 30 tokens，固定生成 128 tokens（greedy），猴子补丁接入自定义 kernel。

| 指标 | 实测值 |
|---|---:|
| 首 token 延迟（TTFT） | 948 ms |
| 128 token 总耗时 | 14.05 s |
| Decode 吞吐 | **≈ 9.1 tokens/s** |

> 在相同设置下，未启用自定义 kernel 的对照约为 8.8 tokens/s。两者差异处于单次测量的波动范围内，**当前端到端收益有限**：4-bit 权重反量化与 MLP 仍占主要耗时，kernel 的收益集中在 attention 子模块。该表用于说明系统可用性，不作为端到端加速结论。

### 4. 数值正确性

与 PyTorch 参考实现对比（FP16）：

| 算子 | 测试形状 | 最大绝对误差 |
|---|---|---:|
| `fused_silu_mul` | `B=4, I=128` | 3.9 × 10⁻³ |
| `fused_attention_decode` | `B=1, H=4, S=16, D=64` | < 1 × 10⁻⁶ |
| `fused_attention_decode` | `B=1, H=16, D=128, S=13…1100` | ≤ 4.8 × 10⁻⁴ |
| `fused_qk_norm_rope_kvcache` | `B=1, S=4, H=4, KVH=2, D=64` | < 1 × 10⁻⁶ |
| RoPE（真实频率） | `B=1, S=4, H=4, KVH=2, D=64` | < 1 × 10⁻⁶ |

> 误差在 FP16 正常范围内；项目测试脚本以 `1e-2` 作为通过阈值。

---

## 快速开始

### 环境要求

- Windows + NVIDIA GPU（当前验证 `sm_75`）
- CUDA Toolkit、Visual Studio 生成工具（MSVC）
- conda 环境中已安装 `torch`、`pybind11`

依赖版本固定在 `requirements.txt` 中（`torch==2.7.1+cu118`、`transformers==5.14.1`、`pybind11==3.0.3` 等）：

```bat
pip install -r requirements.txt
```

### 编译

在 **x64 Native Tools / VS 开发者命令提示符** 中执行：

```bat
conda activate env3.10
cd D:\cppproject\HPCS4Inference
set DISTUTILS_USE_SDK=1
pip install -e . --no-build-isolation --no-deps
```

- `--no-build-isolation`：构建过程需要复用已安装的 torch / pybind11。
- `--no-deps`：只注册本项目，不安装、升级或降级已有依赖。
- 若仅本地使用，也可直接 `python setup.py build_ext --inplace`，产物落在 `hpcs4inference/_C.cp310-win_amd64.pyd`；各脚本会自动定位仓库根目录，无需安装。

### 模型与路径

默认模型目录为 `models/Qwen3-0.6B`（已被 `.gitignore` 忽略），可通过环境变量覆盖：

```bat
set HPCS4INFERENCE_MODEL_PATH=D:\path\to\Qwen3-0.6B
```

Windows 下 CUDA DLL 目录从 `CUDA_HOME` / `CUDA_PATH` 推导；也可显式指定：

```bat
set HPCS4INFERENCE_CUDA_BIN=D:\cuda-toolkit\cuda_development\bin
```

### 验证

```bat
python tests/test_kernels.py
python tests/test_qk_norm.py
python tests/test_rope_real.py
```

### 基准测试

```bat
python benchmarks/simulate_qwen3_decode.py   :: 真实 Qwen3 形状的 decode 模拟
python benchmarks/benchmark_v2.py            :: kernel vs PyTorch 扫描
python benchmarks/benchmark.py               :: 含 QK-Norm / KV Cache 写入
```

### 运行对话示例

```bat
python examples/chat.py              :: 自定义 CUDA kernel 加速 decode
python examples/chat.py --no-kernel  :: PyTorch 原生实现（对照）
```

> 示例使用 `attn_implementation="eager"`。这是必要配置：默认的 SDPA 在 prefill 阶段不传显式因果掩码，会使自定义 `forward` 退化为非因果注意力。

---

## 项目结构

```text
HPCS4Inference/
├── hpcs4inference/          # Python 包
│   ├── __init__.py          # 注册 DLL 目录并导出 _C 中的算子
│   └── runtime.py           # 模型路径 / DLL 路径的集中配置
├── src/                     # C++ / CUDA 源码
│   ├── binding.cpp          # pybind11 绑定
│   ├── qwen3_kernels.cu     # 三个融合 kernel 及 host 封装
│   └── qwen3_kernels.h
├── examples/                # 示例（猴子补丁接入 Qwen3Attention）
├── benchmarks/              # 基准脚本
├── tests/                   # 可执行验证脚本
├── setup.py                 # 主构建入口
├── pyproject.toml
└── CMakeLists.txt           # 辅助构建（IDE / 手动编译）
```

---

## 设计要点

- **算子融合**：将原本多次 kernel 启动（QKV 拆分 → RMSNorm → RoPE → KV Cache 写入，以及 Attention → Softmax → V 加权）合并为单次调用，减少启动开销与显存往返。
- **Online Softmax**：decode attention 采用流式 softmax，无需物化完整注意力矩阵。
- **自适应 Split-K**：根据 `S` 动态选择分段数，在长上下文下增加并行度；长序列结果的合并由第二个 kernel 完成。
- **GQA 感知布局**：QKV 以 `(H + 2·KVH) · D` 打包，KV Cache 保持 `[B, KVH, S, D]`，避免重复存储。
- **非侵入式接入**：示例在运行时替换 `Qwen3Attention.forward`，不修改 transformers 源码——这是后续框架化的核心机制。

---

## 路线图

以下为**规划中的方向，当前尚未实现**：

- [ ] **统一算子接口与后端注册**：将 attention、norm、RoPE、FFN 等抽象为稳定接口，支持 CUDA / Triton / CUTLASS 等多后端注册与调度。
- [ ] **猴子补丁式运行时**：模型无关的 patch 管理器，负责版本探测、算子替换、异常回退，做到“开箱即用、可随时关闭”。
- [ ] **更完整的推理路径**：prefill 融合 attention、分页 KV Cache、连续批处理、CUDA Graphs。
- [ ] **更多数据类型**：BF16 / FP8 / 低比特量化 kernel，适配不同显卡。
- [ ] **自动调优**：按显卡、形状、精度自动选择 kernel 配置。
- [ ] **更多模型**：在 Qwen3 之外支持 Llama、Qwen2.5 等结构。
- [ ] **跨平台**：Linux 构建与 CI。

---

## 已知限制

- 当前仅验证 **Qwen3-0.6B** 的 **decode** 路径，且为 **FP16** 计算。
- kernel 假设 GQA 布局，且 `head_dim` 需为 32 的倍数、不超过 256（内部按每线程最多 8 个元素展开）；其他结构需要适配。
- 尚未实现 paged KV Cache、连续批处理、多卡与多流。
- Windows 构建已验证，其他平台未测试；仓库暂无 CI。
- 端到端收益受模型其余部分（反量化、MLP、LM Head）限制，详见[基准结果](#基准结果)。

---

## 许可证

尚未指定，正式发布前补充。
