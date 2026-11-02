#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// 融合的 QK-Norm + RoPE + KV Cache 写入
// 将原来3个kernel（split QKV → RMSNorm → RoPE）融合为1个
void fused_qk_norm_rope_kvcache(
    torch::Tensor& qkv,           // [B, S, (H + 2*KVH)*D] FP16
    torch::Tensor& q_out,         // [B, S, H*D] FP16
    torch::Tensor& k_cache,       // [B, KVH, S, D] FP16
    torch::Tensor& v_cache,       // [B, KVH, S, D] FP16
    torch::Tensor& q_weight,      // [D] FP16
    torch::Tensor& k_weight,      // [D] FP16
    torch::Tensor& rope_cos,      // [S, D] FP16
    torch::Tensor& rope_sin,      // [S, D] FP16
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset,
    float eps
);

// 融合的 Attention + Softmax + V 加权
// 使用在线 softmax（FlashAttention 思想）
void fused_attention_decode(
    torch::Tensor& q,             // [B, H, 1, D] FP16
    torch::Tensor& k_cache,       // [B, H, S, D] FP16
    torch::Tensor& v_cache,       // [B, H, S, D] FP16
    torch::Tensor& output,        // [B, H, 1, D] FP16
    int seq_len
);

// 融合的 SiLU + Mul（FFN 的激活部分）
// 说明：原接口名为 fused_rmsnorm_silu_mul，但 RMSNorm 结果从未被使用，
//       因此只保留 SiLU(gate) * up。
void fused_silu_mul(
    torch::Tensor& gate,          // [B*S, I] FP16
    torch::Tensor& up,            // [B*S, I] FP16
    torch::Tensor& output         // [B*S, I] FP16
);
