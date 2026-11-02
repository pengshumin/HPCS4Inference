#include "qwen3_kernels.h"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>   // 新增
#include <c10/cuda/CUDAStream.h>     // 新增

#define WARP_SIZE 32

// ============================================================
// 工具函数：warp-level reduce sum（FP32）
// ============================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 内部被调用
__global__ void fused_qk_norm_rope_kvcache_kernel(
    const __half* __restrict__ qkv,
    __half* __restrict__ q_out,
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ q_weight,
    const __half* __restrict__ k_weight,
    const __half* __restrict__ rope_cos,
    const __half* __restrict__ rope_sin,
    int num_heads, int num_kv_heads, int head_dim,
    int position_offset, float eps, int seq_len,
    int seq_per_block)
{
    // ---------- 变量定义（之前漏掉了这部分）----------
    int num_seq_blocks = (seq_len + seq_per_block - 1) / seq_per_block;
    int seq_block = blockIdx.x % num_seq_blocks;
    int bh = blockIdx.x / num_seq_blocks;
    int b = bh / num_heads;
    int h = bh % num_heads;
    int lane = threadIdx.x % 32;
    int ept = (head_dim + 31) / 32;
    int kv_h = h % num_kv_heads;

    const int qkv_stride_s = (num_heads + 2 * num_kv_heads) * head_dim;
    const int q_base = 0;
    const int k_base = num_heads * head_dim;
    const int v_base = (num_heads + num_kv_heads) * head_dim;

    int seq_start = seq_block * seq_per_block;
    int seq_end = min(seq_start + seq_per_block, seq_len);

    for (int s = seq_start; s < seq_end; s++) {
        int pos = position_offset + s;
        int qkv_s_off = b * seq_len * qkv_stride_s + s * qkv_stride_s;

        // ---------- Q: RMSNorm + RoPE ----------
        float q_sq = 0.0f;
        for (int k = 0; k < ept; k++) {
            int d = lane + k * 32;
            if (d < head_dim) {
                float v = __half2float(qkv[qkv_s_off + q_base + h * head_dim + d]);
                q_sq += v * v;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            q_sq += __shfl_down_sync(0xffffffff, q_sq, off);
        float q_rms = rsqrtf(__shfl_sync(0xffffffff, q_sq, 0) / head_dim + eps);

        int num_pairs = head_dim / 2;
        for (int k = 0; k < (num_pairs + 31) / 32; k++) {
            int pair_idx = lane + k * 32;
            if (pair_idx < num_pairs) {
                int de  = pair_idx * 2;
                int do_ = de + 1;
                float v_e = __half2float(qkv[qkv_s_off + q_base + h * head_dim + de]);
                float v_o = __half2float(qkv[qkv_s_off + q_base + h * head_dim + do_]);
                float x_e = v_e * q_rms * __half2float(q_weight[de]);
                float x_o = v_o * q_rms * __half2float(q_weight[do_]);
                float c   = __half2float(rope_cos[pos * head_dim + de]);
                float sn  = __half2float(rope_sin[pos * head_dim + de]);
                int q_out_base = b * seq_len * num_heads * head_dim
                               + s * num_heads * head_dim + h * head_dim;
                q_out[q_out_base + de]  = __float2half(x_e * c - x_o * sn);
                q_out[q_out_base + do_] = __float2half(x_e * sn + x_o * c);
            }
        }

        // ---------- K: RMSNorm + RoPE ----------
        float k_sq = 0.0f;
        for (int k = 0; k < ept; k++) {
            int d = lane + k * 32;
            if (d < head_dim) {
                float v = __half2float(qkv[qkv_s_off + k_base + kv_h * head_dim + d]);
                k_sq += v * v;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            k_sq += __shfl_down_sync(0xffffffff, k_sq, off);
        float k_rms = rsqrtf(__shfl_sync(0xffffffff, k_sq, 0) / head_dim + eps);

        for (int k = 0; k < (num_pairs + 31) / 32; k++) {
            int pair_idx = lane + k * 32;
            if (pair_idx < num_pairs) {
                int de  = pair_idx * 2;
                int do_ = de + 1;
                float v_e = __half2float(qkv[qkv_s_off + k_base + kv_h * head_dim + de]);
                float v_o = __half2float(qkv[qkv_s_off + k_base + kv_h * head_dim + do_]);
                float x_e = v_e * k_rms * __half2float(k_weight[de]);
                float x_o = v_o * k_rms * __half2float(k_weight[do_]);
                float c   = __half2float(rope_cos[pos * head_dim + de]);
                float sn  = __half2float(rope_sin[pos * head_dim + de]);
                int kc_idx = b * num_kv_heads * seq_len * head_dim
                           + kv_h * seq_len * head_dim
                           + pos * head_dim;
                k_cache[kc_idx + de]  = __float2half(x_e * c - x_o * sn);
                k_cache[kc_idx + do_] = __float2half(x_e * sn + x_o * c);
            }
        }

        // ---------- V ----------
        for (int k = 0; k < ept; k++) {
            int d = lane + k * 32;
            if (d < head_dim) {
                int vc_idx = b * num_kv_heads * seq_len * head_dim
                           + kv_h * seq_len * head_dim
                           + pos * head_dim + d;
                v_cache[vc_idx] = qkv[qkv_s_off + v_base + kv_h * head_dim + d];
            }
        }
    }
}

// 用上了
void fused_qk_norm_rope_kvcache(
    torch::Tensor& qkv, torch::Tensor& q_out,
    torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& q_weight, torch::Tensor& k_weight,
    torch::Tensor& rope_cos, torch::Tensor& rope_sin,
    int num_heads, int num_kv_heads, int head_dim,
    int position_offset, float eps)
{
    int B = qkv.size(0);
    int S = qkv.size(1);
    int seq_per_block = 16;
    int num_seq_blocks = (S + seq_per_block - 1) / seq_per_block;
    dim3 grid(B * num_heads * num_seq_blocks);
    dim3 block(32);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_qk_norm_rope_kvcache_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(q_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(rope_cos.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(rope_sin.data_ptr<at::Half>()),
        num_heads, num_kv_heads, head_dim,
        position_offset, eps, S, seq_per_block);
}

// ============================================================
// Kernel 2: 融合 Attention Decode（Split-K 版本）
// 阶段 1: 每个 block 处理 (b, h, split_id)，输出部分 (m, l, acc)
// 阶段 2: 合并 num_splits 个部分结果
// ============================================================
#define NUM_WARPS 8
#define WARPSIZE 32
#define NUM_SPLITS 8

// 内部被调用
// ---------- 阶段 1：部分结果 ----------
__global__ void fused_attention_decode_split_kernel(
    const __half* __restrict__ q,        // [B, H, 1, D]
    const __half* __restrict__ k_cache,  // [B, H, S, D]
    const __half* __restrict__ v_cache,  // [B, H, S, D]
    float* __restrict__ partial_acc,     // [B*H*NUM_SPLITS, D]
    float* __restrict__ partial_m,       // [B*H*NUM_SPLITS]
    float* __restrict__ partial_l,       // [B*H*NUM_SPLITS]
    int num_heads, int seq_len, int head_dim, int num_splits)
{
    int split_id = blockIdx.x % num_splits;
    int bh = blockIdx.x / num_splits;
    int b = bh / num_heads;
    int h = bh % num_heads;
    int warp_id = threadIdx.x / WARPSIZE;
    int lane    = threadIdx.x % WARPSIZE;
    int ept = head_dim / WARPSIZE;   // 每线程元素数

    // 加载 Q（旧索引：隐式向量化，比 coalesced 更快）
    const float scale = rsqrtf((float)head_dim);
    float q_local[8];
    for (int i = 0; i < ept; i++) {
        int d = lane * ept + i;
        int q_idx = (b * num_heads + h) * head_dim + d;
        q_local[i] = __half2float(q[q_idx]) * scale;
    }

    // 本 block 负责的 S 范围
    int s_per_split = (seq_len + num_splits - 1) / num_splits;
    int s_start_split = split_id * s_per_split;
    int s_end_split = min(s_start_split + s_per_split, seq_len);

    // 本 block 内，每个 warp 负责一段
    int s_per_warp = (s_end_split - s_start_split + NUM_WARPS - 1) / NUM_WARPS;
    int s_start = s_start_split + warp_id * s_per_warp;
    int s_end = min(s_start + s_per_warp, s_end_split);

    // 在线 softmax
    float m_local = -CUDART_INF_F;
    float l_local = 0.0f;
    float acc[8] = {0};

    for (int s = s_start; s < s_end; s++) {
        // score = Q · K^T
        float partial = 0.0f;
        for (int i = 0; i < ept; i++) {
            int d = lane * ept + i;
            int kc_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
            partial += q_local[i] * __half2float(k_cache[kc_idx]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, off);
        float score = __shfl_sync(0xffffffff, partial, 0);

        // 在线 softmax 更新
        float m_new = fmaxf(m_local, score);
        float alpha = expf(m_local - m_new);
        float p = expf(score - m_new);
        l_local = l_local * alpha + p;
        m_local = m_new;

        // 累加 V
        for (int i = 0; i < ept; i++) {
            int d = lane * ept + i;
            int vc_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
            acc[i] = acc[i] * alpha + p * __half2float(v_cache[vc_idx]);
        }
    }

    // 跨 warp 归约
    __shared__ float s_m[NUM_WARPS];
    __shared__ float s_l[NUM_WARPS];
    __shared__ float s_acc[NUM_WARPS][8 * 32];

    if (lane == 0) {
        s_m[warp_id] = m_local;
        s_l[warp_id] = l_local;
    }
    for (int i = 0; i < ept; i++) {
        int d = lane * ept + i;
        s_acc[warp_id][d] = acc[i];
    }
    __syncthreads();

    if (warp_id == 0) {
        // 找全局 max
        float m_global = -CUDART_INF_F;
        for (int w = 0; w < NUM_WARPS; w++)
            m_global = fmaxf(m_global, s_m[w]);

        // 合并 l 和 acc
        float l_global = 0.0f;
        float acc_final[8] = {0};
        for (int w = 0; w < NUM_WARPS; w++) {
            float scale_w = expf(s_m[w] - m_global);
            l_global += s_l[w] * scale_w;
            for (int i = 0; i < ept; i++) {
                int d = lane * ept + i;
                acc_final[i] += s_acc[w][d] * scale_w;
            }
        }

        // 写出部分结果
        int out_idx = bh * num_splits + split_id;
        if (lane == 0) {
            partial_m[out_idx] = m_global;
            partial_l[out_idx] = l_global;
        }
        for (int i = 0; i < ept; i++) {
            int d = lane * ept + i;
            partial_acc[out_idx * head_dim + d] = acc_final[i];
        }
    }
}

// 内部被调用
// ---------- 阶段 2：合并部分结果 ----------
__global__ void fused_attention_decode_merge_kernel(
    const float* __restrict__ partial_acc,  // [B*H*NUM_SPLITS, D]
    const float* __restrict__ partial_m,    // [B*H*NUM_SPLITS]
    const float* __restrict__ partial_l,    // [B*H*NUM_SPLITS]
    __half* __restrict__ output,            // [B, H, 1, D]
    int num_heads, int head_dim, int num_splits)
{
    // 一个 block 处理一个 (b, h)
    int bh = blockIdx.x;
    int tid = threadIdx.x;

    // 每个线程处理多个 d
    for (int d = tid; d < head_dim; d += blockDim.x) {
        // 找全局 max
        float m_global = -CUDART_INF_F;
        for (int k = 0; k < num_splits; k++) {
            m_global = fmaxf(m_global, partial_m[bh * num_splits + k]);
        }

        // 合并 l 和 acc
        float l_global = 0.0f;
        float acc = 0.0f;
        for (int k = 0; k < num_splits; k++) {
            float sk = expf(partial_m[bh * num_splits + k] - m_global);
            l_global += partial_l[bh * num_splits + k] * sk;
            acc += partial_acc[(bh * num_splits + k) * head_dim + d] * sk;
        }

        output[bh * head_dim + d] = __float2half(acc / l_global);
    }
}

// 用上了
// ---------- Host 端 ----------
void fused_attention_decode(
    torch::Tensor& q, torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& output, int seq_len)
{
    /*
    q: 三角函数旋转位置编码后的矩阵
    k_cache: [batch, num_heads, seq_len, head_dim]
    v_cache: [batch, num_heads, seq_len, head_dim]
    output: [batch, num_heads, 1, head_dim], 最终输出写这
    seq_len: 输入序列长度
    */
    int B = q.size(0);
    int H = q.size(1);
    int D = q.size(3);

    // 根据 seq_len 自适应选择 num_splits
    int num_splits = 1;
    if (seq_len > 64) num_splits = 2;
    if (seq_len > 256) num_splits = 4;
    if (seq_len > 1024) num_splits = NUM_SPLITS;   // 8

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 分配临时 buffer（float32）
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto partial_acc = torch::empty({B * H * num_splits, D}, opts_f32);
    auto partial_m = torch::empty({B * H * num_splits}, opts_f32);
    auto partial_l = torch::empty({B * H * num_splits}, opts_f32);

    // 阶段 1
    dim3 grid1(B * H * num_splits);
    dim3 block1(NUM_WARPS * WARPSIZE);
    fused_attention_decode_split_kernel<<<grid1, block1, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
        partial_acc.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        H, seq_len, D, num_splits);

    // 阶段 2
    dim3 grid2(B * H);
    dim3 block2(256);
    fused_attention_decode_merge_kernel<<<grid2, block2, 0, stream>>>(
        partial_acc.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        H, D, num_splits);
}

// 内部被调用
// ============================================================
// Kernel 3: 融合 SiLU + Mul
// 说明：原 kernel 名为 fused_rmsnorm_silu_mul，但 RMSNorm 结果从未被使用。
//       这里简化为纯 SiLU(gate) * up，删除冗余参数。
// ============================================================
__global__ void fused_silu_mul_kernel(
    const __half* __restrict__ gate,    // [B*S, I]
    const __half* __restrict__ up,      // [B*S, I]
    __half* __restrict__ output,        // [B*S, I]
    int intermediate_dim)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = tid; i < intermediate_dim; i += blockDim.x) {
        float g = __half2float(gate[row * intermediate_dim + i]);
        float u = __half2float(up[row * intermediate_dim + i]);
        float silu_g = g / (1.0f + expf(-g));
        output[row * intermediate_dim + i] = __float2half(silu_g * u);
    }
}   

// 用上了
void fused_silu_mul(
    torch::Tensor& gate, torch::Tensor& up,
    torch::Tensor& output)
{
    int rows = gate.size(0);
    int I = gate.size(1);

    dim3 grid(rows);
    dim3 block(256);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_silu_mul_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        I
    );
}