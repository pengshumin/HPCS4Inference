#include <torch/extension.h>
#include "qwen3_kernels.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HPCS4Inference optimized CUDA kernels (Qwen3)";

    m.def("fused_qk_norm_rope_kvcache", &fused_qk_norm_rope_kvcache,
          "Fused QK-Norm + RoPE + KV Cache write",
          py::arg("qkv"), py::arg("q_out"),
          py::arg("k_cache"), py::arg("v_cache"),
          py::arg("q_weight"), py::arg("k_weight"),
          py::arg("rope_cos"), py::arg("rope_sin"),
          py::arg("num_heads"), py::arg("num_kv_heads"),
          py::arg("head_dim"),
          py::arg("position_offset"), py::arg("eps"));

    m.def("fused_attention_decode", &fused_attention_decode,
          "Fused attention decode with online softmax",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"),
          py::arg("output"), py::arg("seq_len"));

    m.def("fused_silu_mul", &fused_silu_mul,
          "Fused SiLU(gate) * up",
          py::arg("gate"), py::arg("up"), py::arg("output"));
}
