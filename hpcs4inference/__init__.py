"""HPCS4Inference: 面向 Qwen3 的融合 CUDA kernel 推理加速。"""

from .runtime import (
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    get_model_path,
    setup_windows_dll_paths,
)

# 必须先注册 DLL 目录，再导入编译扩展。
setup_windows_dll_paths()

from ._C import (  # noqa: E402
    fused_attention_decode,
    fused_qk_norm_rope_kvcache,
    fused_silu_mul,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "PROJECT_ROOT",
    "fused_attention_decode",
    "fused_qk_norm_rope_kvcache",
    "fused_silu_mul",
    "get_model_path",
]
