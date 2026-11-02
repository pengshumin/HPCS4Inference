"""运行时配置：模型路径与 Windows DLL 目录。

所有机器相关的路径都集中在这里，脚本不要再硬编码盘符。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-0.6B"
MODEL_PATH_ENV = "HPCS4INFERENCE_MODEL_PATH"
CUDA_BIN_ENV = "HPCS4INFERENCE_CUDA_BIN"


def get_model_path() -> Path:
    """返回模型目录；可用 HPCS4INFERENCE_MODEL_PATH 覆盖默认值。"""
    raw = os.environ.get(MODEL_PATH_ENV)
    path = Path(raw).expanduser() if raw else DEFAULT_MODEL_PATH
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def setup_windows_dll_paths() -> None:
    """在 Windows 上注册 CUDA 和 PyTorch 的 DLL 目录。

    需要在导入编译扩展（hpcs4inference._C）之前调用；包初始化时会自动调用。
    DLL 目录按以下顺序收集：
      1. HPCS4INFERENCE_CUDA_BIN（显式覆盖）
      2. CUDA_HOME / CUDA_PATH 下的 bin
      3. 当前 torch 安装目录下的 lib
    """
    if sys.platform != "win32":
        return

    candidates: list[Path] = []

    cuda_bin = os.environ.get(CUDA_BIN_ENV)
    if cuda_bin:
        candidates.append(Path(cuda_bin))

    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(env_var)
        if cuda_home:
            candidates.append(Path(cuda_home) / "bin")

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        candidates.append(Path(torch.__file__).resolve().parent / "lib")

    for directory in candidates:
        if directory.is_dir():
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                pass
