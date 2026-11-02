from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pybind11

setup(
    name="HPCS4Inference",
    version="0.1.0",
    packages=["hpcs4inference"],
    ext_modules=[
        CUDAExtension(
            name="hpcs4inference._C",
            sources=[
                "src/binding.cpp",
                "src/qwen3_kernels.cu",
            ],
            include_dirs=[pybind11.get_include()],
            extra_compile_args={
                # /utf-8: 源码含中文注释，避免 MSVC 按 GBK 解码时吞掉换行
                "cxx": ["-O3", "/std:c++17", "/utf-8"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-arch=sm_75",
                    "-Xcompiler=/utf-8",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

'''
# 编译（在 VS 开发者命令提示符里执行）
conda activate env3.10
cd D:\cppproject\HPCS4Inference

# 关键：设置 DISTUTILS_USE_SDK
set DISTUTILS_USE_SDK=1

# 编译，不创建新环境隔离
# --no-deps：只注册本项目，不安装/升级/降级环境中已有的依赖
pip install -e . --no-build-isolation --no-deps
'''
