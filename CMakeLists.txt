# 说明：主构建方式是 setup.py（pip install -e .）。
# 此 CMake 文件保留给 IDE 集成和手动构建使用。
cmake_minimum_required(VERSION 3.20)
project(HPCS4Inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)  # GTX 1650 Ti = Turing

# 优化标志（Release）
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

if(MSVC)
    # 源码含中文注释，避免 MSVC 按 GBK 解码时吞掉换行
    add_compile_options(/utf-8)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# 编译CUDA内核为库
add_library(qwen3_kernels STATIC
    src/qwen3_kernels.cu
)
target_include_directories(qwen3_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(qwen3_kernels PRIVATE CUDA::cudart ${TORCH_LIBRARIES})

# pybind11扩展：生成 hpcs4inference._C 中的 _C 模块
pybind11_add_module(_C
    src/binding.cpp
)
target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)
target_link_libraries(_C PRIVATE qwen3_kernels CUDA::cudart ${TORCH_LIBRARIES})
