#!/bin/bash

# 设置环境变量以解决 AMD GPU 在 WSL 下的兼容性问题
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
# 禁用某些可能导致崩溃的优化
export HIP_FORCE_DEV_KERNEL=1

echo "正在启动后端服务..."
cd backend
python3 main.py

