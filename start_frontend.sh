#!/bin/bash

# 设置环境变量以解决 AMD GPU 在 WSL 下的兼容性问题
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
# 禁用某些可能导致崩溃的优化
export HIP_FORCE_DEV_KERNEL=1

# 设置 DashScope API Key（请替换为您的实际API Key）
# 方式1：直接在这里设置（不推荐，会暴露在代码中）
export DASHSCOPE_API_KEY="sk-2f8ad974d53748c58af29aeb347088d9"

# 方式2：从环境变量读取（推荐）
# 如果已经在系统环境变量中设置了DASHSCOPE_API_KEY，则无需在这里设置

echo "正在启动后端服务..."
cd backend
python3 main.py

