#!/bin/bash

# ==========================================
# 1. 核心构建开关 (1=启用, 0=禁用)
# ==========================================
export ENABLE_GCC=0
export ENABLE_CLANG=0
export ENABLE_TRITON=1

# 是否默认清理构建目录 (--clean / --no-clean)
export DEFAULT_DO_CLEAN="--clean"

# ==========================================
# 2. 目标平台与并发设置
# ==========================================
# 平台选择: "x86" 或 "rv" (RISC-V)
# export PLATFORM="rv"
export PLATFORM="x86"

# 编译线程数
export MAX_MULTITHREADING=8
export THREADS_LIST="1 4 8"

# ==========================================
# 3. 基准测试列表 (Benchmarks)
# ==========================================
# 在这里列出你想运行的测试名称，用空格分隔
# 示例: "matmul softmax layernorm correlation dropout resize rope warp"
export BENCHMARKS_LIST="add"
export MODE="Benchmark"

# ==========================================
# 4. 路径设置 (根据你的环境修改)
# ==========================================
# 源码目录 (通常不需要改，除非你移动了脚本)
DIR_PATH=$(dirname $(readlink -f "$0")) # 获取当前脚本绝对路径
export SRC_DIR="${DIR_PATH}/src"
export TRITON_PLUGIN_DIRS="${DIR_PATH}/triton-cpu"

# LLVM 与工具链路径
export LLVM_BUILD_DIR="${DIR_PATH}/llvm-project/build"
if [ "$PLATFORM" = "rv" ]; then
  # RISC-V 工具链路径
  export CLANG_BUILD_DIR="${DIR_PATH}/llvm-project/build-86b69c-rv"
  export RISCV_GNU_TOOLCHAIN_DIR="/home/buddy-team-share/spacemit-toolchain-linux-glibc-x86_64-v1.0.1"
else
  # x86 工具链路径
  export CLANG_BUILD_DIR="${DIR_PATH}/llvm-project/build"
  export GCC_X86_BUILD_DIR="/usr"
fi

# ==========================================
# 5. 编译器参数 (高级设置)
# ==========================================
# 通用 C++ 标准
export CPP_STD="-std=c++17"

# x86 优化参数
export X86_FLAGS="-march=native -fvectorize -fslp-vectorize -O3"

# RISC-V 优化参数
export RV_FLAGS="-march=rv64gcv -fvectorize -fslp-vectorize -O3"
export RV_GCC_FLAGS="-march=rv64gcv_zvl256b -mabi=lp64d -O3"
