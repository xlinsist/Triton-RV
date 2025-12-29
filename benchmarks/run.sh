#!/bin/bash

# 获取当前脚本的目录
DIR=$(dirname $(readlink -f "$0"))

# 加载全局配置
if [ ! -f "${DIR}/global_config.sh" ]; then    
  echo "Error: global_config.sh not found!"    
  exit 1
fi
source "${DIR}/global_config.sh"

export LD_LIBRARY_PATH="${DIR}/openmp-sysroot-riscv/lib:$LD_LIBRARY_PATH"

# 将基准测试列表转换为数组
IFS=' ' read -r -a BENCHMARKS <<< "$BENCHMARKS_LIST"
# 根据全局配置加载线程配置
IFS=' ' read -r -a THREADS <<< "$THREADS_LIST"
# 根据启用的编译器设置 COMPILERS 数组
COMPILERS=()
if [ "$ENABLE_GCC" -eq 1 ]; then
  COMPILERS+=("gcc")
fi
if [ "$ENABLE_CLANG" -eq 1 ]; then
  COMPILERS+=("clang")
fi
if [ "$ENABLE_TRITON" -eq 1 ]; then
  COMPILERS+=("triton")
fi

# 遍历每个基准测试
for BENCHMARK in "${BENCHMARKS[@]}"; do
  BUILD_DIR="${DIR}/build-${BENCHMARK}"
  BIN_DIR="${BUILD_DIR}/bin"

  for COMPILER in "${COMPILERS[@]}"; do
    for f_sub in "${BIN_DIR}/${COMPILER}/"*/; do
      if [ ! -d "${f_sub}" ]; then
        continue
      fi

      kernel_name=$(basename "$f_sub")
      echo "Processing kernel: ${kernel_name}"

      # 加载形状配置
      source "${f_sub}/${kernel_name}.cfg"

      for THREAD in "${THREADS[@]}"; do
        for shape in "${SHAPE[@]}"; do
          for kernel in "${f_sub}/${kernel_name}"*.elf; do
            echo "Running kernel: ${kernel}"
            log_file="${f_sub}/$(basename "${kernel}" .elf)_T${THREAD}_S${shape}.log"

            DB_FILE="${BUILD_DIR}/${kernel_name}" TRITON_CPU_MAX_THREADS=${THREAD} bash -c "${kernel} ${shape}" 2> "${log_file}"
          done
        done
      done
    done
  done
done

echo "Run all benchmarks completed."
