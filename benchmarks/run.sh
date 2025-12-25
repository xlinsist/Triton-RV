#!/bin/bash

MODE="Benchmark"
DIR=`dirname $0`
export LD_LIBRARY_PATH="${DIR}/openmp-sysroot-riscv/lib:$LD_LIBRARY_PATH"

# 将数组改为字符串
BENCHMARKS_STR="matmul softmax correlation layernorm dropout rope resize"
# BENCHMARKS_STR="softmax"

# 将字符串转换为数组进行循环
BENCHMARKS=($BENCHMARKS_STR)

for BENCHMARK in "${BENCHMARKS[@]}"; do

  BUILD_DIR="${DIR}/build-${BENCHMARK}"
  BIN_DIR=${BUILD_DIR}/bin/

  # 将数组改为字符串
  THREAD_STR="1 4 8"

  # COMPILER_STR="triton gcc clang"
  COMPILER_STR="clang triton"

  # 将字符串转换为数组
  THREAD=($THREAD_STR)
  COMPILER=($COMPILER_STR)

  for compiler in ${COMPILER[@]}; do
    for f_sub in `ls ${BIN_DIR}/${compiler}`; do
      ### FIXME: Check whether is a kernel directory
      kernel_dir=${BIN_DIR}/${compiler}/${f_sub}
      echo "${kernel_dir}"
      if [ ! -d "${kernel_dir}" ];then
          continue
      fi

      kernel_name=`basename ${f_sub}`
      echo ${kernel_name}

      # shape array
      # NOTE: get from config
      source ${kernel_dir}/${kernel_name}.cfg

      for thread in ${THREAD[@]}; do
        for shape in ${SHAPE[@]}; do
          for kernel in `ls -v ${kernel_dir}/${kernel_name}*.elf`; do
            echo ${kernel}
            tmp=`basename ${kernel} .elf`
            block_shape=${tmp#*_}
            DB_FILE=${BUILD_DIR}/${kernel_name} TRITON_CPU_MAX_THREADS=${thread} bash -c "${kernel} ${shape}" 2> "${kernel_dir}/${tmp}_T${thread}_S${shape}.log"

          done
        done
      done

    done
  done
done

echo "Run all benchmarks completed."
