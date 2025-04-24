#!/bin/bash

DIR="$(dirname "$0")"
REMOTE_BASE="/home/user/triton-benchmark/build-rv-0423"
# BENCHMARKS=("layernorm" "correlation" "resize")
BENCHMARKS=("rope")
SUBDIRS=("triton" "clang")

for BENCHMARK in "${BENCHMARKS[@]}"; do
  REMOTE_BIN_BASE="${REMOTE_BASE}/build-${BENCHMARK}/bin"
  LOCAL_BIN_DIR="${DIR}/build-${BENCHMARK}/bin"
  
  for SUBDIR in "${SUBDIRS[@]}"; do
    LOCAL_DIR="${LOCAL_BIN_DIR}/${SUBDIR}/${BENCHMARK}"
    REMOTE_DIR="user@192.168.15.167:${REMOTE_BIN_BASE}/${SUBDIR}/${BENCHMARK}"
    REMOTE_FULL_PATH="${REMOTE_BIN_BASE}/${SUBDIR}/${BENCHMARK}"
    
    # 创建并清空远程目录
    ssh user@192.168.15.167 "mkdir -p ${REMOTE_FULL_PATH} && rm -rf ${REMOTE_FULL_PATH}/*"
    
    # 复制.elf文件
    if ls "${LOCAL_DIR}"/*.{elf,cfg} >/dev/null 2>&1; then
      echo "Copying ${LOCAL_DIR}/*.{elf,cfg} to ${REMOTE_DIR}"
      scp -r "${LOCAL_DIR}"/*.{elf,cfg} "${REMOTE_DIR}/"
    else
      echo "Warning: No .elf or .cfg files found in ${LOCAL_DIR}"
    fi
  done
done
