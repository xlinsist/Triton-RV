#!/bin/bash

# 获取当前脚本的目录
DIR=$(dirname $(readlink -f "$0"))

# 加载全局配置
if [ ! -f "${DIR}/global_config.sh" ]; then    
  echo "Error: global_config.sh not found!"    
  exit 1
fi
source "${DIR}/global_config.sh"

# 将基准测试列表转换为数组
IFS=' ' read -r -a BENCHMARKS <<< "$BENCHMARKS_LIST"

echo "Copying openmp-sysroot-riscv to ${REMOTE_BASE}"
rsync -avz --ignore-existing \
  "${DIR}/openmp-sysroot-riscv/" \
  "${REMOTE_URL}:${REMOTE_BASE}/openmp-sysroot-riscv/"

for BENCHMARK in "${BENCHMARKS[@]}"; do
  REMOTE_BIN_BASE="${REMOTE_BASE}/build-${BENCHMARK}/bin"
  LOCAL_BIN_DIR="${DIR}/build-${BENCHMARK}/bin"
  
  for SUBDIR in "${SUBDIRS[@]}"; do
    LOCAL_DIR="${LOCAL_BIN_DIR}/${SUBDIR}/${BENCHMARK}"
    REMOTE_DIR="${REMOTE_URL}:${REMOTE_BIN_BASE}/${SUBDIR}/${BENCHMARK}"
    REMOTE_FULL_PATH="${REMOTE_BIN_BASE}/${SUBDIR}/${BENCHMARK}"
    
    # 创建并清空远程目录
    ssh "${REMOTE_URL}" "mkdir -p ${REMOTE_FULL_PATH} && rm -rf ${REMOTE_FULL_PATH}/*"
    
    # 复制.elf文件
    if ls "${LOCAL_DIR}"/*.{elf,cfg} >/dev/null 2>&1; then
      echo "Copying ${LOCAL_DIR}/*.{elf,cfg} to ${REMOTE_DIR}"
      scp -r "${LOCAL_DIR}"/*.{elf,cfg} "${REMOTE_DIR}/"
    else
      echo "Warning: No .elf or .cfg files found in ${LOCAL_DIR}"
    fi
  done
done

echo "Copying global_config.sh to ${REMOTE_BASE}"
scp "${DIR}/global_config.sh" "${REMOTE_URL}:${REMOTE_BASE}/"

echo "Copying run.sh to ${REMOTE_BASE}"
scp "${DIR}/run.sh" "${REMOTE_URL}:${REMOTE_BASE}/"

echo "Copying report.sh to ${REMOTE_BASE}"
scp "${DIR}/report.sh" "${REMOTE_URL}:${REMOTE_BASE}/"

echo "All files processed."
