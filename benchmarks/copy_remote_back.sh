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

for BENCHMARK in "${BENCHMARKS[@]}"; do
    REMOTE_BASE="${REMOTE_BASE}/build-${BENCHMARK}/report.xls"
    REMOTE="${REMOTE_URL}:${REMOTE_BASE}"

    BUILD_DIR=${DIR}/build-${BENCHMARK}/

    scp ${REMOTE} ${BUILD_DIR}
done
echo "All reports copied from remote platform."
