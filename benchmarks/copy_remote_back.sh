#!/bin/bash

DIR=`dirname $0`
# REMOTE_URL="user@192.168.15.167" # 根据远程平台做修改
REMOTE_URL="user@192.168.15.175" # 根据远程平台做修改

# 将数组改为字符串
BENCHMARKS_STR="matmul softmax correlation layernorm dropout rope resize"

# 将字符串转换为数组进行循环
BENCHMARKS=($BENCHMARKS_STR)

for BENCHMARK in "${BENCHMARKS[@]}"; do
    REMOTE_BASE="/home/user/triton-benchmark/build-rv/build-${BENCHMARK}/report.xls" # 根据远程平台做修改
    REMOTE="${REMOTE_URL}:${REMOTE_BASE}"

    BUILD_DIR=${DIR}/build-${BENCHMARK}/

    scp ${REMOTE} ${BUILD_DIR}
done
echo "All reports copied from remote platform."
