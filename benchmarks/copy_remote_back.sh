#!/bin/bash

DIR=`dirname $0`
# REMOTE_URL="user@192.168.15.167" # 根据远程平台做修改
REMOTE_URL="user@192.168.15.175" # 根据远程平台做修改

BENCHMARKS=("matmul" "softmax" "correlation" "layernorm"  "dropout" "rope" "resize")

for BENCHMARK in "${BENCHMARKS[@]}"; do
    REMOTE_BASE="/home/user/triton-benchmark/build-rv/build-${BENCHMARK}/report.xls" # 根据远程平台做修改
    REMOTE="${REMOTE_URL}:${REMOTE_BASE}"

    BUILD_DIR=${DIR}/build-${BENCHMARK}/

    scp ${REMOTE} ${BUILD_DIR}
done
