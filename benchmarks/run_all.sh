MODE="Benchmark"
DIR=`dirname $0`
LD_LIBRARY_PATH="${DIR}/openmp-sysroot-riscv/lib:$LD_LIBRARY_PATH"

BENCHMARKS=("matmul" "softmax" "correlation" "layernorm"  "dropout" "rope" "resize")
# BENCHMARKS=("softmax")
