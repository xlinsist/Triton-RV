#!/bin/bash

DIR=$(dirname $(readlink -f "$0"))

# ==========================================
# 加载全局配置
# ==========================================
if [ -f "${DIR}/global_config.sh" ]; then    
  source "${DIR}/global_config.sh"
else    
  echo "Error: global_config.sh not found!"    
  exit 1
fi

# 允许命令行参数覆盖 global_config 中的设置
DO_CLEAN=${DO_CLEAN:-$DEFAULT_DO_CLEAN}

# 将配置中的字符串转为数组
BENCHMARKS=($BENCHMARKS_LIST)

export LD_LIBRARY_PATH
echo "--------------------------------------------------"
echo "Mode: $MODE"
echo "Platform: $PLATFORM"
echo "Benchmarks to run: ${BENCHMARKS[@]}"
echo "Build Targets: GCC=${ENABLE_GCC}, Clang=${ENABLE_CLANG}, Triton=${ENABLE_TRITON}"
echo "--------------------------------------------------"

# ==========================================
# 用户接口 (Help & Args)
# ==========================================
help()
{
cat <<END
Build Triton-Benchmark.

Usage: ./build.sh [--clean | --no-clean]
                  [--help]
                  [--platform x86 | rv]

Options:
  --clean | --no-clean    Override clean setting (Current: $DO_CLEAN)
  --platform <arch>       Override platform (Current: $PLATFORM)
  --help                  Print this help message
END
}

while [ $# -gt 0 ]; do
    case $1 in
        --clean | --no-clean) DO_CLEAN=$1 ;;
        --help | -h) help; exit 0 ;;
        --platform) PLATFORM=$2; shift ;;
        ?*) echo "Invalid options:\"$1\", try $0 --help"; exit 1 ;;
    esac
    shift
done

# ==========================================
# 工具链环境配置 (根据 Config 组装)
# ==========================================

AR="${LLVM_BUILD_DIR}/bin/llvm-ar"
AS="${LLVM_BUILD_DIR}/bin/llvm-as"
PYC="python"

case $PLATFORM in
    x86)
      CLANGPP="${CLANG_BUILD_DIR}/bin/clang++ ${X86_FLAGS} ${CPP_STD}"
      GCC="${GCC_X86_BUILD_DIR}/bin/g++ ${X86_FLAGS} ${CPP_STD}"
      OBJDUMP="${GCC_X86_BUILD_DIR}/bin/objdump"
      ;;
    rv)
      CLANGPP="${CLANG_BUILD_DIR}/bin/clang++ --target=riscv64-unknown-linux-gnu \
              --sysroot=${RISCV_GNU_TOOLCHAIN_DIR}/sysroot \
              --gcc-toolchain=${RISCV_GNU_TOOLCHAIN_DIR} \
              ${RV_FLAGS} ${CPP_STD}"
      GCC="${RISCV_GNU_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-g++ \
          ${RV_GCC_FLAGS} ${CPP_STD}"
      OBJDUMP="${RISCV_GNU_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-objdump"
      ;;
    ?*)
      echo "Unknown platform option: $PLATFORM"
      exit -1
      ;;
esac

# ==========================================
# 核心构建函数 (保持大部分逻辑不变)
# ==========================================

build_support_lib() {
  echo "  -> building support lib..."
  # echo "${COMPILER} -fPIC -I ${DIR}/include -c ${SRC_DIR}/support/support.cpp -o ${OBJ_DIR}/support.o"
  ${COMPILER} -fPIC -I ${DIR}/include -c ${SRC_DIR}/support/support.cpp -o ${OBJ_DIR}/support.o
  ${OBJDUMP} -d ${OBJ_DIR}/support.o &> ${OBJ_DIR}/support.s
  ${AR} rcs ${LIB_DIR}/libsupport.a ${OBJ_DIR}/support.o
}

build_c_kernel_lib_and_driver() {
  name=$(basename ${C_KERNEL} .cpp)
  echo "  -> building c kernel ${C_KERNEL}..."
  ${COMPILER} -fPIC -I ${DIR}/include -c ${C_KERNEL} -fopenmp -o ${OBJ_DIR}/${name}.o
  ${OBJDUMP} -d ${OBJ_DIR}/${name}.o &> ${OBJ_DIR}/${name}.s

  find ${OBJ_DIR} -not -name "support.o" -name "*.o" | xargs ${AR} rcs ${LIB_DIR}/libkernel.a

  name=$(basename ${DRIVER} .cpp)
  echo "  -> generating elf of ${DRIVER}..."
  KERNEL_BIN_DIR=${BIN_DIR}/${name}/
  mkdir -p ${KERNEL_BIN_DIR}

  ${COMPILER} "${DRIVER}" \
    -I "${DIR}/include" -I "${KERNEL_LAUNCHER_INCLUDE_DIR}" \
    -L "${LIB_DIR}" -fopenmp -lkernel -lsupport -latomic \
    ${CPP_STD} -L${CLANG_BUILD_DIR}/lib -D"${KERNEL_ENABLE}" -fPIC \
    -o "${KERNEL_BIN_DIR}/${name}.elf"
  
  ${OBJDUMP} -d ${KERNEL_BIN_DIR}/${name}.elf &> ${KERNEL_BIN_DIR}/${name}.elf.s
  cp ${SRC_DIR}/main/${name}.cfg ${KERNEL_BIN_DIR}
}

build_triton_kernel_lib_and_driver() {
  echo "  -> building triton kernel ${TRITON_KERNEL}..."
  name=$(basename ${TRITON_KERNEL} .py)
  KERNEL_AUX_FILE_DIR=${BUILD_DIR}/aux/src/${TUNNING_ARG}
  mkdir -p ${KERNEL_LAUNCHER_INCLUDE_DIR}
  mkdir -p ${KERNEL_AUX_FILE_DIR}

  ENABLE_AUTOTUNING=${TUNNING_ARG} \
  KERNEL_LAUNCHER_INCLUDE_DIR=${KERNEL_LAUNCHER_INCLUDE_DIR} \
  KERNEL_AUX_FILE_DIR=${KERNEL_AUX_FILE_DIR} \
  ${PYC} ${TRITON_KERNEL}

  driver_name=$(basename ${DRIVER} .cpp)
  KERNEL_BIN_DIR=${BIN_DIR}/${driver_name}
  mkdir -p ${KERNEL_BIN_DIR}
  cp ${SRC_DIR}/main/${driver_name}.cfg ${KERNEL_BIN_DIR}

  # Multi-thread compilation logic
  [ -e /tmp/fd1_hyc ] || mkfifo /tmp/fd1_hyc
  exec 6<>/tmp/fd1_hyc
  rm -rf /tmp/fd1_hyc
  for ((i=1;i<=$MAX_MULTITHREADING;i++)); do echo >&6; done

  for tunning_dir in ${KERNEL_AUX_FILE_DIR}_*; do
    read -u6
    {
        # echo "----------tuning ${tunning_dir}...-------------"
        block_shape=${tunning_dir#*${TUNNING_ARG}_}
        mkdir -p ${OBJ_DIR}/${name}_${TUNNING_ARG}_${block_shape}
        sed -i 's/trunc nuw nsw/trunc/g; s/trunc nuw/trunc/g; s/trunc nsw/trunc/g' ${tunning_dir}/*.llir

        # .llir -> .bc -> .o
        for kernel_ir in ${tunning_dir}/*.llir; do
            kernel_name=$(basename ${kernel_ir} .llir)
            ${AS} -o ${KERNEL_AUX_FILE_DIR}_${block_shape}/${kernel_name}.bc ${kernel_ir}
            ${COMPILER} -c ${KERNEL_AUX_FILE_DIR}_${block_shape}/${kernel_name}.bc \
                -o ${OBJ_DIR}/${name}_${TUNNING_ARG}_${block_shape}/${kernel_name}.o
            ${OBJDUMP} -d ${OBJ_DIR}/${name}_${TUNNING_ARG}_${block_shape}/${kernel_name}.o \
                 &> ${KERNEL_AUX_FILE_DIR}_${block_shape}/${kernel_name}.s
        done

        # launcher.cpp -> .o
        for kernel_launcher in ${tunning_dir}/*.cpp; do
            launcher_name=$(basename ${kernel_launcher} .cpp)
            ${COMPILER} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -c ${kernel_launcher} \
                -fopenmp -o ${OBJ_DIR}/${name}_${TUNNING_ARG}_${block_shape}/${launcher_name}.o
        done

        # Link kernel lib
        find ${OBJ_DIR}/${name}_${TUNNING_ARG}_${block_shape}/ -not -name "support.o" -name "*.o" | \
            xargs ${AR} rcs ${LIB_DIR}/libkernel_${TUNNING_ARG}_${block_shape}.a
        
        # Compile driver
        ${COMPILER} ${DRIVER} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} \
            -L ${LIB_DIR} -fopenmp -L${CLANG_BUILD_DIR}/lib \
            -lkernel_${TUNNING_ARG}_${block_shape} -lsupport -latomic \
            ${CPP_STD} -D${KERNEL_ENABLE} -fPIC \
            -o ${KERNEL_BIN_DIR}/${driver_name}_${TUNNING_ARG}_${block_shape}.elf
        
        echo >&6
    } &
  done
  wait
  exec 6>&-
}

create_dir_hierarchy(){
  mkdir -p ${LIB_DIR} ${BIN_DIR} ${OBJ_DIR}
  if [ "${PLATFORM}" == "rv" ]; then
    cp ./openmp-sysroot-riscv/lib/* ${LIB_DIR}
  fi
}

build_driver() {
  local toolchain=$1
  local benchmark_name=$2
  
  case $toolchain in
    gcc)    COMPILER=${GCC};     TYPE="gcc";;
    clang)  COMPILER=${CLANGPP}; TYPE="clang";;
    triton) COMPILER=${CLANGPP}; TYPE="triton";;
    *) echo "Unknown toolchain: $toolchain"; exit 1 ;;
  esac

  LIB_DIR=${BUILD_DIR}/lib/${TYPE}
  BIN_DIR=${BUILD_DIR}/bin/${TYPE}
  OBJ_DIR=${BUILD_DIR}/obj/${TYPE}
  
  if [ "$toolchain" == "triton" ]; then
    KERNEL_ENABLE=TRITON_KERNEL_ENABLE
  else
    KERNEL_ENABLE=C_KERNEL_ENABLE
  fi

  # Cleaning
  if [ "x$DO_CLEAN" = "x--clean" ]; then
    rm -rf $BIN_DIR $LIB_DIR $OBJ_DIR
    [ "${KERNEL_ENABLE}" == "TRITON_KERNEL_ENABLE" ] && rm -rf ${BUILD_DIR}/aux/
  fi

  create_dir_hierarchy
  if [ "${KERNEL_ENABLE}" == "TRITON_KERNEL_ENABLE" ]; then
    mkdir -p ${KERNEL_LAUNCHER_INCLUDE_DIR}
    mkdir -p ${BUILD_DIR}/aux/src
  fi

  build_support_lib

  if [ "${MODE}" == "Accuracy" ]; then COMPILER+=" -DCHECK_ACCURACY "; fi

  if [ "${KERNEL_ENABLE}" == "C_KERNEL_ENABLE" ]; then
    build_c_kernel_lib_and_driver
  else
    build_triton_kernel_lib_and_driver "$benchmark_name"
  fi
}

# ==========================================
# 主循环 (Main Execution)
# ==========================================

for BENCHMARK in "${BENCHMARKS[@]}"; do
  echo ">>> Processing Benchmark: $BENCHMARK"

  # 1. 自动推导文件路径 (Convention over Configuration)
  export C_KERNEL="${SRC_DIR}/c/${BENCHMARK}.cpp"
  export TRITON_KERNEL="${SRC_DIR}/triton/${BENCHMARK}.py"
  export DRIVER="${SRC_DIR}/main/${BENCHMARK}.cpp"

  # 2. 映射 Kernel Tuning Name (如果每个benchmark名字都不规则，只能在这里列出)
  case "$BENCHMARK" in
    "matmul")      TUNNING_ARG="matmul_kernel" ;;
    "softmax")     TUNNING_ARG="softmax_kernel" ;;
    "layernorm")   TUNNING_ARG="_layer_norm_fwd_fused" ;; # 特殊命名
    "correlation") TUNNING_ARG="correlation_kernel" ;;
    "dropout")     TUNNING_ARG="dropout_kernel" ;;
    "resize")      TUNNING_ARG="resize_kernel" ;;
    "rope")        TUNNING_ARG="rope_kernel" ;;
    "warp")        TUNNING_ARG="warp_kernel" ;;
    *) 
       echo "Warning: Unknown benchmark '$BENCHMARK', assuming kernel name is '${BENCHMARK}_kernel'"
       TUNNING_ARG="${BENCHMARK}_kernel"
       ;;
  esac
  export TUNNING_ARG

  # 检查文件是否存在
  if [[ ! -f "$C_KERNEL" ]] && [[ ! -f "$TRITON_KERNEL" ]]; then
     echo "Files for $BENCHMARK not found, skipping."
     continue
  fi

  BUILD_DIR="${DIR}/build-${BENCHMARK}"
  KERNEL_LAUNCHER_INCLUDE_DIR="${BUILD_DIR}/aux/include"

  # 3. 根据 global_config.sh 的开关执行构建
  if [ "${ENABLE_GCC}" == "1" ]; then
      echo "  [GCC] Building..."
      build_driver gcc $BENCHMARK
  fi

  if [ "${ENABLE_CLANG}" == "1" ]; then
      echo "  [Clang] Building..."
      build_driver clang $BENCHMARK
  fi

  if [ "${ENABLE_TRITON}" == "1" ]; then
      echo "  [Triton] Building..."
      build_driver triton $BENCHMARK
  fi
  
  echo "<<< Finished $BENCHMARK"
  echo ""

  # 清理环境变量，防止污染下一次循环
  unset C_KERNEL TRITON_KERNEL DRIVER TUNNING_ARG
done
