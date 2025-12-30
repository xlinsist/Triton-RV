#!/bin/bash

DIR=$(dirname $(readlink -f "$0"))

# ==========================================
# 1. 加载全局配置
# ==========================================
if [ -f "${DIR}/global_config.sh" ]; then    
  source "${DIR}/global_config.sh"
else    
  echo "Error: global_config.sh not found!"    
  exit 1
fi

# 允许命令行参数覆盖 global_config
DO_CLEAN=${DO_CLEAN:-$DEFAULT_DO_CLEAN}
BENCHMARKS=($BENCHMARKS_LIST)
export LD_LIBRARY_PATH

echo "--------------------------------------------------"
echo "Platform: $PLATFORM"
echo "Benchmarks: ${BENCHMARKS[@]}"
echo "Target: Triton Only"
echo "--------------------------------------------------"

# ==========================================
# 2. 用户接口 (Help & Args)
# ==========================================
help() {
cat <<END
Build Triton-Benchmark (Triton Only).

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
# 3. 工具链配置 (Clang/LLVM for Triton)
# ==========================================

AR="${LLVM_BUILD_DIR}/bin/llvm-ar"
AS="${LLVM_BUILD_DIR}/bin/llvm-as"
PYC="python"

case $PLATFORM in
    x86)
      CXX="${CLANG_BUILD_DIR}/bin/clang++ ${X86_FLAGS} ${CPP_STD}"
      OBJDUMP="${GCC_X86_BUILD_DIR}/bin/objdump"
      ;;
    rv)
      CXX="${CLANG_BUILD_DIR}/bin/clang++ --target=riscv64-unknown-linux-gnu \
              --sysroot=${RISCV_GNU_TOOLCHAIN_DIR}/sysroot \
              --gcc-toolchain=${RISCV_GNU_TOOLCHAIN_DIR} \
              ${RV_FLAGS} ${CPP_STD}"
      OBJDUMP="${RISCV_GNU_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-objdump"
      ;;
    ?*)
      echo "Unknown platform option: $PLATFORM"
      exit -1
      ;;
esac

# ==========================================
# 4. 核心构建函数
# ==========================================

# 编译基础支持库
build_support_lib() {
  local obj_dir=$1
  local lib_dir=$2
  
  echo "  -> Building support lib..."
  ${CXX} -fPIC -I ${DIR}/include -c ${SRC_DIR}/support/support.cpp -o ${obj_dir}/support.o
  ${OBJDUMP} -d ${obj_dir}/support.o &> ${obj_dir}/support.s
  ${AR} rcs ${lib_dir}/libsupport.a ${obj_dir}/support.o
}

# 编译 Triton Kernel 并链接 Driver
build_triton_benchmark() {
  local benchmark_name=$1
  local triton_kernel_file=$2
  local driver_file=$3
  local tunning_arg=$4

  # 定义构建目录结构
  local TYPE="triton"
  local LIB_DIR=${BUILD_DIR}/lib/${TYPE}
  local BIN_DIR=${BUILD_DIR}/bin/${TYPE}
  local OBJ_DIR=${BUILD_DIR}/obj/${TYPE}
  
  # Kernel 生成的相关路径
  local KERNEL_LAUNCHER_INCLUDE_DIR=${BUILD_DIR}/aux/include
  local KERNEL_AUX_FILE_DIR=${BUILD_DIR}/aux/src/${tunning_arg}

  # 清理逻辑
  if [ "x$DO_CLEAN" = "x--clean" ]; then
    rm -rf $BIN_DIR $LIB_DIR $OBJ_DIR
    rm -rf ${BUILD_DIR}/aux/
  fi

  # 创建目录
  mkdir -p ${LIB_DIR} ${BIN_DIR} ${OBJ_DIR} ${KERNEL_LAUNCHER_INCLUDE_DIR} ${KERNEL_AUX_FILE_DIR}
  
  # RISC-V 特殊依赖拷贝
  if [ "${PLATFORM}" == "rv" ] && [ -d "./openmp-sysroot-riscv/lib" ]; then
    cp ./openmp-sysroot-riscv/lib/* ${LIB_DIR}
  fi

  # 1. 编译 Support Lib
  build_support_lib "${OBJ_DIR}" "${LIB_DIR}"

  # 2. 调用 Python 生成 Kernel 代码
  echo "  -> Generating kernels from ${triton_kernel_file}..."
  
  ENABLE_AUTOTUNING=${tunning_arg} \
  KERNEL_LAUNCHER_INCLUDE_DIR=${KERNEL_LAUNCHER_INCLUDE_DIR} \
  KERNEL_AUX_FILE_DIR=${KERNEL_AUX_FILE_DIR} \
  ${PYC} ${triton_kernel_file}

  local kernel_cpp_name=$(basename ${driver_file} .cpp)
  local kernel_py_name=$(basename ${triton_kernel_file} .py)
  [[ "$kernel_cpp_name" == "$kernel_py_name" ]] || { echo "Error: cpp name and py name mismatch"; exit 1; }
  local kernel_bin_dir=${BIN_DIR}/${kernel_py_name}
  mkdir -p ${kernel_bin_dir}

  cp ${SRC_DIR}/main/${kernel_py_name}.cfg  ${kernel_bin_dir}

  # 3. 多线程编译生成的 Kernel 中间代码
  echo "  -> Compiling kernels and linking..."
  
  # 初始化多线程文件描述符
  [ -e /tmp/fd1_triton ] || mkfifo /tmp/fd1_triton
  exec 6<>/tmp/fd1_triton
  rm -rf /tmp/fd1_triton
  for ((i=1;i<=$MAX_MULTITHREADING;i++)); do echo >&6; done

  # 遍历生成的调优目录
  # 注意：这里假设 Python 脚本生成了对应的文件夹
  shopt -s nullglob # 防止没匹配到文件时报错
  for tunning_dir in ${KERNEL_AUX_FILE_DIR}_*; do
    read -u6
    {
        block_shape=${tunning_dir#*${tunning_arg}_}
        local current_obj_dir=${OBJ_DIR}/${kernel_py_name}_${tunning_arg}_${block_shape}
        mkdir -p ${current_obj_dir}
        
        # 修正生成的 IR 代码
        sed -i 's/trunc nuw nsw/trunc/g; s/trunc nuw/trunc/g; s/trunc nsw/trunc/g' ${tunning_dir}/*.llir

        # A. 编译 .llir -> .bc -> .o
        for kernel_ir in ${tunning_dir}/*.llir; do
            kname=$(basename ${kernel_ir} .llir)
            ${AS} -o ${KERNEL_AUX_FILE_DIR}_${block_shape}/${kname}.bc ${kernel_ir}
            ${CXX} -c ${KERNEL_AUX_FILE_DIR}_${block_shape}/${kname}.bc \
                -o ${current_obj_dir}/${kname}.o
            ${OBJDUMP} -d ${current_obj_dir}/${kname}.o \
                 &> ${KERNEL_AUX_FILE_DIR}_${block_shape}/${kname}.s
        done

        # B. 编译 launcher.cpp -> .o
        for kernel_launcher in ${tunning_dir}/*.cpp; do
            lname=$(basename ${kernel_launcher} .cpp)
            ${CXX} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -c ${kernel_launcher} \
                -fopenmp -o ${current_obj_dir}/${lname}.o
        done

        # C. 链接为静态库 (libkernel_xxx.a)
        find ${current_obj_dir}/ -not -name "support.o" -name "*.o" | \
            xargs ${AR} rcs ${LIB_DIR}/libkernel_${tunning_arg}_${block_shape}.a
        
        # D. 最终链接生成可执行文件 (ELF)
        ${CXX} ${driver_file} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} \
            -L ${LIB_DIR} -fopenmp -L${CLANG_BUILD_DIR}/lib \
            -lkernel_${tunning_arg}_${block_shape} -lsupport -latomic \
            ${CPP_STD} -DTRITON_KERNEL_ENABLE -fPIC \
            -o ${kernel_bin_dir}/${kernel_py_name}_${tunning_arg}_${block_shape}.elf
        
        echo >&6
    } &
  done
  wait
  exec 6>&-
  shopt -u nullglob
}

# ==========================================
# 5. 主循环 (Main Execution)
# ==========================================

for BENCHMARK in "${BENCHMARKS[@]}"; do
  echo ">>> Processing Benchmark: $BENCHMARK"

  # 路径配置
  TRITON_KERNEL="${SRC_DIR}/triton/${BENCHMARK}.py"
  DRIVER="${SRC_DIR}/main/${BENCHMARK}.cpp"

  # 映射 Kernel Tuning Name
  case "$BENCHMARK" in
    "matmul")      TUNNING_ARG="matmul_kernel" ;;
    "softmax")     TUNNING_ARG="softmax_kernel" ;;
    "layernorm")   TUNNING_ARG="_layer_norm_fwd_fused" ;;
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

  # 检查必要文件
  if [[ ! -f "$TRITON_KERNEL" ]]; then
     echo "Triton kernel file for $BENCHMARK not found at $TRITON_KERNEL, skipping."
     continue
  fi

  # 设置当前 benchmark 的构建根目录
  BUILD_DIR="${DIR}/build-${BENCHMARK}"
  
  # 执行构建
  build_triton_benchmark "$BENCHMARK" "$TRITON_KERNEL" "$DRIVER" "$TUNNING_ARG"

  echo "<<< Finished $BENCHMARK"
  echo ""
done
