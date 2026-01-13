#!/bin/bash

# This script runs the benchmarks on the target machine (local or remote).
# It iterates through all enabled benchmarks, compilers, thread counts, and shapes.

# Get the directory of the current script.
# On the remote machine, this will be the directory where run.sh was copied.
DIR=$(dirname "$(readlink -f "$0")")

# Load global configuration.
if [ ! -f "${DIR}/global_config.sh" ]; then
  echo "Error: global_config.sh not found!"
  exit 1
fi
source "${DIR}/global_config.sh"

# Prepend the OpenMP sysroot library path for RISC-V.
if [ "$PLATFORM" = "rv" ]; then
    export LD_LIBRARY_PATH="${DIR}/openmp-sysroot-riscv/lib:$LD_LIBRARY_PATH"
fi

# Convert space-separated lists from config into bash arrays.
IFS=' ' read -r -a BENCHMARKS <<< "$BENCHMARKS_LIST"
IFS=' ' read -r -a THREADS <<< "$THREADS_LIST"

# Determine which compilers are enabled.
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

if [ ${#COMPILERS[@]} -eq 0 ]; then
  echo "Warning: No compilers enabled in global_config.sh. Nothing to run."
  exit 0
fi

echo "Starting benchmark run..."

# Iterate over each benchmark.
for BENCHMARK in "${BENCHMARKS[@]}"; do
  echo "--- Processing Benchmark: ${BENCHMARK} ---"
  BUILD_DIR="${DIR}/build-${BENCHMARK}"
  BIN_DIR="${BUILD_DIR}/bin"

  # Iterate over each enabled compiler.
  for COMPILER in "${COMPILERS[@]}"; do
    echo "  - Compiler: ${COMPILER}"
    
    # The compiled kernels are in subdirectories, e.g., bin/triton/matmul/
    for kernel_dir in "${BIN_DIR}/${COMPILER}/"*/; do
      if [ ! -d "${kernel_dir}" ]; then
        echo "    Warning: No kernel directories found in ${BIN_DIR}/${COMPILER}/. Skipping."
        continue
      fi

      kernel_name=$(basename "$kernel_dir")
      echo "    - Kernel: ${kernel_name}"

      # Load the shape configuration file (e.g., matmul.cfg) for this kernel.
      if [ ! -f "${kernel_dir}/${kernel_name}.cfg" ]; then
          echo "      Error: Shape config file ${kernel_dir}/${kernel_name}.cfg not found. Skipping."
          continue
      fi
      source "${kernel_dir}/${kernel_name}.cfg"

      # Iterate over all thread counts.
      for THREAD in "${THREADS[@]}"; do
        # Iterate over all shapes defined in the .cfg file.
        for shape in "${SHAPE[@]}"; do
          # Iterate over all executable .elf files for this kernel.
          for kernel_elf in "${kernel_dir}/${kernel_name}"*.elf; do
            if [ ! -x "${kernel_elf}" ]; then
                echo "      Warning: Kernel file ${kernel_elf} not found or not executable. Skipping."
                continue
            fi
            
            echo "      - Running: $(basename "${kernel_elf}") with T=${THREAD}, S=${shape}"
            log_file="${kernel_dir}/$(basename "${kernel_elf}" .elf)_T${THREAD}_S${shape}.log"

            # Execute the benchmark.
            DB_FILE="${BUILD_DIR}/${kernel_name}" \
            TRITON_CPU_MAX_THREADS=${THREAD} \
            bash -c "${kernel_elf} ${shape}" > "${log_file}" 2>&1
          done
        done
      done
    done
  done
done

echo "---"
echo "Benchmark run completed."
