# Triton-RV

## Overview

This repo conducts a comparative analysis of runtime auto-tuning in Triton-CPU across multi-core x86 and RISC-V architectures, benchmarking it against compiler-based auto-tuning approaches to systematically evaluate the impact of various tuning parameters.
1. We introduced cross-platform capabilities to the toolchain, enabling seamless support for both x86 and RISC-V architectures.
2. We developed a visualization module that converts benchmark results into CSV format for streamlined data analysis.
3. We bumped this repo to a Triton-CPU version (2fa1c59), which introduces critical enhancements for CPU performance including Sleef math library integration and optimized multi-threading support.

## Environmental Setup of triton-cpu

### **1. Clone and Initialize**

```sh
$ git clone git@github.com:xlinsist/Triton-RV.git
$ cd Triton-RV
$ git submodule update --init
```

### **2. Build LLVM**

```sh
$ cd benchmarks
$ cd ./llvm-project  # cloned as a submodule
$ git checkout 86b69c3 # In compliance with the triton-cpu version we bumped
$ mkdir build
$ cd build
$ cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang;openmp" \
        -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
$ ninja
```

### **3. build SLEEF**

SLEEF is a dependency of triton-cpu. Although documentation of triton-cpu does not mention the need for manual building, this step is essential to avoid runtime issues.

```sh
$ cd benchmarks/
$ cd ./triton-cpu # cloned as a submodule. Since triton-cpu is under development, this is a forked repo
$ git submodule update --init # clone SLEEF as submodule of triton-cpu
$ cd third_party/sleef
$ mkdir build # provided in the SLEEF README for building the project.
$ cmake -S . -B build
$ cmake --build build -j --clean-first
```

### **4. Edit and Build triton-cpu**

```sh
$ cd benchmarks
$ cd ./triton-cpu # cloned as a submodule. Since triton-cpu is under development, this is a forked repo
$ git checkout Triton-RV # the version of triton-cpu we bumped currently
$ git apply ../patch/triton-cpu-0001-driver.patch
$ git apply ../patch/triton-cpu-0002-autotuning.patch
$ export LLVM_BUILD_DIR=../llvm-project/build
$ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
         LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
         LLVM_SYSPATH=$LLVM_BUILD_DIR \
         pip install -e python
```
> WARNING: Currently the patches applied to triton-cpu may have correctness issues that need to be fixed. We are working on it.

### **5. Modify global configuration**

modify `global_config.sh` before running. For instance, change the following variables:
```sh
$ export CLANG_BUILD_DIR="${DIR_PATH}/llvm-project/build" # Since Clang is built along with LLVM, this path can be used directly.
$ export GCC_X86_BUILD_DIR="/usr" # By default, the system-installed GCC is used; modify as needed.
```

## Running on x86

```sh
$ cd benchmarks
$ ./build.sh  # Customize sections marked with "Make your changes here if you need," including method, benchmark, and toolchain paths.
$ ./run.sh
$ ./report.sh
```
Check the `performance_report_overall.csv` generated under `benchmarks` directory.

## Cross-compiling for RISC-V

### **1. Preparing GCC**

To build on a RISC-V platform, the riscv-gnu-toolchain is required. You can download a precompiled package from: [https://archive.spacemit.com/toolchain/](https://archive.spacemit.com/toolchain/).

Set up the environment variable as follows:

```sh
$ export RISCV_GNU_TOOLCHAIN_DIR=<path-to-your-spacemit-toolchain-linux-glibc-x86_64-v1.0.1>
```

### **2. Preparing Clang**

To ensure the Clang version matches the llvm version used by triton-cpu, it is recommended to build it from source.

> **Note:** Please clone a separate `llvm-project` instead of reusing the one that `triton-cpu` depends on. This is because when compiling Clang from source, `-DLLVM_TARGETS_TO_BUILD` must include `RISCV` to support cross-compilation. Recompiling the existing `llvm-project` used by `triton-cpu` may result in runtime errors, such as `LLVMRISCVAsmParser` import failures.

```sh
$ git clone git@github.com:llvm/llvm-project.git
$ mkdir llvm-project/build
$ cd llvm-project/build
$ git checkout 86b69c3 # Ensure it matches the version used by triton-cpu
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ ninja check-mlir check-clang
$ export CLANG_BUILD_DIR=<path-to-this-llvm-project>/build # For examples, export CLANG_BUILD_DIR=./llvm-project/build-86b69c-rv
```
### **3. Modify global configuration**
modify `global_config.sh`, set the environment variables above before running.

### **4. Running on a RISC-V Platform**

First, use `build.sh` to cross-compile and generate ELF files locally. Then, transfer them to a remote RISC-V machine for execution using `run.sh`. Finally, copy the output directory back to the local machine and run `report.sh` to generate performance results.

```sh
$ cd benchmarks
$ ./build.sh # modify `global_config.sh` before running
$ ./copy_to_remote.sh # Modify REMOTE IP and file paths accordingly.
$ <Use SSH to connect to the REMOTE IP>

// On the remote RISC-V machine:
# <Navigate to the correct directory>
# ./run.sh
# ./report.sh
# exit // Exit the remote session

// Back on the local machine:
$ ./copy_remote_back.sh # Modify REMOTE IP and file paths accordingly.
$ python get_data.py
```

Check the `performance_report_overall.csv` generated under `benchmarks` directory.

## Output Examples

The generated `performance_report_overall.csv` after running `report.sh` as demonstrated above should look like this:
```
benchmark,shape,method,thread,parameter,time(s),speedup
...
matmul,"(32, 32, 32, 10)",clang,1,,0.0247168,1.0
matmul,"(32, 32, 32, 10)",clang,4,,0.0150131,1.6463
matmul,"(32, 32, 32, 10)",clang,8,,0.0303547,0.8143
matmul,"(32, 32, 32, 10)",clang,16,,0.0151946,1.6267
matmul,"(32, 32, 32, 10)",clang,32,,0.0207143,1.1932
matmul,"(32, 32, 32, 10)",clang,64,,0.047209,0.5236
matmul,"(32, 32, 32, 10)",triton_tuned,1,"(32, 32, 8)",0.00235865,10.4792
matmul,"(32, 32, 32, 10)",triton_tuned,4,"(32, 32, 8)",0.00205009,12.0564
matmul,"(32, 32, 32, 10)",triton_tuned,8,"(32, 32, 4)",0.00259357,9.53
matmul,"(32, 32, 32, 10)",triton_tuned,16,"(32, 32, 16)",0.0021222,11.6468
matmul,"(32, 32, 32, 10)",triton_tuned,32,"(32, 32, 16)",0.00173468,14.2486
matmul,"(32, 32, 32, 10)",triton_tuned,64,"(32, 32, 16)",0.00184007,13.4325
matmul,"(64, 64, 64, 10)",clang,1,,0.0178245,1.0
matmul,"(64, 64, 64, 10)",clang,4,,0.0146382,1.2177
matmul,"(64, 64, 64, 10)",clang,8,,0.0135641,1.3141
matmul,"(64, 64, 64, 10)",clang,16,,0.0143403,1.243
matmul,"(64, 64, 64, 10)",clang,32,,0.0202123,0.8819
matmul,"(64, 64, 64, 10)",clang,64,,0.0276551,0.6445
matmul,"(64, 64, 64, 10)",triton_tuned,1,"(32, 32, 32)",0.0125893,1.4158
matmul,"(64, 64, 64, 10)",triton_tuned,4,"(4, 32, 4)",0.0124491,1.4318
matmul,"(64, 64, 64, 10)",triton_tuned,8,"(8, 4, 8)",0.0132844,1.3418
matmul,"(64, 64, 64, 10)",triton_tuned,16,"(8, 32, 8)",0.013894,1.2829
matmul,"(64, 64, 64, 10)",triton_tuned,32,"(8, 32, 32)",0.016862,1.0571
matmul,"(64, 64, 64, 10)",triton_tuned,64,"(16, 4, 4)",0.016894,1.0551
matmul,"(128, 128, 128, 10)",clang,1,,0.0401563,1.0
matmul,"(128, 128, 128, 10)",clang,4,,0.0398675,1.0072
matmul,"(128, 128, 128, 10)",clang,8,,0.0158121,2.5396
matmul,"(128, 128, 128, 10)",clang,16,,0.0194217,2.0676
matmul,"(128, 128, 128, 10)",clang,32,,0.0238589,1.6831
matmul,"(128, 128, 128, 10)",clang,64,,0.0553912,0.725
matmul,"(128, 128, 128, 10)",triton_tuned,1,"(32, 16, 4)",0.0164989,2.4339
matmul,"(128, 128, 128, 10)",triton_tuned,4,"(32, 8, 32)",0.0129716,3.0957
matmul,"(128, 128, 128, 10)",triton_tuned,8,"(16, 32, 8)",0.0136829,2.9348
matmul,"(128, 128, 128, 10)",triton_tuned,16,"(8, 8, 4)",0.01424,2.82
matmul,"(128, 128, 128, 10)",triton_tuned,32,"(32, 8, 32)",0.0152967,2.6252
matmul,"(128, 128, 128, 10)",triton_tuned,64,"(16, 4, 32)",0.0167945,2.391
...
```
.
