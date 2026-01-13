#ifdef TRITON_KERNEL_ENABLE
#include "matmul_kernel_launcher.h"
#endif

#include "support/support.h"
#include <vector>
#include <cmath>
#include <iostream>

int main(int argc, char *argv[]) {
  // 1. 默认参数
  int M = 32, N = 32, K = 32;
  int RUN_COUNT = 10;

  // 2. 解析参数
  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);
    if (Shape.size() == 4) {
      M = Shape[0]; N = Shape[1]; K = Shape[2]; RUN_COUNT = Shape[3];
    } else {
        std::cerr << "Invalid shape format: MxNxKxRUN_COUNT" << std::endl;
        return -1;
    }
  }

  printf("Matmul Data shape %dx%dx%dx%d\n", M, N, K, RUN_COUNT);

  // 3. 资源分配与初始化 (使用 vector 替代 malloc)
  std::vector<float> arg0(M * K);
  std::vector<float> arg1(K * N);
  std::vector<float> real_out(M * N, 0.0f); // 初始化为0

  random_init(arg0);
  random_init(arg1);

#ifdef TRITON_KERNEL_ENABLE
  // 4. 定义 Grid 计算逻辑和 Kernel 调用
  // Grid 的计算通常与 Problem Size 相关，保留在 host 程序中
  int grid_M = std::ceil(1.0 * M / matmul_kernel_BLOCK_SIZE_M);
  int grid_N = std::ceil(1.0 * N / matmul_kernel_BLOCK_SIZE_N);
  int grid_size = grid_M * grid_N;

  // 5. 调用通用 Benchmark
  // 使用 Lambda 捕获所有需要的变量
  benchmark_kernel(RUN_COUNT, [&]() {
    matmul_kernel_omp(
        grid_size,  // gridX
        1, 1,       // gridY, gridZ
        matmul_kernel, 
        arg0.data(), arg1.data(), real_out.data(), // 获取原始指针
        M, N, K,    // Dimensions
        K, 1,       // Strides for arg0 (MxK) -> row major: stride_row=K, stride_col=1
        N, 1,       // Strides for arg1 (KxN) -> row major: stride_row=N, stride_col=1
        N, 1        // Strides for out  (MxN)
    );
  });
#else
  std::cerr << "Triton Kernel Not Enabled!" << std::endl;
#endif

  // vector 自动析构，无需 free
  return 0;
}
