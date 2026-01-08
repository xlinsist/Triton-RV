#ifdef TRITON_KERNEL_ENABLE
#include "dropout_kernel_launcher.h"
#endif

#include "support/support.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

int main(int argc, char *argv[]) {
  // 1. 默认参数
  int N = 4096;
  int RUN_COUNT = 10;
  float ratio = 0.5f;
  int seed = 1234;

  // 2. 解析参数
  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);
    if (Shape.size() == 2) {
      N = Shape[0]; RUN_COUNT = Shape[1];
    } else {
        std::cerr << "Invalid shape format: NxRUN_COUNT" << std::endl;
        return -1;
    }
  }

  printf("Dropout Data shape N:%d\n", N);

  // 3. 初始化
  std::vector<float> input(N);
  std::vector<float> real_out(N, 0.0f);

  random_init(input, -1.0, 1.0);

#ifdef TRITON_KERNEL_ENABLE
  // 4. Grid
  // 假设 BLOCK_SIZE 已在头文件中定义
  int grid = std::ceil((float)N / dropout_kernel_BLOCK_SIZE);

  // 5. Benchmark
  benchmark_kernel(RUN_COUNT, [&]() {
    dropout_kernel_omp(
        grid, 1, 1, 
        &dropout_kernel, 
        input.data(), 
        real_out.data(), 
        N, ratio, seed
    );
  });

  // 6. 简单的 Ratio 验证 (保留原代码逻辑)
  int zero_count = 0;
  for (float val : real_out) {
      if (val == 0.0f) zero_count++;
  }
  printf("Triton Dropout Actual Ratio: %.3f (Expected: %.3f)\n", 
         (float)zero_count / N, ratio);

#else
  std::cerr << "Triton Kernel Not Enabled!" << std::endl;
#endif

  return 0;
}
