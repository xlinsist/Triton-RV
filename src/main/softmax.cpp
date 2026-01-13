#ifdef TRITON_KERNEL_ENABLE
#include "softmax_kernel_launcher.h"
#endif

#include "support/support.h"
#include <vector>
#include <iostream>
#include <cassert>

int main(int argc, char *argv[]) {
  // 1. 默认参数
  int R = 32, C = 64;
  int RUN_COUNT = 10;

  // 2. 解析参数
  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);
    if (Shape.size() == 3) {
      R = Shape[0]; C = Shape[1]; RUN_COUNT = Shape[2];
    } else {
        std::cerr << "Invalid shape format: RxCxRUN_COUNT" << std::endl;
        return -1;
    }
  }

  printf("Softmax Data shape %dx%dx%d\n", R, C, RUN_COUNT);
  assert(R != 0 && C != 0 && "Invalid shape\n");

  // 3. 资源分配与初始化
  std::vector<float> input(R * C);
  std::vector<float> real_out(R * C, 0.0f);

  random_init(input);

#ifdef TRITON_KERNEL_ENABLE
  // 4. 调用通用 Benchmark
  benchmark_kernel(RUN_COUNT, [&]() {
    // Softmax kernel 通常以 Row 数量作为 grid X
    softmax_kernel_omp(
        R, 1, 1,        // grid dims
        &softmax_kernel, 
        real_out.data(), input.data(), // pointers
        C,              // n_cols
        C,              // stride_row_in (假设 contiguous)
        C               // stride_row_out
    );
  });
#else
  std::cerr << "Triton Kernel Not Enabled!" << std::endl;
#endif

  return 0;
}
