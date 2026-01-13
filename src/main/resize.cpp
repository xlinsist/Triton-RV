#ifdef TRITON_KERNEL_ENABLE
#include "resize_kernel_launcher.h"
#endif

#include "support/support.h"
#include <vector>
#include <iostream>
#include <cassert>

int main(int argc, char *argv[]) {
  // 1. 默认参数
  int H = 128, W = 128, C = 3;
  int RUN_COUNT = 10;

  // 2. 解析参数
  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);
    if (Shape.size() == 4) {
      H = Shape[0]; W = Shape[1]; C = Shape[2]; RUN_COUNT = Shape[3];
    } else {
        std::cerr << "Invalid shape format: HxWxCxRUN_COUNT" << std::endl;
        return -1;
    }
  }

  printf("Resize Data shape H:%d W:%d C:%d\n", H, W, C);

  // 3. 初始化数据 (int8)
  size_t input_size = (size_t)H * W * C;
  // 输出是输入的 2x2 倍大小 (Nearest Neighbor Upsampling x2)
  size_t output_size = (size_t)H * 2 * W * 2 * C * 2; // 注意：原代码逻辑似乎隐含了深度也变了？或者是内存分配预留？
  // 根据原代码：H * 2 * W * 2 * C * 2，这里严格遵循原代码分配大小
  
  std::vector<int8_t> input(input_size);
  std::vector<int8_t> real_out(output_size, 0);

  random_init(input, 0, 255);

#ifdef TRITON_KERNEL_ENABLE
  // 4. Grid 设置
  int gridX = H * 2;
  int gridY = C;
  int gridZ = 1;

  // 5. Benchmark
  benchmark_kernel(RUN_COUNT, [&]() {
    resize_kernel_omp(
        gridX, gridY, gridZ, 
        &resize_kernel, 
        input.data(), 
        real_out.data(), 
        C, H, W
    );
  });
#else
  std::cerr << "Triton Kernel Not Enabled!" << std::endl;
#endif

  return 0;
}
