#ifdef TRITON_KERNEL_ENABLE
#include "correlation_kernel_launcher.h"
#endif

#include "support/support.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

#define OUT_SHIFT 0

int main(int argc, char *argv[]) {
  // 1. 默认参数
  int OUT_CHANNEL = 5;
  int IN_CHANNEL = 58;
  int HEIGHT = 112;
  int WIDTH = 88;
  int RUN_COUNT = 10;

  // 2. 解析参数
  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);
    if (Shape.size() == 5) {
      OUT_CHANNEL = Shape[0]; IN_CHANNEL = Shape[1];
      HEIGHT = Shape[2]; WIDTH = Shape[3]; RUN_COUNT = Shape[4];
    } else {
        std::cerr << "Invalid shape: OUT_CxIN_CxHxWxRUN" << std::endl;
        return -1;
    }
  }

  printf("Correlation Shape: OutC:%d InC:%d H:%d W:%d\n", 
         OUT_CHANNEL, IN_CHANNEL, HEIGHT, WIDTH);

  // 3. 初始化
  size_t in_size = (size_t)HEIGHT * WIDTH * IN_CHANNEL;
  size_t out_size = (size_t)HEIGHT * WIDTH * OUT_CHANNEL;

  std::vector<int8_t> src0(in_size);
  std::vector<int8_t> src1(in_size);
  std::vector<int8_t> real_out(out_size, 0);

  random_init(src0, 0, 255);
  random_init(src1, 0, 255);

#ifdef TRITON_KERNEL_ENABLE
  // 4. Grid 计算
  // 假设 BLOCK_SIZE_H / W 定义在头文件中
  int gridZ = OUT_CHANNEL;
  int gridY = std::ceil((float)HEIGHT / correlation_kernel_BLOCK_SIZE_H);
  int gridX = std::ceil((float)WIDTH / correlation_kernel_BLOCK_SIZE_W);

  // 5. Benchmark
  benchmark_kernel(RUN_COUNT, [&]() {
    correlation_kernel_omp(
        gridX, gridY, gridZ, 
        &correlation_kernel,
        src0.data(), 
        src1.data(), 
        real_out.data(),
        OUT_CHANNEL, IN_CHANNEL, HEIGHT, WIDTH,
        HEIGHT * WIDTH, // stride or total_pixels? 原代码传的是 H*W
        OUT_SHIFT
    );
  });
#else
  std::cerr << "Triton Kernel Not Enabled!" << std::endl;
#endif

  return 0;
}
