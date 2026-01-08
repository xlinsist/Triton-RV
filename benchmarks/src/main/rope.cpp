#ifdef TRITON_KERNEL_ENABLE
#include "rope_kernel_fw_launcher.h"
#endif

#include "support/support.h"
#include <vector>
#include <iostream>
#include <cassert>

int main(int argc, char *argv[]) {
  // 1. 默认参数
  int SEQ_LEN = 128;
  int BATCH_NUM = 32;
  int HEAD_NUM = 12;
  int HEAD_DIM = 64;
  int RUN_COUNT = 10;

  // 2. 解析参数
  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);
    if (Shape.size() == 5) {
      SEQ_LEN = Shape[0]; BATCH_NUM = Shape[1]; HEAD_NUM = Shape[2];
      HEAD_DIM = Shape[3]; RUN_COUNT = Shape[4];
    } else {
        std::cerr << "Invalid shape format: SEQ_LENxBATCH_NUMxHEAD_NUMxHEAD_DIMxRUN_COUNT" << std::endl;
        return -1;
    }
  }

  printf("RoPE Data shape S:%d B:%d H:%d D:%d, Run count: %d\n", 
         SEQ_LEN, BATCH_NUM, HEAD_NUM, HEAD_DIM, RUN_COUNT);

  // 3. 资源分配与初始化
  size_t total_elements = (size_t)SEQ_LEN * BATCH_NUM * HEAD_NUM * HEAD_DIM;
  size_t freq_elements = (size_t)SEQ_LEN * HEAD_DIM;

  std::vector<float> t(total_elements);
  std::vector<float> freq_cos(freq_elements);
  std::vector<float> freq_sin(freq_elements);
  std::vector<float> real_out(total_elements, 0.0f);

  random_init(t, -1.0, 1.0);
  random_init(freq_cos, -1.0, 1.0);
  random_init(freq_sin, -1.0, 1.0);

#ifdef TRITON_KERNEL_ENABLE
  // 4. Benchmark
  benchmark_kernel(RUN_COUNT, [&]() {
    rope_kernel_fw_omp(
        HEAD_NUM, BATCH_NUM, SEQ_LEN,   // Grid Dimensions
        &rope_kernel_fw,                // Kernel Ptr
        t.data(),                       // Input tensor
        (int)(BATCH_NUM * HEAD_NUM * HEAD_DIM), // stride_t_seq (假设 Row-Major: S, B, H, D)
        (int)(HEAD_NUM * HEAD_DIM),             // stride_t_batch
        real_out.data(), 
        freq_cos.data(), 
        freq_sin.data(),
        (int)HEAD_DIM, // stride_cos_seq
        (int)HEAD_DIM, // stride_sin_seq
        SEQ_LEN, 
        HEAD_DIM
    );
  });
#else
  std::cerr << "Triton Kernel Not Enabled!" << std::endl;
#endif

  return 0;
}
