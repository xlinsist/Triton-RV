#ifdef TRITON_KERNEL_ENABLE
#include "_layer_norm_fwd_fused_launcher.h"
#endif

#include "support/support.h"
#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
  // 1. 默认参数
  int N = 4096; // Token length
  int D = 1024; // Embedding dim
  int RUN_COUNT = 10;

  // 2. 解析参数 NxDxRUN_COUNT
  if (argc >= 2) {
    std::vector<int> Shape = splitStringToInts(argv[1]);
    if (Shape.size() == 3) {
      N = Shape[0]; D = Shape[1]; RUN_COUNT = Shape[2];
    } else {
        std::cerr << "Invalid shape format: NxDxRUN_COUNT" << std::endl;
        return -1;
    }
  }

  printf("LayerNorm Data shape %dx%d, Run count: %d\n", N, D, RUN_COUNT);

  // 3. 初始化数据
  std::vector<float> x(N * D);
  std::vector<float> w(D);
  std::vector<float> b(D);
  
  // 输出 Buffer
  std::vector<float> out(N * D);
  std::vector<float> mean(N);
  std::vector<float> rstd(N);

  random_init(x, -2.0, 2.0);
  random_init(w, 0.0, 1.0);
  random_init(b, 0.0, 1.0);

#ifdef TRITON_KERNEL_ENABLE
  // 4. Benchmark
  benchmark_kernel(RUN_COUNT, [&]() {
    _layer_norm_fwd_fused_omp(
        N, 1, 1,                 // GridX, Y, Z (这里通常 LayerNorm 是按 Row 并行，GridX=N)
        &_layer_norm_fwd_fused,  // Kernel Ptr
        x.data(), 
        out.data(), 
        w.data(), 
        b.data(), 
        mean.data(), 
        rstd.data(), 
        D,                       // stride_x_row (假设连续，则为 D)
        D,                       // stride_out_row
        1e-5                     // eps
    );
  });
#else
  std::cerr << "Triton Kernel Not Enabled!" << std::endl;
#endif

  return 0;
}
