#ifdef TRITON_KERNEL_ENABLE
#include "warp_kernel_launcher.h"
#endif

#include "support/support.h"
#include <vector>
#include <iostream>
#include <cstdint>

int main(int argc, char *argv[]) {
    // 1. 默认参数
    int H = 128, W = 128, C = 3;
    int RUN_COUNT = 10;

    // 2. 解析参数 HxWxCxRUN_COUNT
    if (argc >= 2) {
        std::vector<int> Shape = splitStringToInts(argv[1]);
        if (Shape.size() == 4) {
            H = Shape[0]; W = Shape[1]; C = Shape[2]; RUN_COUNT = Shape[3];
        } else {
            std::cerr << "Invalid shape format: HxWxCxRUN_COUNT" << std::endl;
            return -1;
        }
    }

    printf("Warp Data shape %dx%dx%d, Run count: %d\n", H, W, C, RUN_COUNT);

    // 3. 初始化数据 (使用 vector)
    std::vector<int8_t> input(H * W * C);
    std::vector<int16_t> offset(H * W);
    std::vector<int8_t> out(H * W * C, 0);

    // 利用模板化的 random_init
    random_init(input, -128, 127);
    random_init(offset, -32768, 32767);

#ifdef TRITON_KERNEL_ENABLE
    // 4. 定义 Grid
    int gridX = H;
    int gridY = C;
    int gridZ = 1;

    // 5. Benchmark
    benchmark_kernel(RUN_COUNT, [&]() {
        warp_kernel_omp(
            gridX, gridY, gridZ, 
            &warp_kernel, 
            input.data(), 
            offset.data(), 
            out.data(), 
            C, H, W
        );
    });
#else
    std::cerr << "Triton Kernel Not Enabled!" << std::endl;
#endif

    return 0;
}
