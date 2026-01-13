#ifndef SUPPORT_H
#define SUPPORT_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <functional>
#include <cassert>

const std::string TRITON_KERNEL = "Triton Kernel";

std::vector<int> splitStringToInts(const std::string &str, char delimiter = 'x');

// 1. 通用随机初始化函数
// 使用 std::vector 自动管理内存，避免手动 malloc/free
template <typename T>
void random_init(std::vector<T>& data, double min_val = 0, double max_val = 1) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dis((T)min_val, (T)max_val);
        for (auto &v : data) v = dis(gen);
    } else {
        std::uniform_int_distribution<int> dis((int)min_val, (int)max_val);
        for (auto &v : data) v = static_cast<T>(dis(gen));
    }
}

// 2. 通用 Benchmark 函数
// 接收一个 lambda 表达式作为 kernel_launcher，负责处理计时和打印
inline void benchmark_kernel(int run_count, std::function<void()> kernel_launcher) {
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    // 预热 (可选，防止首次运行的开销影响计时)
    // kernel_launcher(); 

    auto beginTime = high_resolution_clock::now();
    
    for (int i = 0; i < run_count; i++) {
        kernel_launcher();
    }

    auto endTime = high_resolution_clock::now();
    
    std::chrono::duration<double> time_interval = endTime - beginTime;
    
    std::cerr << "Running " << TRITON_KERNEL 
              << " Time: " << time_interval.count() << " s" << std::endl;
}

// 3. 其它辅助函数声明
bool getBoolEnv(const std::string &env);

std::optional<int64_t> getIntEnv(const std::string &env);

std::optional<std::string> getStringEnv(const std::string &env);

// Data base
std::string getDB(const std::string &Shape);

std::unique_ptr<uint32_t[][3]> get_all_grids(uint32_t gridX, uint32_t gridY,
                                             uint32_t gridZ);

#endif // SUPPORT_H


