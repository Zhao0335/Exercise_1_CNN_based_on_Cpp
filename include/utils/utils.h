#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <chrono>

// 简单的日志函数
inline void log(const std::string& message) {
    std::cout << message << std::endl;
}

// 随机数生成器
class Random {
public:
    Random() {
        // 使用当前时间作为种子
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        gen = std::mt19937(static_cast<unsigned int>(seed));
    }
    
    // 生成[min, max)范围内的浮点数
    float randFloat(float min = 0.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(gen);
    }
    
    // 生成[min, max]范围内的整数
    int randInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }
    
private:
    std::mt19937 gen;
    unsigned long long seed;
};

#endif // UTILS_H