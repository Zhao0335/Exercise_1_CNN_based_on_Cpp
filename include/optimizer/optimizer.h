#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../tensor/tensor.h"

// SGD优化器
class SGD {
public:
    // 构造函数
    SGD(float lr = 0.01f);
    
    // 更新参数
    void step(Tensor<float>& params, const Tensor<float>& grads);
    
    // 批量更新多个参数
    void step(std::vector<Tensor<float>*>& params, std::vector<Tensor<float>*>& grads);
    
private:
    float lr_; // 学习率
};

#endif // OPTIMIZER_H