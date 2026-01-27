#ifndef LOSS_H
#define LOSS_H

#include "../tensor/tensor.h"

// 交叉熵损失函数
class CrossEntropyLoss {
public:
    // 计算损失值
    float forward(const Tensor<float>& predictions, const Tensor<float>& labels);
    
    // 计算梯度
    Tensor<float> backward(const Tensor<float>& predictions, const Tensor<float>& labels);
};

#endif // LOSS_H