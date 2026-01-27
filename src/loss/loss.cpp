#include "../../include/loss/loss.h"
#include <cmath>

// 交叉熵损失函数的前向传播
float CrossEntropyLoss::forward(const Tensor<float>& predictions, const Tensor<float>& labels) {
    float loss = 0.0f;
    size_t batch_size = predictions.getShape()[0];
    size_t num_classes = predictions.getShape()[1];
    
    // 遍历每个样本
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < num_classes; j++) {
            // 计算单维索引
            size_t pred_idx = i * num_classes + j;
            size_t label_idx = i * num_classes + j;
            
            // 避免log(0)的情况，添加一个小的epsilon
            float pred = predictions[pred_idx];
            if (pred < 1e-8) {
                pred = 1e-8;
            }
            
            // 交叉熵损失公式：-sum(y * log(p))
            loss -= labels[label_idx] * std::log(pred);
        }
    }
    
    // 返回平均损失
    return loss / batch_size;
}

// 交叉熵损失函数的反向传播
Tensor<float> CrossEntropyLoss::backward(const Tensor<float>& predictions, const Tensor<float>& labels) {
    size_t batch_size = predictions.getShape()[0];
    size_t num_classes = predictions.getShape()[1];
    
    // 创建梯度张量
    Tensor<float> grad(predictions.getShape());
    
    // 遍历每个样本
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < num_classes; j++) {
            // 计算单维索引
            size_t idx = i * num_classes + j;
            
            // 避免除以0的情况，添加一个小的epsilon
            float pred = predictions[idx];
            if (pred < 1e-8) {
                pred = 1e-8;
            }
            
            // 交叉熵损失的梯度：(p - y) / batch_size
            grad[idx] = (pred - labels[idx]) / batch_size;
        }
    }
    
    return grad;
}