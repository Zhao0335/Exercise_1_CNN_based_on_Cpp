#include "../../include/optimizer/optimizer.h"

// SGD优化器构造函数
SGD::SGD(float lr) {
    lr_ = lr;
}

// 更新单个参数
void SGD::step(Tensor<float>& params, const Tensor<float>& grads) {
    // 参数更新：params = params - lr * grads
    for (size_t i = 0; i < params.size(); i++) {
        params[i] -= lr_ * grads[i];
    }
}

// 批量更新多个参数
void SGD::step(std::vector<Tensor<float>*>& params, std::vector<Tensor<float>*>& grads) {
    // 遍历所有参数-梯度对
    for (size_t i = 0; i < params.size(); i++) {
        Tensor<float>* param = params[i];
        Tensor<float>* grad = grads[i];
        
        // 参数更新：params = params - lr * grads
        for (size_t j = 0; j < param->size(); j++) {
            (*param)[j] -= lr_ * (*grad)[j];
        }
    }
}
