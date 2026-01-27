#ifndef LAYERS_H
#define LAYERS_H

#include "../tensor/tensor.h"

// 基础层类
class Layer
{
public:
    virtual ~Layer() {}

    // 前向传播
    virtual Tensor<float> forward(const Tensor<float> &input) = 0;

    // 反向传播（简化实现）
    virtual Tensor<float> backward(const Tensor<float> &grad_output)
    {
        return Tensor<float>();
    }
};

// 全连接层
class Linear : public Layer
{
public:
    // 构造函数
    Linear(int in_features, int out_features);

    // 前向传播
    Tensor<float> forward(const Tensor<float> &input) override;

    // 反向传播
    Tensor<float> backward(const Tensor<float> &grad_output) override;

    // 获取权重和偏置
    Tensor<float> &getWeights() { return weights_; }
    Tensor<float> &getBias() { return bias_; }

    // 获取前向传播输入
    Tensor<float> &getInput() { return input_; }

private:
    int in_features_;
    int out_features_;
    Tensor<float> weights_;
    Tensor<float> bias_;
    Tensor<float> input_; // 存储前向传播的输入
};

// ReLU激活函数
class ReLU : public Layer
{
public:
    // 前向传播
    Tensor<float> forward(const Tensor<float> &input) override;

    // 反向传播
    Tensor<float> backward(const Tensor<float> &grad_output) override;

private:
    Tensor<float> input_; // 存储前向传播的输入
};

// 卷积层
class Conv2d : public Layer
{
public:
    // 构造函数
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);

    // 前向传播
    Tensor<float> forward(const Tensor<float> &input) override;

    // 反向传播
    Tensor<float> backward(const Tensor<float> &grad_output) override;

    // 获取权重和偏置
    Tensor<float> &getWeights() { return weights_; }
    Tensor<float> &getBias() { return bias_; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    Tensor<float> weights_;
    Tensor<float> bias_;
    Tensor<float> input_; // 存储前向传播的输入
};

// 最大池化层
class MaxPool2d : public Layer
{
public:
    // 构造函数
    MaxPool2d(int kernel_size, int stride = 1, int padding = 0);

    // 前向传播
    Tensor<float> forward(const Tensor<float> &input) override;

    // 反向传播
    Tensor<float> backward(const Tensor<float> &grad_output) override;

private:
    int kernel_size_;
    int stride_;
    int padding_;
    Tensor<float> input_;                                                    // 存储前向传播的输入
    std::vector<std::vector<std::vector<std::vector<size_t>>>> max_indices_; // 存储最大值索引
};

// Softmax激活函数
class Softmax : public Layer
{
public:
    // 前向传播
    Tensor<float> forward(const Tensor<float> &input) override;

    // 反向传播
    Tensor<float> backward(const Tensor<float> &grad_output) override;

private:
    Tensor<float> output_; // 存储前向传播的输出
};

#endif // LAYERS_H