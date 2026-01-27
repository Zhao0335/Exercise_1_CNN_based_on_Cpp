#include "../../include/layers/layers.h"
#include "../../include/utils/utils.h"
#include <iostream>
#include <cmath>

// 全连接层构造函数
Linear::Linear(int in_features, int out_features) {
    in_features_ = in_features;
    out_features_ = out_features;
    
    // 初始化权重 [out_features, in_features]
    std::vector<size_t> weight_shape = {static_cast<size_t>(out_features), static_cast<size_t>(in_features)};
    weights_ = Tensor<float>(weight_shape);
    
    // 初始化偏置 [out_features]
    std::vector<size_t> bias_shape = {static_cast<size_t>(out_features)};
    bias_ = Tensor<float>(bias_shape);
    
    // 使用随机数初始化权重和偏置
    Random rand;
    for (size_t i = 0; i < weights_.size(); i++) {
        // 使用Xavier初始化
        float limit = std::sqrt(6.0f / (in_features + out_features));
        weights_[i] = rand.randFloat(-limit, limit);
    }
    
    for (size_t i = 0; i < bias_.size(); i++) {
        bias_[i] = 0.0f;
    }
}

// 全连接层前向传播
Tensor<float> Linear::forward(const Tensor<float>& input) {
    // 保存输入，用于反向传播
    input_ = input;
    
    // 输入形状：[batch_size, in_features]
    size_t batch_size = input.getShape()[0];
    
    // 输出形状：[batch_size, out_features]
    std::vector<size_t> output_shape = {batch_size, static_cast<size_t>(out_features_)};
    Tensor<float> output(output_shape, 0.0f);
    
    // 矩阵乘法：output = input * weights^T + bias
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < static_cast<size_t>(out_features_); o++) {
            for (size_t i = 0; i < static_cast<size_t>(in_features_); i++) {
                size_t input_idx = b * in_features_ + i;
                size_t weight_idx = o * in_features_ + i;
                output[b * out_features_ + o] += input[input_idx] * weights_[weight_idx];
            }
            output[b * out_features_ + o] += bias_[o];
        }
    }
    
    return output;
}

// 全连接层反向传播
Tensor<float> Linear::backward(const Tensor<float>& grad_output) {
    // 输入形状：[batch_size, in_features]
    size_t batch_size = input_.getShape()[0];
    
    // 计算权重梯度：grad_weights = grad_output^T * input
    std::vector<size_t> grad_weights_shape = {static_cast<size_t>(out_features_), static_cast<size_t>(in_features_)};
    Tensor<float> grad_weights(grad_weights_shape, 0.0f);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < static_cast<size_t>(out_features_); o++) {
            for (size_t i = 0; i < static_cast<size_t>(in_features_); i++) {
                size_t grad_output_idx = b * out_features_ + o;
                size_t input_idx = b * in_features_ + i;
                size_t grad_weights_idx = o * in_features_ + i;
                grad_weights[grad_weights_idx] += grad_output[grad_output_idx] * input_[input_idx];
            }
        }
    }
    
    // 计算偏置梯度：grad_bias = sum(grad_output, dim=0)
    std::vector<size_t> grad_bias_shape = {static_cast<size_t>(out_features_)};
    Tensor<float> grad_bias(grad_bias_shape, 0.0f);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < static_cast<size_t>(out_features_); o++) {
            size_t grad_output_idx = b * out_features_ + o;
            grad_bias[o] += grad_output[grad_output_idx];
        }
    }
    
    // 计算输入梯度：grad_input = grad_output * weights
    std::vector<size_t> grad_input_shape = {batch_size, static_cast<size_t>(in_features_)};
    Tensor<float> grad_input(grad_input_shape, 0.0f);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < static_cast<size_t>(in_features_); i++) {
            for (size_t o = 0; o < static_cast<size_t>(out_features_); o++) {
                size_t grad_output_idx = b * out_features_ + o;
                size_t weight_idx = o * in_features_ + i;
                size_t grad_input_idx = b * in_features_ + i;
                grad_input[grad_input_idx] += grad_output[grad_output_idx] * weights_[weight_idx];
            }
        }
    }
    
    // 更新权重和偏置（简化实现，实际应通过优化器）
    float learning_rate = 0.01f;
    for (size_t i = 0; i < weights_.size(); i++) {
        weights_[i] -= learning_rate * grad_weights[i];
    }
    for (size_t i = 0; i < bias_.size(); i++) {
        bias_[i] -= learning_rate * grad_bias[i];
    }
    
    return grad_input;
}

// ReLU激活函数前向传播
Tensor<float> ReLU::forward(const Tensor<float>& input) {
    // 保存输入，用于反向传播
    input_ = input;
    
    // 复制输入形状
    Tensor<float> output(input.getShape());
    
    // 逐元素应用ReLU：max(0, x)
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::max(0.0f, input[i]);
    }
    
    return output;
}

// ReLU激活函数反向传播
Tensor<float> ReLU::backward(const Tensor<float>& grad_output) {
    // 复制输入形状
    Tensor<float> grad_input(input_.getShape());
    
    // 逐元素应用ReLU梯度：1 if x > 0 else 0
    for (size_t i = 0; i < input_.size(); i++) {
        grad_input[i] = grad_output[i] * (input_[i] > 0.0f ? 1.0f : 0.0f);
    }
    
    return grad_input;
}

// 卷积层构造函数
Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    kernel_size_ = kernel_size;
    stride_ = stride;
    padding_ = padding;
    
    // 初始化权重 [in_channels, out_channels, kernel_size, kernel_size]
    std::vector<size_t> weight_shape = {
        static_cast<size_t>(in_channels),
        static_cast<size_t>(out_channels),
        static_cast<size_t>(kernel_size),
        static_cast<size_t>(kernel_size)
    };
    weights_ = Tensor<float>(weight_shape);
    
    // 初始化偏置 [out_channels]
    std::vector<size_t> bias_shape = {static_cast<size_t>(out_channels)};
    bias_ = Tensor<float>(bias_shape);
    
    // 使用随机数初始化权重和偏置
    Random rand;
    for (size_t i = 0; i < weights_.size(); i++) {
        // 使用Xavier初始化
        float limit = std::sqrt(6.0f / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size));
        weights_[i] = rand.randFloat(-limit, limit);
    }
    
    for (size_t i = 0; i < bias_.size(); i++) {
        bias_[i] = 0.0f;
    }
}

// 卷积层前向传播
Tensor<float> Conv2d::forward(const Tensor<float>& input) {
    // 保存输入，用于反向传播
    input_ = input;
    
    // 使用Tensor类的conv2d方法
    std::vector<size_t> stride = {static_cast<size_t>(stride_), static_cast<size_t>(stride_)};
    std::vector<size_t> padding = {static_cast<size_t>(padding_), static_cast<size_t>(padding_)};
    
    Tensor<float> output = input.conv2d(weights_, stride, padding);
    
    // 添加偏置
    size_t batch_size = output.getShape()[0];
    size_t channels = output.getShape()[1];
    size_t height = output.getShape()[2];
    size_t width = output.getShape()[3];
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t idx = b * channels * height * width + c * height * width + h * width + w;
                    output[idx] += bias_[c];
                }
            }
        }
    }
    
    return output;
}

// 卷积层反向传播
Tensor<float> Conv2d::backward(const Tensor<float>& grad_output) {
    // 输入形状：[batch_size, in_channels, in_height, in_width]
    size_t batch_size = input_.getShape()[0];
    size_t in_channels = input_.getShape()[1];
    size_t in_height = input_.getShape()[2];
    size_t in_width = input_.getShape()[3];
    
    // 输出梯度形状：[batch_size, out_channels, out_height, out_width]
    size_t out_channels = grad_output.getShape()[1];
    size_t out_height = grad_output.getShape()[2];
    size_t out_width = grad_output.getShape()[3];
    
    // 卷积核形状：[in_channels, out_channels, kernel_height, kernel_width]
    size_t kernel_height = weights_.getShape()[2];
    size_t kernel_width = weights_.getShape()[3];
    
    // 计算偏置梯度：grad_bias = sum(grad_output, dim=(0,2,3))
    std::vector<size_t> grad_bias_shape = {static_cast<size_t>(out_channels)};
    Tensor<float> grad_bias(grad_bias_shape, 0.0f);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < out_channels; c++) {
            for (size_t h = 0; h < out_height; h++) {
                for (size_t w = 0; w < out_width; w++) {
                    size_t idx = b * out_channels * out_height * out_width + c * out_height * out_width + h * out_width + w;
                    grad_bias[c] += grad_output[idx];
                }
            }
        }
    }
    
    // 计算权重梯度：grad_weights = conv2d(input, grad_output, stride=1, padding=kernel_size-1)
    std::vector<size_t> grad_weights_shape = {static_cast<size_t>(in_channels), static_cast<size_t>(out_channels), static_cast<size_t>(kernel_height), static_cast<size_t>(kernel_width)};
    Tensor<float> grad_weights(grad_weights_shape, 0.0f);
    
    // 简化实现：逐元素计算权重梯度
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t kh = 0; kh < kernel_height; kh++) {
                    for (size_t kw = 0; kw < kernel_width; kw++) {
                        for (size_t oh = 0; oh < out_height; oh++) {
                            for (size_t ow = 0; ow < out_width; ow++) {
                                size_t ih = oh * stride_ + kh;
                                size_t iw = ow * stride_ + kw;
                                
                                if (ih < in_height && iw < in_width) {
                                    size_t input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + ih * in_width + iw;
                                    size_t grad_output_idx = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow;
                                    size_t grad_weights_idx = ic * out_channels * kernel_height * kernel_width + oc * kernel_height * kernel_width + kh * kernel_width + kw;
                                    grad_weights[grad_weights_idx] += input_[input_idx] * grad_output[grad_output_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 计算输入梯度：grad_input = conv_transpose2d(grad_output, weights, stride, padding)
    std::vector<size_t> grad_input_shape = {batch_size, static_cast<size_t>(in_channels), static_cast<size_t>(in_height), static_cast<size_t>(in_width)};
    Tensor<float> grad_input(grad_input_shape, 0.0f);
    
    // 简化实现：逐元素计算输入梯度
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t ic = 0; ic < in_channels; ic++) {
            for (size_t ih = 0; ih < in_height; ih++) {
                for (size_t iw = 0; iw < in_width; iw++) {
                    for (size_t oc = 0; oc < out_channels; oc++) {
                        for (size_t kh = 0; kh < kernel_height; kh++) {
                            for (size_t kw = 0; kw < kernel_width; kw++) {
                                size_t oh = (ih - kh + padding_) / stride_;
                                size_t ow = (iw - kw + padding_) / stride_;
                                
                                if (oh < out_height && ow < out_width && (ih - kh + padding_) % stride_ == 0 && (iw - kw + padding_) % stride_ == 0) {
                                    size_t grad_output_idx = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow;
                                    size_t weight_idx = ic * out_channels * kernel_height * kernel_width + oc * kernel_height * kernel_width + kh * kernel_width + kw;
                                    size_t grad_input_idx = b * in_channels * in_height * in_width + ic * in_height * in_width + ih * in_width + iw;
                                    grad_input[grad_input_idx] += grad_output[grad_output_idx] * weights_[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 更新权重和偏置
    float learning_rate = 0.01f;
    for (size_t i = 0; i < weights_.size(); i++) {
        weights_[i] -= learning_rate * grad_weights[i];
    }
    for (size_t i = 0; i < bias_.size(); i++) {
        bias_[i] -= learning_rate * grad_bias[i];
    }
    
    return grad_input;
}

// 最大池化层构造函数
MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding) {
    kernel_size_ = kernel_size;
    stride_ = stride;
    padding_ = padding;
}

// 最大池化层前向传播
Tensor<float> MaxPool2d::forward(const Tensor<float>& input) {
    // 保存输入，用于反向传播
    input_ = input;
    
    // 输入形状：[batch_size, channels, in_height, in_width]
    size_t batch_size = input.getShape()[0];
    size_t channels = input.getShape()[1];
    size_t in_height = input.getShape()[2];
    size_t in_width = input.getShape()[3];
    
    // 计算输出形状
    size_t out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    size_t out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    // 输出形状：[batch_size, channels, out_height, out_width]
    std::vector<size_t> output_shape = {batch_size, channels, out_height, out_width};
    Tensor<float> output(output_shape, 0.0f);
    
    // 初始化最大值索引
    max_indices_.resize(batch_size);
    for (size_t b = 0; b < batch_size; b++) {
        max_indices_[b].resize(channels);
        for (size_t c = 0; c < channels; c++) {
            max_indices_[b][c].resize(out_height);
            for (size_t h = 0; h < out_height; h++) {
                max_indices_[b][c][h].resize(out_width, 0);
            }
        }
    }
    
    // 输入填充
    std::vector<size_t> padding_vec = {static_cast<size_t>(padding_), static_cast<size_t>(padding_)};
    Tensor<float> padded_input = input.pad(padding_vec, 0.0f);
    
    // 核心池化操作
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_height; oh++) {
                for (size_t ow = 0; ow < out_width; ow++) {
                    float max_val = -std::numeric_limits<float>::max();
                    size_t max_idx = 0;
                    
                    // 遍历池化窗口
                    for (size_t kh = 0; kh < static_cast<size_t>(kernel_size_); kh++) {
                        for (size_t kw = 0; kw < static_cast<size_t>(kernel_size_); kw++) {
                            size_t ih = oh * stride_ + kh;
                            size_t iw = ow * stride_ + kw;
                            
                            size_t padded_idx = b * channels * (in_height + 2 * padding_) * (in_width + 2 * padding_) + c * (in_height + 2 * padding_) * (in_width + 2 * padding_) + ih * (in_width + 2 * padding_) + iw;
                            if (padded_input[padded_idx] > max_val) {
                                max_val = padded_input[padded_idx];
                                max_idx = ih * (in_width + 2 * padding_) + iw;
                            }
                        }
                    }
                    
                    size_t output_idx = b * channels * out_height * out_width + c * out_height * out_width + oh * out_width + ow;
                    output[output_idx] = max_val;
                    max_indices_[b][c][oh][ow] = max_idx;
                }
            }
        }
    }
    
    return output;
}

// 最大池化层反向传播
Tensor<float> MaxPool2d::backward(const Tensor<float>& grad_output) {
    // 输入形状：[batch_size, channels, in_height, in_width]
    size_t batch_size = input_.getShape()[0];
    size_t channels = input_.getShape()[1];
    size_t in_height = input_.getShape()[2];
    size_t in_width = input_.getShape()[3];
    
    // 输出梯度形状：[batch_size, channels, out_height, out_width]
    size_t out_height = grad_output.getShape()[2];
    size_t out_width = grad_output.getShape()[3];
    
    // 计算输入梯度
    std::vector<size_t> grad_input_shape = {batch_size, channels, in_height, in_width};
    Tensor<float> grad_input(grad_input_shape, 0.0f);
    
    // 输入填充
    std::vector<size_t> padding_vec = {static_cast<size_t>(padding_), static_cast<size_t>(padding_)};
    Tensor<float> padded_grad_input = grad_input.pad(padding_vec, 0.0f);
    
    // 核心反向传播操作
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_height; oh++) {
                for (size_t ow = 0; ow < out_width; ow++) {
                    size_t grad_output_idx = b * channels * out_height * out_width + c * out_height * out_width + oh * out_width + ow;
                    float grad = grad_output[grad_output_idx];
                    
                    // 获取最大值索引
                    size_t max_idx = max_indices_[b][c][oh][ow];
                    size_t ih = max_idx / (in_width + 2 * padding_);
                    size_t iw = max_idx % (in_width + 2 * padding_);
                    
                    // 将梯度传递到最大值位置
                    size_t padded_idx = b * channels * (in_height + 2 * padding_) * (in_width + 2 * padding_) + c * (in_height + 2 * padding_) * (in_width + 2 * padding_) + ih * (in_width + 2 * padding_) + iw;
                    padded_grad_input[padded_idx] += grad;
                }
            }
        }
    }
    
    // 去除填充
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < in_height; h++) {
                for (size_t w = 0; w < in_width; w++) {
                    size_t padded_idx = b * channels * (in_height + 2 * padding_) * (in_width + 2 * padding_) + c * (in_height + 2 * padding_) * (in_width + 2 * padding_) + (h + padding_) * (in_width + 2 * padding_) + (w + padding_);
                    size_t grad_input_idx = b * channels * in_height * in_width + c * in_height * in_width + h * in_width + w;
                    grad_input[grad_input_idx] = padded_grad_input[padded_idx];
                }
            }
        }
    }
    
    return grad_input;
}

// Softmax激活函数前向传播
Tensor<float> Softmax::forward(const Tensor<float>& input) {
    // 复制输入形状
    Tensor<float> output(input.getShape());
    
    size_t batch_size = input.getShape()[0];
    size_t features = input.getShape()[1];
    
    // 逐样本应用Softmax
    for (size_t b = 0; b < batch_size; b++) {
        // 计算指数和
        float sum_exp = 0.0f;
        for (size_t f = 0; f < features; f++) {
            size_t idx = b * features + f;
            sum_exp += std::exp(input[idx]);
        }
        
        // 计算Softmax值
        for (size_t f = 0; f < features; f++) {
            size_t idx = b * features + f;
            output[idx] = std::exp(input[idx]) / sum_exp;
        }
    }
    
    // 保存输出，用于反向传播
    output_ = output;
    
    return output;
}

// Softmax激活函数反向传播
Tensor<float> Softmax::backward(const Tensor<float>& grad_output) {
    // 输入形状：[batch_size, features]
    size_t batch_size = output_.getShape()[0];
    size_t features = output_.getShape()[1];
    
    // 计算输入梯度
    std::vector<size_t> grad_input_shape = {batch_size, features};
    Tensor<float> grad_input(grad_input_shape, 0.0f);
    
    // 逐样本计算梯度
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < features; i++) {
            for (size_t j = 0; j < features; j++) {
                size_t grad_output_idx = b * features + j;
                size_t output_i_idx = b * features + i;
                size_t output_j_idx = b * features + j;
                size_t grad_input_idx = b * features + i;
                
                if (i == j) {
                    grad_input[grad_input_idx] += grad_output[grad_output_idx] * output_[output_i_idx] * (1.0f - output_[output_j_idx]);
                } else {
                    grad_input[grad_input_idx] -= grad_output[grad_output_idx] * output_[output_i_idx] * output_[output_j_idx];
                }
            }
        }
    }
    
    return grad_input;
}
