#ifndef DATA_H
#define DATA_H

#include "../tensor/tensor.h"
#include <string>

// MNIST数据集加载器
class MNISTLoader
{
public:
    // 构造函数
    explicit MNISTLoader(const std::string &data_dir);

    // 加载训练集
    void loadTrainSet();

    // 加载测试集
    void loadTestSet();

    // 获取训练集图像
    const Tensor<float>& getTrainImages() const;

    // 获取训练集标签
    const Tensor<float>& getTrainLabels() const;

    // 获取测试集图像
    const Tensor<float>& getTestImages() const;

    // 获取测试集标签
    const Tensor<float>& getTestLabels() const;

private:
    // 读取MNIST图像文件
    Tensor<float> readImages(const std::string &filename);

    // 读取MNIST标签文件
    Tensor<float> readLabels(const std::string &filename);

    // 字节交换函数
    static uint32_t swapBytes(uint32_t value);

    std::string data_dir_; // 数据目录

    // 训练集数据
    Tensor<float> train_images_;
    Tensor<float> train_labels_;

    // 测试集数据
    Tensor<float> test_images_;
    Tensor<float> test_labels_;
};


#endif // DATA_H