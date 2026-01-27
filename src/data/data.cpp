#include "../../include/data/data.h"
#include <fstream>
#include <iostream>
#include <cstdint>

// MNISTLoader构造函数
MNISTLoader::MNISTLoader(const std::string &data_dir)
{
    data_dir_ = data_dir;
    std::cout << "MNISTLoader initialized with data directory: " << data_dir_ << std::endl;
}

// 加载训练集
void MNISTLoader::loadTrainSet()
{
    std::cout << "Loading training set..." << std::endl;

    // 构建训练集文件路径
    std::string images_file = data_dir_ + "/train-images.idx3-ubyte";
    std::string labels_file = data_dir_ + "/train-labels.idx1-ubyte";

    // 读取图像和标签
    train_images_ = readImages(images_file);
    train_labels_ = readLabels(labels_file);

    std::cout << "Training set loaded successfully!" << std::endl;
}

// 加载测试集
void MNISTLoader::loadTestSet()
{
    std::cout << "Loading test set..." << std::endl;

    // 构建测试集文件路径
    std::string images_file = data_dir_ + "/t10k-images.idx3-ubyte";
    std::string labels_file = data_dir_ + "/t10k-labels.idx1-ubyte";

    // 读取图像和标签
    test_images_ = readImages(images_file);
    test_labels_ = readLabels(labels_file);

    std::cout << "Test set loaded successfully!" << std::endl;
}

// 获取训练集图像
const Tensor<float> &MNISTLoader::getTrainImages() const
{
    return train_images_;
}

// 获取训练集标签
const Tensor<float> &MNISTLoader::getTrainLabels() const
{
    return train_labels_;
}

// 获取测试集图像
const Tensor<float> &MNISTLoader::getTestImages() const
{
    return test_images_;
}

// 获取测试集标签
const Tensor<float> &MNISTLoader::getTestLabels() const
{
    return test_labels_;
}

// 静态字节交换函数
uint32_t MNISTLoader::swapBytes(uint32_t value)
{
    uint32_t result = 0;
    result |= (value & 0x000000FF) << 24;
    result |= (value & 0x0000FF00) << 8;
    result |= (value & 0x00FF0000) >> 8;
    result |= (value & 0xFF000000) >> 24;
    return result;
}

// 读取MNIST图像文件
Tensor<float> MNISTLoader::readImages(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        // 返回一个空张量
        std::vector<size_t> shape = {1, 1, 28, 28};
        return Tensor<float>(shape);
    }

    // 读取文件头
    uint32_t magic, num_images, height, width;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char *>(&height), sizeof(height));
    file.read(reinterpret_cast<char *>(&width), sizeof(width));

    // 转换字节序（MNIST文件使用大端字节序）
    magic = swapBytes(magic);
    num_images = swapBytes(num_images);
    height = swapBytes(height);
    width = swapBytes(width);

    std::cout << "Reading images from " << filename << ": " << num_images << " images, "
              << height << "x" << width << " pixels each" << std::endl;

    // 创建输出张量 [num_images, 1, height, width]
    std::vector<size_t> shape = {
        static_cast<size_t>(num_images),
        1,
        static_cast<size_t>(height),
        static_cast<size_t>(width)};
    Tensor<float> images(shape);

    // 读取图像数据
    for (uint32_t i = 0; i < num_images; ++i)
    {
        for (uint32_t h = 0; h < height; ++h)
        {
            for (uint32_t w = 0; w < width; ++w)
            {
                uint8_t pixel;
                file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));

                // 计算单维索引
                size_t idx = i * height * width + h * width + w;

                // 将像素值归一化到 [0, 1]
                images[idx] = static_cast<float>(pixel) / 255.0f;
            }
        }
    }

    file.close();
    return images;
}

// 读取MNIST标签文件
Tensor<float> MNISTLoader::readLabels(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        // 返回一个空张量
        std::vector<size_t> shape = {1, 10};
        return Tensor<float>(shape);
    }

    // 读取文件头
    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

    // 转换字节序
    magic = swapBytes(magic);
    num_labels = swapBytes(num_labels);

    std::cout << "Reading labels from " << filename << ": " << num_labels << " labels" << std::endl;

    // 读取标签数据
    std::vector<uint8_t> raw_labels(num_labels);
    file.read(reinterpret_cast<char *>(raw_labels.data()), num_labels);

    file.close();

    // 创建独热编码张量 [num_labels, 10]
    std::vector<size_t> shape = {
        static_cast<size_t>(num_labels),
        10};
    Tensor<float> one_hot(shape);

    // 填充独热编码
    for (uint32_t i = 0; i < num_labels; ++i)
    {
        uint8_t label = raw_labels[i];
        if (label < 10)
        {
            // 计算单维索引
            size_t idx = i * 10 + label;
            one_hot[idx] = 1.0f;
        }
    }

    return one_hot;
}
