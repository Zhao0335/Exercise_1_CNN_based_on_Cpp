#include "include/tensor/tensor.h"
#include "include/data/data.h"
#include "include/loss/loss.h"
#include "include/layers/layers.h"
#include "include/optimizer/optimizer.h"
#include "include/utils/utils.h"
#include <iostream>
#include <vector>

// 计算模型准确率
float calculateAccuracy(const Tensor<float> &predictions, const Tensor<float> &labels)
{
    size_t batch_size = predictions.getShape()[0];
    size_t num_classes = predictions.getShape()[1];
    size_t correct = 0;

    for (size_t i = 0; i < batch_size; i++)
    {
        // 找到预测值最大的索引
        size_t pred_idx = 0;
        float max_prob = 0.0f;
        for (size_t j = 0; j < num_classes; j++)
        {
            float prob = predictions[i * num_classes + j];
            if (prob > max_prob)
            {
                max_prob = prob;
                pred_idx = j;
            }
        }

        // 找到标签中为1的索引（真实类别）
        size_t true_idx = 0;
        for (size_t j = 0; j < num_classes; j++)
        {
            if (labels[i * num_classes + j] == 1.0f)
            {
                true_idx = j;
                break;
            }
        }

        // 比较预测类别和真实类别
        if (pred_idx == true_idx)
        {
            correct++;
        }
    }

    return static_cast<float>(correct) / batch_size;
}

int main()
{
    std::cout << "CNN for MNIST Classification" << std::endl;
    std::cout << "================================" << std::endl;

    // 1. 加载MNIST数据集
    std::cout << "Loading MNIST dataset..." << std::endl;
    MNISTLoader mnist_loader("data");
    mnist_loader.loadTrainSet();
    mnist_loader.loadTestSet();

    // 获取数据
    const Tensor<float> &train_images = mnist_loader.getTrainImages();
    const Tensor<float> &train_labels = mnist_loader.getTrainLabels();
    const Tensor<float> &test_images = mnist_loader.getTestImages();
    const Tensor<float> &test_labels = mnist_loader.getTestLabels();

    std::cout << "Dataset loaded successfully!" << std::endl;
    std::cout << "Training set size: " << train_images.getShape()[0] << std::endl;
    std::cout << "Test set size: " << test_images.getShape()[0] << std::endl;

    // 2. 构建CNN模型
    std::cout << "Building CNN model..." << std::endl;

    // 模型架构：Conv2d -> ReLU -> MaxPool2d -> Linear -> Softmax
    Conv2d conv1(1, 32, 3, 1, 1); // 输入通道1，输出通道32，卷积核3x3，步长1，填充1
    ReLU relu1;
    MaxPool2d pool1(2, 2);        // 池化核2x2，步长2
    Linear fc1(32 * 14 * 14, 10); // 输入特征：32通道 * 14x14特征图，输出特征：10个类别
    Softmax softmax;

    // 3. 定义损失函数和优化器
    CrossEntropyLoss criterion;
    SGD optimizer(0.01f); // 学习率0.01

    // 4. 训练参数
    int epochs = 5;
    int batch_size = 64;
    int train_size = train_images.getShape()[0];
    int test_size = test_images.getShape()[0];

    // 5. 训练模型
    std::cout << "Starting training..." << std::endl;
    std::cout << "================================" << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        int batch_count = 0;

        // 批次训练
        for (int b = 0; b < train_size; b += batch_size)
        {
            // 计算当前批次大小
            int current_batch_size = std::min(batch_size, train_size - b);

            std::cout << "Processing batch " << (b / batch_size + 1) << "/" << (train_size / batch_size + 1) << "..." << std::endl;

            // 提取当前批次
            std::vector<size_t> img_batch_shape = {static_cast<size_t>(current_batch_size), 1, 28, 28};
            Tensor<float> img_batch(img_batch_shape);

            std::vector<size_t> label_batch_shape = {static_cast<size_t>(current_batch_size), 10};
            Tensor<float> label_batch(label_batch_shape);

            // 复制数据到批次张量
            for (int i = 0; i < current_batch_size; i++)
            {
                // 复制图像数据
                for (int c = 0; c < 1; c++)
                {
                    for (int h = 0; h < 28; h++)
                    {
                        for (int w = 0; w < 28; w++)
                        {
                            size_t src_idx = (b + i) * (1 * 28 * 28) + c * (28 * 28) + h * 28 + w;
                            size_t dst_idx = i * (1 * 28 * 28) + c * (28 * 28) + h * 28 + w;
                            img_batch[dst_idx] = train_images[src_idx];
                        }
                    }
                }

                // 复制标签数据
                for (int l = 0; l < 10; l++)
                {
                    size_t src_idx = (b + i) * 10 + l;
                    size_t dst_idx = i * 10 + l;
                    label_batch[dst_idx] = train_labels[src_idx];
                }
            }

            std::cout << "Starting forward propagation..." << std::endl;
            // 前向传播
            Tensor<float> conv_out = conv1.forward(img_batch);
            Tensor<float> relu_out = relu1.forward(conv_out);
            Tensor<float> pool_out = pool1.forward(relu_out);

            // 展平池化输出
            std::vector<size_t> flat_shape = {static_cast<size_t>(current_batch_size), 32 * 14 * 14};
            Tensor<float> flat_out(flat_shape);

            // 手动展平
            for (int i = 0; i < current_batch_size; i++)
            {
                for (int c = 0; c < 32; c++)
                {
                    for (int h = 0; h < 14; h++)
                    {
                        for (int w = 0; w < 14; w++)
                        {
                            size_t src_idx = i * (32 * 14 * 14) + c * (14 * 14) + h * 14 + w;
                            size_t dst_idx = i * (32 * 14 * 14) + c * (14 * 14) + h * 14 + w;
                            flat_out[dst_idx] = pool_out[src_idx];
                        }
                    }
                }
            }

            // 全连接层前向传播
            Tensor<float> fc_out = fc1.forward(flat_out);
            Tensor<float> pred = softmax.forward(fc_out);

            // 计算损失和准确率
            float loss = criterion.forward(pred, label_batch);
            float acc = calculateAccuracy(pred, label_batch);

            // 反向传播
            std::cout << "Starting backward propagation..." << std::endl;
            Tensor<float> grad_output = criterion.backward(pred, label_batch);
            Tensor<float> grad_softmax = softmax.backward(grad_output);
            Tensor<float> grad_fc1 = fc1.backward(grad_softmax);

            // 展平梯度（与前向传播的展平对应）
            std::vector<size_t> grad_flat_shape = {static_cast<size_t>(current_batch_size), 32 * 14 * 14};
            Tensor<float> grad_flat(grad_flat_shape);
            for (int i = 0; i < current_batch_size; i++)
            {
                for (int c = 0; c < 32; c++)
                {
                    for (int h = 0; h < 14; h++)
                    {
                        for (int w = 0; w < 14; w++)
                        {
                            size_t idx = i * (32 * 14 * 14) + c * (14 * 14) + h * 14 + w;
                            grad_flat[idx] = grad_fc1[idx];
                        }
                    }
                }
            }

            Tensor<float> grad_pool1 = pool1.backward(grad_flat);
            Tensor<float> grad_relu1 = relu1.backward(grad_pool1);
            Tensor<float> grad_conv1 = conv1.backward(grad_relu1);

            epoch_loss += loss;
            epoch_acc += acc;
            batch_count++;
        }

        // 计算平均损失和准确率
        epoch_loss /= batch_count;
        epoch_acc /= batch_count;

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << std::endl;
        std::cout << "Loss: " << epoch_loss << ", Accuracy: " << (epoch_acc * 100) << "%" << std::endl;
        std::cout << "================================" << std::endl;
    }

    // 6. 测试模型
    std::cout << "Testing model..." << std::endl;
    float test_acc = 0.0f;
    int test_batch_count = 0;

    for (int b = 0; b < test_size; b += batch_size)
    {
        // 计算当前批次大小
        int current_batch_size = std::min(batch_size, test_size - b);

        // 提取当前批次
        std::vector<size_t> img_batch_shape = {static_cast<size_t>(current_batch_size), 1, 28, 28};
        Tensor<float> img_batch(img_batch_shape);

        std::vector<size_t> label_batch_shape = {static_cast<size_t>(current_batch_size), 10};
        Tensor<float> label_batch(label_batch_shape);

        // 复制数据到批次张量
        for (int i = 0; i < current_batch_size; i++)
        {
            // 复制图像数据
            for (int c = 0; c < 1; c++)
            {
                for (int h = 0; h < 28; h++)
                {
                    for (int w = 0; w < 28; w++)
                    {
                        size_t src_idx = (b + i) * (1 * 28 * 28) + c * (28 * 28) + h * 28 + w;
                        size_t dst_idx = i * (1 * 28 * 28) + c * (28 * 28) + h * 28 + w;
                        img_batch[dst_idx] = test_images[src_idx];
                    }
                }
            }

            // 复制标签数据
            for (int l = 0; l < 10; l++)
            {
                size_t src_idx = (b + i) * 10 + l;
                size_t dst_idx = i * 10 + l;
                label_batch[dst_idx] = test_labels[src_idx];
            }
        }

        // 前向传播
        Tensor<float> conv_out = conv1.forward(img_batch);
        Tensor<float> relu_out = relu1.forward(conv_out);
        Tensor<float> pool_out = pool1.forward(relu_out);

        // 展平池化输出
        std::vector<size_t> flat_shape = {static_cast<size_t>(current_batch_size), 32 * 14 * 14};
        Tensor<float> flat_out(flat_shape);

        // 手动展平
        for (int i = 0; i < current_batch_size; i++)
        {
            for (int c = 0; c < 32; c++)
            {
                for (int h = 0; h < 14; h++)
                {
                    for (int w = 0; w < 14; w++)
                    {
                        size_t src_idx = i * (32 * 14 * 14) + c * (14 * 14) + h * 14 + w;
                        size_t dst_idx = i * (32 * 14 * 14) + c * (14 * 14) + h * 14 + w;
                        flat_out[dst_idx] = pool_out[src_idx];
                    }
                }
            }
        }

        // 全连接层前向传播
        Tensor<float> fc_out = fc1.forward(flat_out);
        Tensor<float> pred = softmax.forward(fc_out);

        // 计算准确率
        float acc = calculateAccuracy(pred, label_batch);
        test_acc += acc;
        test_batch_count++;
    }

    // 计算平均准确率
    test_acc /= test_batch_count;
    std::cout << "Test Accuracy: " << (test_acc * 100) << "%" << std::endl;
    std::cout << "================================" << std::endl;

    std::cout << "Training completed successfully!" << std::endl;

    return 0;
}
