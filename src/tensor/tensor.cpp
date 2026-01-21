#include "../../include/tensor/tensor.h"
#include <iostream>
#include <stdexcept>

//--------计算 strides 和 索引转换--------------//
template <typename T>
void Tensor<T>::computeStrides()
{
    strides.resize(shape.size());
    if (shape.empty())
        return;
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

template <typename T>
size_t Tensor<T>::computeIndex(const std::vector<size_t> &indices) const
{
    if (indices.size() != shape.size())
    {
        throw std::out_of_range("维度不符");
    }
    size_t index = 0;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (indices[i] >= shape[i])
        {
            throw std::out_of_range("超出维度范围" + std::to_string(i));
        }
        index += indices[i] * strides[i];
    }
    return index;
}

//-------------- 构造与析构函数--------------//

// 默认构造函数
template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &_shape, const T &initial_value)
    : shape(_shape)
{
    computeStrides();
    _size = 1;
    for (const auto &dim : shape)
    {
        _size *= dim;
    }
    data.resize(_size, initial_value);
}

// 拷贝构造函数
template <typename T>
Tensor<T>::Tensor(const Tensor &other)
    : data(other.data), shape(other.shape), strides(other.strides), _size(other._size)
{
}

// 移动构造函数
template <typename T>
Tensor<T>::Tensor(Tensor &&other) noexcept
    : data(std::move(other.data)), shape(std::move(other.shape)), strides(std::move(other.strides)), _size(other._size)
{
    other._size = 0;
}

// 析构函数
template <typename T>
Tensor<T>::~Tensor() {}

//-------------- 基本属性获取 --------------//

// 访问总元素数量
template <typename T>
size_t Tensor<T>::size() const
{
    return _size;
}

// 访问张量形状
template <typename T>
const std::vector<size_t> &Tensor<T>::getShape() const
{
    return shape;
}

// 打印张量信息
template <typename T>
void Tensor<T>::print() const
{
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); i++)
    {
        std::cout << shape[i];
        if (i != shape.size() - 1)
            std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < data.size(); i++)
    {
        std::cout << data[i];
        if (i != data.size() - 1)
            std::cout << ", ";
    }
    std::cout << "])" << std::endl;
}

// 访问张量维度
template <typename T>
size_t Tensor<T>::dim() const
{
    return shape.size();
}

// 单维索引访问
template <typename T>
T &Tensor<T>::operator[](const size_t &idx)
{
    if (idx >= _size)
    {
        throw std::out_of_range("单维索引超出范围");
    }
    return data[idx];
}

// 多维索引访问
template <typename T>
T &Tensor<T>::operator[](const std::vector<size_t> &indices)
{
    return data[computeIndex(indices)];
}

// 单维索引访问 (const 版本)
template <typename T>
const T &Tensor<T>::operator[](const size_t &idx) const
{
    if (idx >= _size)
    {
        throw std::out_of_range("单维索引超出范围");
    }
    return data[idx];
}

// 多维索引访问 (const 版本)
template <typename T>
const T &Tensor<T>::operator[](const std::vector<size_t> &indices) const
{
    return data[computeIndex(indices)];
}

//-------------- Tensor 运算重载 --------------//
// Tensor 加法运算重载
template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) const
{
    if (this->getShape() != other.getShape())
    {
        throw std::invalid_argument("张量形状不匹配");
    }
    Tensor<T> result(this->getShape());
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] + other.data[i];
    }
    return result;
}

// Tensor 减法运算重载
template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const
{
    if (this->getShape() != other.getShape())
    {
        throw std::invalid_argument("张量形状不匹配");
    }
    Tensor<T> result(this->getShape());
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] - other.data[i];
    }
    return result;
}

// Tensor 哈达玛积运算重载
template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const
{
    if (this->getShape() != other.getShape())
    {
        throw std::invalid_argument("张量形状不匹配");
    }
    Tensor<T> result(this->getShape());
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] * other.data[i];
    }
    return result;
}

// Tensor 除法运算重载
template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T> &other) const
{
    if (this->getShape() != other.getShape())
    {
        throw std::invalid_argument("张量形状不匹配");
    }
    Tensor<T> result(this->getShape());
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] / other.data[i];
    }
    return result;
}

// -------------- Tensor,val 运算重载 --------------//
// Tensor 与标量加法运算重载
template <typename T>
Tensor<T> Tensor<T>::operator+(const T &val) const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] + val;
    }
    return result;
}

// Tensor 与标量减法运算重载
template <typename T>
Tensor<T> Tensor<T>::operator-(const T &val) const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] - val;
    }
    return result;
}

// Tensor 与标量乘法运算重载
template <typename T>
Tensor<T> Tensor<T>::operator*(const T &val) const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] * val;
    }
    return result;
}

// Tensor 与标量除法运算重载
template <typename T>
Tensor<T> Tensor<T>::operator/(const T &val) const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] / val;
    }
    return result;
}

//-------------- val, Tensor 运算友元重载 --------------//
// 已在头文件中实现

//-------------- Tensor 赋值运算重载 --------------//
// Tensor 赋值运算符重载
template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other)
{
    if (this != &other)
    {
        this->data = other.data;
        this->shape = other.shape;
        this->strides = other.strides;
        this->_size = other._size;
    }
    return *this;
}

// Tensor 加法赋值运算符重载
template <typename T>
Tensor<T> &Tensor<T>::operator+=(const Tensor<T> &other)
{
    if (this->getShape() != other.getShape())
    {
        throw std::invalid_argument("张量形状不匹配");
    }
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] += other.data[i];
    }
    return *this;
}

// Tensor 减法赋值运算符重载
template <typename T>
Tensor<T> &Tensor<T>::operator-=(const Tensor<T> &other)
{
    if (this->getShape() != other.getShape())
    {
        throw std::invalid_argument("张量形状不匹配");
    }
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] -= other.data[i];
    }
    return *this;
}

// Tensor 乘法赋值运算符重载
template <typename T>
Tensor<T> &Tensor<T>::operator*=(const Tensor<T> &other)
{
    if (this->getShape() != other.getShape())
    {
        throw std::invalid_argument("张量形状不匹配");
    }
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] *= other.data[i];
    }
    return *this;
}

// Tensor 除法赋值运算符重载
template <typename T>
Tensor<T> &Tensor<T>::operator/=(const Tensor<T> &other)
{
    if (this->getShape() != other.getShape())
    {
        throw std::invalid_argument("张量形状不匹配");
    }
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] /= other.data[i];
    }
    return *this;
}

// Tensor 与标量加法赋值运算符重载
template <typename T>
Tensor<T> &Tensor<T>::operator+=(const T &val)
{
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] += val;
    }
    return *this;
}

// Tensor 与标量减法赋值运算符重载
template <typename T>
Tensor<T> &Tensor<T>::operator-=(const T &val)
{
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] -= val;
    }
    return *this;
}

// Tensor 与标量乘法赋值运算符重载
template <typename T>
Tensor<T> &Tensor<T>::operator*=(const T &val)
{
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] *= val;
    }
    return *this;
}

// Tensor 与标量除法赋值运算符重载
template <typename T>
Tensor<T> &Tensor<T>::operator/=(const T &val)
{
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] /= val;
    }
    return *this;
}

//-------------- 维度操作 --------------//
// Tensor 重塑形状
template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<size_t> &new_shape) const
{
    size_t new_size = 1;
    for (auto dim : new_shape)
    {
        new_size *= dim;
    }
    if (new_size != this->_size)
    {
        throw std::invalid_argument("新形状与原始数据大小不匹配");
    }
    Tensor<T> result(*this);
    result.shape = new_shape;
    result.computeStrides();
    return result;
}

// Tensor 转置两个维度
template <typename T>
Tensor<T> Tensor<T>::transpose(const int dim1, const int dim2) const
{
    if (dim1 < 0 || dim2 < 0 || static_cast<size_t>(dim1) >= shape.size() || static_cast<size_t>(dim2) >= shape.size())
    {
        throw std::out_of_range("维度索引超出范围");
    }
    Tensor<T> result(*this);
    std::swap(result.shape[dim1], result.shape[dim2]);
    std::swap(result.strides[dim1], result.strides[dim2]);
    return result;
}

// Tensor 压缩维度
template <typename T>
Tensor<T> Tensor<T>::squeeze(int dim) const
{
    Tensor<T> result(*this);
    if (dim == -1)
    {
        // 压缩所有为1的维度
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] != 1)
            {
                new_shape.push_back(shape[i]);
            }
        }
        result.shape = new_shape;
    }
    else
    {
        // 压缩指定维度
        if (dim < 0 || static_cast<size_t>(dim) >= shape.size())
        {
            throw std::out_of_range("维度索引超出范围");
        }
        if (shape[dim] != 1)
        {
            throw std::invalid_argument("指定维度大小不为1，无法压缩");
        }
        result.shape.erase(result.shape.begin() + dim);
    }
    result.computeStrides();
    return result;
}

// Tensor 扩展维度
template <typename T>
Tensor<T> Tensor<T>::unsqueeze(int dim) const
{
    if (dim < 0 || static_cast<size_t>(dim) >= shape.size() + 1)
    {
        throw std::out_of_range("维度索引超出范围");
    }
    Tensor<T> result(*this);
    result.shape.insert(result.shape.begin() + dim, 1);
    result.computeStrides();
    return result;
}

// Tensor 维度置换
template <typename T>
Tensor<T> Tensor<T>::permute(const std::vector<size_t> &dim_order) const
{
    if (dim_order.size() != shape.size())
    {
        throw std::invalid_argument("维度顺序大小不匹配");
    }
    Tensor<T> result(*this);
    std::vector<size_t> new_shape(shape.size());
    std::vector<size_t> new_strides(strides.size());
    for (size_t i = 0; i < dim_order.size(); i++)
    {
        if (dim_order[i] >= shape.size())
        {
            throw std::out_of_range("维度索引超出范围");
        }
        new_shape[i] = shape[dim_order[i]];
        new_strides[i] = strides[dim_order[i]];
    }
    result.shape = new_shape;
    result.strides = new_strides;
    return result;
}

//-------------- CNN核心运算 --------------//
// conv2d 二维卷积运算
/*
    Tensor: [批量大小, 输入通道数, 高度, 宽度]
    kernel: [输入通道数, 输出通道数，卷积核高度, 卷积核宽度]
    stride: 卷积步长，默认为{1,1}
    padding: 填充大小，默认为{0,0}
*/
template <typename T>
Tensor<T> Tensor<T>::conv2d(const Tensor<T> &kernel, const std::vector<size_t> &stride, const std::vector<size_t> &padding) const
{
    if (shape.size() != 4 || kernel.shape.size() != 4)
    {
        throw std::invalid_argument("输入张量必须为4D [batch, in_channels, height, width]，卷积核必须为4D [in_channels, out_channels, k_height, k_width]");
    }

    size_t batch_size = shape[0];
    size_t in_channels = shape[1];
    size_t in_height = shape[2];
    size_t in_width = shape[3];

    size_t kernel_in_channels = kernel.shape[0];
    size_t out_channels = kernel.shape[1];
    size_t kernel_height = kernel.shape[2];
    size_t kernel_width = kernel.shape[3];

    if (in_channels != kernel_in_channels)
    {
        throw std::invalid_argument("输入通道数与卷积核输入通道数不匹配");
    }

    size_t out_height = (in_height + 2 * padding[0] - kernel_height) / stride[0] + 1;
    size_t out_width = (in_width + 2 * padding[1] - kernel_width) / stride[1] + 1;

    std::vector<size_t> out_shape = {batch_size, out_channels, out_height, out_width};
    Tensor<T> result(out_shape, T(0));
    Tensor<T> padded_input = this->pad(padding);

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t oc = 0; oc < out_channels; ++oc)
        {
            for (size_t i = 0; i < out_height; ++i)
            {
                for (size_t j = 0; j < out_width; ++j)
                {
                    T sum = T(0);
                    for (size_t ic = 0; ic < in_channels; ++ic)
                    {
                        for (size_t m = 0; m < kernel_height; ++m)
                        {
                            for (size_t n = 0; n < kernel_width; ++n)
                            {
                                size_t pad_h = i * stride[0] + m;
                                size_t pad_w = j * stride[1] + n;
                                sum += padded_input[{b, ic, pad_h, pad_w}] * kernel[{ic, oc, m, n}];
                            }
                        }
                    }
                    result[{b, oc, i, j}] = sum;
                }
            }
        }
    }
    return result;
}

// pad 填充
template <typename T>
Tensor<T> Tensor<T>::pad(const std::vector<size_t> &padding, const T &pad_value) const
{
    if (shape.size() < 2 || padding.size() != 2)
    {
        throw std::invalid_argument("张量维度不足或填充参数错误");
    }
    std::vector<size_t> new_shape = shape;
    new_shape[shape.size() - 2] += 2 * padding[0]; // 高度方向填充
    new_shape[shape.size() - 1] += 2 * padding[1]; // 宽度方向填充

    Tensor<T> result(new_shape, pad_value);

    // 复制原始数据到新张量的正确位置
    for (size_t i = 0; i < shape[shape.size() - 2]; ++i)
    {
        for (size_t j = 0; j < shape[shape.size() - 1]; ++j)
        {
            std::vector<size_t> src_indices(shape.size(), 0);
            std::vector<size_t> dst_indices(shape.size(), 0);
            for (size_t d = 0; d < shape.size() - 2; ++d)
            {
                src_indices[d] = 0;
                dst_indices[d] = 0;
            }
            src_indices[shape.size() - 2] = i;
            src_indices[shape.size() - 1] = j;
            dst_indices[shape.size() - 2] = i + padding[0];
            dst_indices[shape.size() - 1] = j + padding[1];
            result[dst_indices] = (*this)[src_indices];
        }
    }
    return result;
}

// maxPool2d 二维最大池化（四维张量 [B, C, H, W]）
template <typename T>
Tensor<T> Tensor<T>::maxPool2d(const std::vector<size_t> &kernel_size, const std::vector<size_t> &stride, const std::vector<size_t> &padding) const
{
    // 输入维度：shape = [B, C, H, W]
    size_t in_height = shape[2];
    size_t in_width = shape[3];
    // 池化核尺寸
    size_t kernel_height = kernel_size[0];
    size_t kernel_width = kernel_size[1];

    // 计算输出尺寸
    size_t out_height = (in_height + 2 * padding[0] - kernel_height) / stride[0] + 1;
    size_t out_width = (in_width + 2 * padding[1] - kernel_width) / stride[1] + 1;

    // 构造输出形状：通道数不变，仅改变高宽
    std::vector<size_t> out_shape = shape;
    out_shape[2] = out_height;
    out_shape[3] = out_width;
    Tensor<T> result(out_shape, T(0));

    // 输入填充（和卷积共用同一个pad函数）
    Tensor<T> padded_input = this->pad(padding, T(0));

    // 预创建四维索引
    std::vector<size_t> in_indices(4, 0);
    std::vector<size_t> out_indices(4, 0);

    // 核心循环：批量B → 通道C → 输出高 → 输出宽
    for (size_t b = 0; b < shape[0]; ++b)
    {
        out_indices[0] = b;
        in_indices[0] = b;
        for (size_t c = 0; c < shape[1]; ++c) // 通道独立池化
        {
            out_indices[1] = c;
            in_indices[1] = c;
            for (size_t i = 0; i < out_height; ++i) // 输出高度
            {
                out_indices[2] = i;
                for (size_t j = 0; j < out_width; ++j) // 输出宽度
                {
                    out_indices[3] = j;
                    // 初始化窗口最大值（取极小值）
                    T max_val = std::numeric_limits<T>::lowest();

                    // 遍历池化窗口
                    for (size_t m = 0; m < kernel_height; ++m)
                    {
                        in_indices[2] = i * stride[0] + m;
                        for (size_t n = 0; n < kernel_width; ++n)
                        {
                            in_indices[3] = j * stride[1] + n;
                            // 更新窗口最大值
                            if (padded_input[in_indices] > max_val)
                            {
                                max_val = padded_input[in_indices];
                            }
                        }
                    }
                    // 写入最大值到输出张量
                    result[out_indices] = max_val;
                }
            }
        }
    }
    return result;
}

// avgPool2d 二维平均池化（四维张量 [B, C, H, W]）
template <typename T>
Tensor<T> Tensor<T>::avgPool2d(const std::vector<size_t> &kernel_size, const std::vector<size_t> &stride, const std::vector<size_t> &padding) const
{
    // 输入维度：shape = [B, C, H, W]
    size_t in_height = shape[2];
    size_t in_width = shape[3];
    // 池化核尺寸
    size_t kernel_height = kernel_size[0];
    size_t kernel_width = kernel_size[1];
    // 池化窗口元素总数（用于计算平均值）
    size_t kernel_elem = kernel_height * kernel_width;

    // 计算输出尺寸
    size_t out_height = (in_height + 2 * padding[0] - kernel_height) / stride[0] + 1;
    size_t out_width = (in_width + 2 * padding[1] - kernel_width) / stride[1] + 1;

    // 构造输出形状：通道数不变
    std::vector<size_t> out_shape = shape;
    out_shape[2] = out_height;
    out_shape[3] = out_width;
    Tensor<T> result(out_shape, T(0));

    // 输入填充
    Tensor<T> padded_input = this->pad(padding, T(0));

    // 预创建四维索引
    std::vector<size_t> in_indices(4, 0);
    std::vector<size_t> out_indices(4, 0);

    // 核心循环：和最大池化完全一致，仅计算逻辑不同
    for (size_t b = 0; b < shape[0]; ++b)
    {
        out_indices[0] = b;
        in_indices[0] = b;
        for (size_t c = 0; c < shape[1]; ++c)
        {
            out_indices[1] = c;
            in_indices[1] = c;
            for (size_t i = 0; i < out_height; ++i)
            {
                out_indices[2] = i;
                for (size_t j = 0; j < out_width; ++j)
                {
                    out_indices[3] = j;
                    // 初始化窗口求和值
                    T sum_val = T(0);

                    // 遍历池化窗口
                    for (size_t m = 0; m < kernel_height; ++m)
                    {
                        in_indices[2] = i * stride[0] + m;
                        for (size_t n = 0; n < kernel_width; ++n)
                        {
                            in_indices[3] = j * stride[1] + n;
                            sum_val += padded_input[in_indices];
                        }
                    }
                    // 计算平均值并写入
                    result[out_indices] = sum_val / kernel_elem;
                }
            }
        }
    }
    return result;
}

//-------------- 激活函数 --------------//
// relu
template <typename T>
Tensor<T> Tensor<T>::relu() const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = std::max(this->data[i], T(0));
    }
    return result;
}

// sigmoid
template <typename T>
Tensor<T> Tensor<T>::sigmoid() const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = T(1) / (T(1) + std::exp(-this->data[i]));
    }
    return result;
}

// tanh
template <typename T>
Tensor<T> Tensor<T>::tanh() const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = std::tanh(this->data[i]);
    }
    return result;
}

// softmax
template <typename T>
Tensor<T> Tensor<T>::softmax(int dim) const
{
    if (dim < -1 || (dim != -1 && static_cast<size_t>(dim) >= shape.size()))
    {
        throw std::out_of_range("维度索引超出范围");
    }

    Tensor<T> result(this->shape);
    std::vector<size_t> indices(shape.size(), 0);
    size_t dim_size;
    size_t outer_size, inner_size;

    // 处理 dim=-1（所有维度整体softmax）
    bool is_full_softmax = (dim == -1);
    if (is_full_softmax)
    {
        dim_size = 1;
        for (size_t s : shape)
            dim_size *= s; // 所有维度元素总数
        outer_size = 1;
        inner_size = 1;
    }
    else
    {
        dim_size = shape[dim];

        // 计算每个切片的softmax
        outer_size = 1;
        for (size_t i = 0; i < static_cast<size_t>(dim); ++i)
        {
            outer_size *= shape[i];
        }
        inner_size = 1;
        for (size_t i = dim + 1; i < shape.size(); ++i)
        {
            inner_size *= shape[i];
        }
    }

    for (size_t outer = 0; outer < outer_size; ++outer)
    {
        for (size_t inner = 0; inner < inner_size; ++inner)
        {
            T max_val = std::numeric_limits<T>::lowest();
            T sum_exp = T(0);

            if (is_full_softmax)
            {
                for (size_t d = 0; d < dim_size; ++d)
                {
                    // 更新最大值
                    if (data[d] > max_val)
                    {
                        max_val = data[d];
                    }
                }
                // 计算指数和
                for (size_t d = 0; d < dim_size; ++d)
                {
                    T exp_val = std::exp(data[d] - max_val);
                    sum_exp += exp_val;
                    result.data[d] = exp_val; // 暂存指数值
                }
            }
            else
            {
                // 还原outer到前dim维的索引
                size_t temp_outer = outer;
                for (int i = static_cast<int>(dim) - 1; i >= 0; --i)
                {
                    indices[i] = temp_outer % shape[i];
                    temp_outer /= shape[i];
                }
                // 还原inner到后shape.size()-dim-1维的索引
                size_t temp_inner = inner;
                for (size_t i = shape.size() - 1; i > static_cast<size_t>(dim); --i)
                {
                    indices[i] = temp_inner % shape[i];
                    temp_inner /= shape[i];
                }
                =
                    // 求当前切片的最大值
                    for (size_t d = 0; d < dim_size; ++d)
                {
                    indices[dim] = d;
                    size_t idx = computeIndex(indices);
                    if (data[idx] > max_val)
                    {
                        max_val = data[idx];
                    }
                }
                // 计算指数和
                for (size_t d = 0; d < dim_size; ++d)
                {
                    indices[dim] = d;
                    size_t idx = computeIndex(indices);
                    T exp_val = std::exp(data[idx] - max_val);
                    sum_exp += exp_val;
                    result.data[idx] = exp_val; // 暂存指数值
                }
            }

            // 归一化
            if (is_full_softmax)
            {
                for (size_t d = 0; d < dim_size; ++d)
                {
                    result.data[d] /= sum_exp;
                }
            }
            else
            {
                for (size_t d = 0; d < dim_size; ++d)
                {
                    indices[dim] = d;
                    size_t idx = computeIndex(indices);
                    result.data[idx] /= sum_exp;
                }
            }
        }
    }
    return result;
}

//-------------- 维度求和 --------------//
// sumDim 沿指定维度求和，保留维度（keepdim=true）或压缩维度（keepdim=false）
template <typename T>
Tensor<T> Tensor<T>::sumDim(int dim, bool keepdim) const
{
    if (dim < 0 || static_cast<size_t>(dim) >= shape.size())
    {
        throw std::out_of_range("维度索引超出范围");
    }

    // 构建输出形状
    std::vector<size_t> out_shape = shape;
    if (!keepdim)
    {
        out_shape.erase(out_shape.begin() + dim);
    }
    else
    {
        out_shape[dim] = 1;
    }

    Tensor<T> result(out_shape, T(0));
    size_t dim_size = shape[dim]; // 待求和维度的大小
    size_t pre_dim_stride = 1;    // 待求和维度之前的总步长
    size_t post_dim_stride = 1;   // 待求和维度之后的总步长

    // 计算前/后维度总步长
    for (int i = 0; i < dim; ++i)
    {
        pre_dim_stride *= shape[i];
    }
    for (size_t i = dim + 1; i < shape.size(); ++i)
    {
        post_dim_stride *= shape[i];
    }

    // 核心求和逻辑
    for (size_t pre = 0; pre < pre_dim_stride; ++pre)
    {
        for (size_t post = 0; post < post_dim_stride; ++post)
        {
            T sum_val = T(0);
            // 遍历待求和维度的所有元素
            for (size_t d = 0; d < dim_size; ++d)
            {
                // 计算原始张量的一维索引
                size_t idx = pre * dim_size * post_dim_stride + d * post_dim_stride + post;
                sum_val += data[idx];
            }
            // 计算结果张量的一维索引
            size_t res_idx = pre * post_dim_stride + post;
            result[res_idx] = sum_val;
        }
    }

    return result;
}

//-------------- 维度求均值 --------------//
// meanDim 沿指定维度求均值，保留维度（keepdim=true）或压缩维度（keepdim=false）
template <typename T>
Tensor<T> Tensor<T>::meanDim(int dim, bool keepdim) const
{
    if (dim < 0 || static_cast<size_t>(dim) >= shape.size())
    {
        throw std::out_of_range("维度索引超出范围");
    }

    // 先求和再除以维度大小
    Tensor<T> sum_tensor = this->sumDim(dim, keepdim);
    T dim_size = static_cast<T>(shape[dim]);
    sum_tensor /= dim_size;

    return sum_tensor;
}

//-------------- 张量扩展 --------------//
// expand 扩展张量维度（仅扩展大小为1的维度到指定大小，不复制数据）
template <typename T>
Tensor<T> Tensor<T>::expand(const std::vector<size_t> &expand_shape) const
{
    if (expand_shape.size() != shape.size())
    {
        throw std::invalid_argument("扩展形状维度数必须与原张量一致");
    }

    // 校验扩展形状合法性：仅能将大小为1的维度扩展到指定大小
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] != 1 && shape[i] != expand_shape[i])
        {
            throw std::invalid_argument("仅能扩展大小为1的维度，维度" + std::to_string(i) + "不满足");
        }
    }

    Tensor<T> result(*this);
    result.shape = expand_shape;
    result.computeStrides(); // 重新计算步长（逻辑上扩展，数据共享）
    return result;
}

//-------------- 张量展平 --------------//
// flatten 展平张量（从start_dim到end_dim，end_dim=-1表示展平到最后一维）
template <typename T>
Tensor<T> Tensor<T>::flatten(int start_dim, int end_dim) const
{
    if (start_dim < 0 || static_cast<size_t>(start_dim) >= shape.size())
    {
        throw std::out_of_range("start_dim超出维度范围");
    }
    if (end_dim == -1)
    {
        end_dim = static_cast<int>(shape.size()) - 1;
    }
    if (end_dim < start_dim || static_cast<size_t>(end_dim) >= shape.size())
    {
        throw std::invalid_argument("end_dim必须≥start_dim且在维度范围内");
    }

    // 构建展平后的形状
    std::vector<size_t> new_shape;
    // 保留start_dim之前的维度
    for (int i = 0; i < start_dim; ++i)
    {
        new_shape.push_back(shape[i]);
    }
    // 计算展平维度的总大小
    size_t flat_size = 1;
    for (int i = start_dim; i <= end_dim; ++i)
    {
        flat_size *= shape[i];
    }
    new_shape.push_back(flat_size);
    // 保留end_dim之后的维度
    for (size_t i = end_dim + 1; i < shape.size(); ++i)
    {
        new_shape.push_back(shape[i]);
    }

    return this->reshape(new_shape);
}

//-------------- 维度求最大值索引 --------------//
// argmax 沿指定维度求最大值的索引，保留维度（keepdim=true）或压缩维度（keepdim=false）
template <typename T>
Tensor<size_t> Tensor<T>::argmax(int dim, bool keepdim) const
{
    if (dim < 0 || static_cast<size_t>(dim) >= shape.size())
    {
        throw std::out_of_range("维度索引超出范围");
    }

    // 构建输出形状（输出为size_t类型的索引张量）
    std::vector<size_t> out_shape = shape;
    if (!keepdim)
    {
        out_shape.erase(out_shape.begin() + dim);
    }
    else
    {
        out_shape[dim] = 1;
    }

    Tensor<size_t> result(out_shape, 0);
    size_t dim_size = shape[dim]; // 待求索引维度的大小
    size_t pre_dim_stride = 1;    // 待求索引维度之前的总步长
    size_t post_dim_stride = 1;   // 待求索引维度之后的总步长

    // 计算前/后维度总步长
    for (int i = 0; i < dim; ++i)
    {
        pre_dim_stride *= shape[i];
    }
    for (size_t i = dim + 1; i < shape.size(); ++i)
    {
        post_dim_stride *= shape[i];
    }

    // 核心求索引逻辑
    for (size_t pre = 0; pre < pre_dim_stride; ++pre)
    {
        for (size_t post = 0; post < post_dim_stride; ++post)
        {
            T max_val = std::numeric_limits<T>::lowest();
            size_t max_idx = 0;
            // 遍历待求索引维度的所有元素
            for (size_t d = 0; d < dim_size; ++d)
            {
                size_t idx = pre * dim_size * post_dim_stride + d * post_dim_stride + post;
                if (data[idx] > max_val)
                {
                    max_val = data[idx];
                    max_idx = d;
                }
            }
            // 写入最大值索引到结果张量
            size_t res_idx = pre * post_dim_stride + post;
            result[res_idx] = max_idx;
        }
    }

    return result;
}

//-------------- 显式实例化常用类型 --------------//
// 显式实例化常用类型
template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;