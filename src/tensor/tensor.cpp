#include "../../include/tensor/tensor.h"
#include <iostream>
#include <stdexcept>

//--------计算 strides 和 索引转换--------------//
template<typename T>
void Tensor<T>::computeStrides()
{
    strides.resize(shape.size());
    if (shape.empty()) return;
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

template<typename T>
size_t Tensor<T>::computeIndex(const std::vector<size_t>& indices) const
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

//默认构造函数
template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& _shape, const T& initial_value)
    : shape(_shape)
{
    computeStrides();
    _size = 1;
    for (const auto& dim : shape)
    {
        _size *= dim;
    }
    data.resize(_size, initial_value);
}

//拷贝构造函数
template<typename T>
Tensor<T>::Tensor(const Tensor& other)
    : data(other.data), shape(other.shape), strides(other.strides), _size(other._size)
{
}

//移动构造函数
template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : data(std::move(other.data)), shape(std::move(other.shape)), strides(std::move(other.strides)), _size(other._size)
{
    other._size = 0;
}

//析构函数
template<typename T>
Tensor<T>::~Tensor(){}



//-------------- 基本属性获取 --------------//

//访问总元素数量
template<typename T>
size_t Tensor<T>::size() const
{
    return _size;
}

//访问张量形状
template<typename T>
const std::vector<size_t>& Tensor<T>::getShape() const
{
    return shape;
}

//打印张量信息
template<typename T>
void Tensor<T>::print() const
{
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); i++)
    {
        std::cout << shape[i];
        if (i != shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < data.size(); i++)
    {
        std::cout << data[i];
        if (i != data.size() - 1) std::cout << ", ";
    }
    std::cout << "])" << std::endl;
}

//访问张量维度
template<typename T>
size_t Tensor<T>::dim() const
{
    return shape.size();
}

//单维索引访问
template<typename T>
T& Tensor<T>::operator[](const size_t& idx)
{
    if (idx >= _size)
    {
        throw std::out_of_range("单维索引超出范围");
    }
    return data[idx];
}

//多维索引访问
template<typename T>
T& Tensor<T>::operator[](const std::vector<size_t>& indices)
{
    return data[computeIndex(indices)];
}

//单维索引访问 (const 版本)
template<typename T>
const T& Tensor<T>::operator[](const size_t& idx) const
{
    if (idx >= _size)
    {
        throw std::out_of_range("单维索引超出范围");
    }
    return data[idx];
}

//多维索引访问 (const 版本)
template<typename T>
const T& Tensor<T>::operator[](const std::vector<size_t>& indices) const
{
    return data[computeIndex(indices)];
}


//-------------- Tensor 运算重载 --------------//
// Tensor 加法运算重载
template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const
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
template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const
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
template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const
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
template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const
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
template<typename T>
Tensor<T> Tensor<T>::operator+(const T& val) const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] + val;
    }
    return result;
}

// Tensor 与标量减法运算重载
template<typename T>
Tensor<T> Tensor<T>::operator-(const T& val) const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] - val;
    }
    return result;
}

// Tensor 与标量乘法运算重载
template<typename T>
Tensor<T> Tensor<T>::operator*(const T& val) const
{
    Tensor<T> result(this->shape);
    for (size_t i = 0; i < this->size(); ++i)
    {
        result[i] = this->data[i] * val;
    }
    return result;
} 

// Tensor 与标量除法运算重载
template<typename T>
Tensor<T> Tensor<T>::operator/(const T& val) const
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
template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other)
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
template<typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other)
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
template<typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other)
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
template<typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other)
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
template<typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other)
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
template<typename T>
Tensor<T>& Tensor<T>::operator+=(const T& val)
{
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] += val;
    }
    return *this;
}

// Tensor 与标量减法赋值运算符重载
template<typename T>
Tensor<T>& Tensor<T>::operator-=(const T& val)
{
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] -= val;
    }
    return *this;
}

// Tensor 与标量乘法赋值运算符重载
template<typename T>
Tensor<T>& Tensor<T>::operator*=(const T& val)
{
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] *= val;
    }
    return *this;
}

// Tensor 与标量除法赋值运算符重载
template<typename T>
Tensor<T>& Tensor<T>::operator/=(const T& val)
{
    for (size_t i = 0; i < this->size(); ++i)
    {
        this->data[i] /= val;
    }
    return *this;
}


//-------------- 显式实例化常用类型 --------------//
// 显式实例化常用类型
template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;