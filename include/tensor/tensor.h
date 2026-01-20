#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

template<typename T>
class Tensor
{
private:
    std::vector<T> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t _size;
    //计算 strides 和 索引转换
    void computeStrides();
    size_t computeIndex(const std::vector<size_t>& indices) const;

public:
    //构造与析构函数
    Tensor(const std::vector<size_t>& _shape, const T& initial_value = T()); //赋值构造
    Tensor(const Tensor& other); //拷贝构造
    Tensor(Tensor&& other) noexcept; //移动构造

    ~Tensor();

    //基本属性获取
    size_t size() const; //访问总元素数量
    const std::vector<size_t>& getShape() const; //访问张量形状
    void print() const; //打印张量信息
    size_t dim() const; //访问张量维度


    //单维索引访问重载
    T& operator[](const size_t& idx);
    const T& operator[](const size_t& idx) const;

    //多维索引访问重载
    T& operator[](const std::vector<size_t>& indices);
    const T& operator[](const std::vector<size_t>& indices) const;

    //Tensor 运算重载
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; //哈达玛积
    Tensor operator/(const Tensor& other) const;

    //Tensor,val 运算重载
    Tensor operator+(const T& val) const;
    Tensor operator-(const T& val) const;
    Tensor operator*(const T& val) const;
    Tensor operator/(const T& val) const;

    //val,Tensor 运算友元重载
    friend Tensor operator+(const T& val, const Tensor& tensor){return tensor + val;}
    friend Tensor operator-(const T& val, const Tensor& tensor){return Tensor(tensor.shape, val) - tensor;}
    friend Tensor operator*(const T& val, const Tensor& tensor){return tensor * val;}
    friend Tensor operator/(const T& val, const Tensor& tensor){return Tensor(tensor.shape, val) / tensor;}

    //Tensor 赋值运算重载
    Tensor& operator=(const Tensor& other);
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    Tensor& operator+=(const T& val);
    Tensor& operator-=(const T& val);
    Tensor& operator*=(const T& val);
    Tensor& operator/=(const T& val);

    //功能函数
    Tensor reshape(const std::vector<size_t>& new_shape) const;

};

using Tensorf = Tensor<float>;
using Tensord = Tensor<double>;
using Tensori = Tensor<int>;