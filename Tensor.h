#pragma once
#include <vector>
#include <fstream>
struct TensorSize {
    uint64_t width;
    uint64_t height;
    uint64_t depth; // глубина, или сколько каналов в im
};

class Tensor {
public:
    TensorSize size;
    std::vector<long double>values; // значения
    uint64_t dw;
    Tensor(uint64_t w, uint64_t h, uint64_t d);
    Tensor(const TensorSize& size);
    Tensor() {
        
    }
    void set(uint64_t d, uint64_t i, uint64_t j, long double x);
    void clear();
    long double& operator()(uint64_t d, uint64_t i, uint64_t j); // индексация
    long double operator()(uint64_t d, uint64_t i, uint64_t j) const; // индексация (другой вариант)
    TensorSize getsize();
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor); // вывод тензора
    friend Tensor& operator+=(Tensor& left, const Tensor& right);
    friend Tensor& operator-=(Tensor& left, const Tensor& right);
    friend Tensor& operator*=(Tensor& left, const int* mul);
};

