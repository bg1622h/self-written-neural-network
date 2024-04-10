#pragma once
#include "Tensor.h"
#include <cassert>

Tensor::Tensor(uint64_t w, uint64_t h, uint64_t d){
    size.width = w;
    size.height = h;
    size.depth = d;
    dw = d * w;
    values = std::vector<long double>(w * h * d, 0);
}

Tensor::Tensor(const TensorSize& size) {
    Tensor(size.width, size.height, size.depth);
}

void Tensor::set(uint64_t d, uint64_t i, uint64_t j, long double x) {
    values[i * dw + j * size.depth + d] = x;
}

void Tensor::clear() {
    values = std::vector<long double>(size.width * size.height * size.depth, 0);
}

TensorSize Tensor::getsize() {
    return size;
}

long double& Tensor::operator()(uint64_t d, uint64_t i, uint64_t j) {
    uint64_t val = i * dw + j * size.depth + d;
    if (val < values.size()) {
        return values[i * dw + j * size.depth + d];
    }
    else {
        assert(false);
    }
}
long double Tensor::operator()(uint64_t d, uint64_t i, uint64_t j) const {
    uint64_t val = i * dw + j * size.depth + d;
    if (val < values.size()) {
        return values[i * dw + j * size.depth + d];
    }
    else {
        //!!assert(false);
        return 0;
    }
}
Tensor& operator+=(Tensor& left, const Tensor& right) { // складываются тензоры равных размеров
    for (uint64_t d = 0; d < left.size.depth; ++d) {
        for (uint64_t i = 0; i < left.size.height; ++i) {
            for (uint64_t j = 0; j < left.size.width; ++j) {
                left(d, i, j) += right(d, i, j);
            }
        }
    }
    return left;
}
Tensor& operator-=(Tensor& left, const Tensor& right) { // складываются тензоры равных размеров
    for (uint64_t d = 0; d < left.size.depth; ++d) {
        for (uint64_t i = 0; i < left.size.height; ++i) {
            for (uint64_t j = 0; j < left.size.width; ++j) {
                left(d, i, j) -= right(d, i, j);
            }
        }
    }
    return left;
}
Tensor& operator*=(Tensor& left, const int& mul) { // складываются тензоры равных размеров
    for (uint64_t d = 0; d < left.size.depth; ++d) {
        for (uint64_t i = 0; i < left.size.height; ++i) {
            for (uint64_t j = 0; j < left.size.width; ++j) {
                left(d, i, j) *= mul;
            }
        }
    }
    return left;
}


std::ostream& operator<<(std::ostream& os, const Tensor& tensor) { // код взят, потому что это отладочная инфа
    for (uint64_t d = 0; d < tensor.size.depth; ++d) {
        for (uint64_t i = 0; i < tensor.size.height; ++i) {
            for (uint64_t j = 0; j < tensor.size.width; ++j)
                os << tensor.values[i * tensor.dw + j * tensor.size.depth + d] << " ";

            os << std::endl;
        }

        os << std::endl;
    }

    return os;
}
