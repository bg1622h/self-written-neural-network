#pragma once
#include "layer.h"
#include <random>
#include <vector>

class ConvLayer : public layer { // именно этот слой потинхоньку вытягивает изображение (делая выходную глубину нужным нам числом), чтобы уже массив получли полносвязный слой
public:
    //double momentum = 0.5;
    std::default_random_engine generator; // генератор случайных чисел
    std::normal_distribution<long double> distribution; // с нормальным распределением
    TensorSize inputsize;
    TensorSize outputsize;

    std::vector<Tensor>w; // фильтры
    std::vector<double>b; // bias E(f) + b

    std::vector<Tensor>dw; // градиент фильтра
    std::vector<double>db; // градиент смещения

    long double eps = 1e-7;

    uint64_t P; // padding
    uint64_t S; // шаг
    uint64_t fc; // всего фильтров
    uint64_t fs; // размерность ядра
    uint64_t fd;
    void initweights();
    ConvLayer(TensorSize size, uint64_t fc, uint64_t fs, uint64_t P, uint64_t S);
    void setw(uint64_t num, uint64_t d,uint64_t i, uint64_t j, long double x);
    void setb(uint64_t num, long double x);
    
    Tensor Forward(const Tensor& X) override;
    
    
    
    Tensor Backward(const Tensor& dout, const Tensor& X) override;
    // обновление весов
    void updateweights(double nu) override;
};