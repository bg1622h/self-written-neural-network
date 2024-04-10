#pragma once
#include "layer.h"
class Relulayer :public layer {
    TensorSize size; // размер слоя
public:
    Relulayer(TensorSize size); 
    
    Tensor Forward(const Tensor& X);
    Tensor Backward(const Tensor& dout, const Tensor& X);
};