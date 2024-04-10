#pragma once
#include "layer.h"
class maxpoolinglayer : public layer {
public:
    TensorSize inputsize;
    TensorSize outputsize;
    Tensor mask;
    /*
    так как при прямом проходе на выход пошёл лишь один элемент,
    то при обратном распространении полученное значение градиента следующего слоя нужно записать
    в место расположения максимального элемента.

    создать бинарную маску, где каждый элемент будет равен единице,
    если в заданной клетке расположен максимум,
    и нулю в противном случае.
    Данную маску можно посчитать при прямом проходе,
    а при обратном распространении ошибки умножить её на градиенты следующего слоя.
    */
    uint64_t scale;
    maxpoolinglayer(TensorSize size, uint64_t scale);
    void full_clear() override;

    Tensor Forward(const Tensor& X) override; 
    Tensor Backward(const Tensor& dout, const Tensor& X) override;
};