#pragma once
#include "Tensor.h"
class layer {
public:
    layer() {

    }
    virtual Tensor Forward(const Tensor& X) {
        return Tensor(1, 1, 1);
    }
    virtual Tensor Backward(const Tensor& dout, const Tensor& X) {
        return Tensor(1, 1, 1);
    }
    virtual Tensor Test(const Tensor& X) {
        return Forward(X);
    }
    virtual void updateweights(double alpha) {

    }
    virtual void printw(int k) {

    }
    virtual void full_clear() {
        return;
    }
};