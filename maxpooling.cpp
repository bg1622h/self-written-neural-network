#include "maxpooling.h"

maxpoolinglayer::maxpoolinglayer(TensorSize size, uint64_t scale){
    inputsize.width = size.width;
    inputsize.height = size.height;
    inputsize.depth = size.depth;
    outputsize.width = size.width / scale;
    outputsize.height = size.height / scale;
    outputsize.depth = size.depth;
    mask = Tensor(inputsize);
    this->scale = scale;
}

void maxpoolinglayer::full_clear(){
    mask = Tensor(inputsize);
}

Tensor maxpoolinglayer::Forward(const Tensor& X) {
    full_clear();
    Tensor output(outputsize);
    for (uint64_t d = 0; d < inputsize.depth; ++d) {
        for (uint64_t i = 0; i < inputsize.height; i += scale) {
            for (uint64_t j = 0; j < inputsize.width; j += scale) {
                uint64_t imax = i;
                uint64_t jmax = j;
                long double max = X(d, i, j);
                for (uint64_t y = i; y < std::min(i + scale,inputsize.height); ++y) {
                    for (uint64_t x = j; x < std::min(j + scale,inputsize.width); ++x) {
                        long double val = X(d, y, x);
                        if (val > max) {
                            max = val;
                            imax = y;
                            jmax = x;
                        }
                        //max = ::max(val, max);
                    }
                }
                if (i / scale < outputsize.height && j / scale < outputsize.width) {
                    output(d, i / scale, j / scale) = max;
                    mask(d, imax, jmax) = 1;
                }
            }
        }
    }
    return output;
}
Tensor maxpoolinglayer::Backward(const Tensor& dout, const Tensor& X) {
    Tensor dx(inputsize); // тензор градиентов
    for (uint64_t d = 0; d < inputsize.depth; ++d) {
        for (uint64_t i = 0; i < inputsize.height; ++i) {
            for (uint64_t j = 0; j < inputsize.width; ++j) {
                dx(d, i, j) = dout(d, i / scale, j / scale) * mask(d, i, j);
                mask(d, i, j) = 0;
            }
        }
    }
    return dx;
}