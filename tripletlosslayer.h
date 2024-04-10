#include "layer.h"
long double dist(const Tensor& expect, const Tensor& output);

class TripletLosslayer :public layer {
public:
    TensorSize size;
    long double margin;

    TripletLosslayer(TensorSize size, long double margin);
    Tensor Forward(const Tensor& X) override;
    long double Tripletloss(const Tensor& a, const Tensor& p, const Tensor& n);
    bool check_in(const Tensor& a, const Tensor& p);
    std::vector<Tensor> dxTripletloss(const Tensor& a, const Tensor& p, const Tensor& n);
};
