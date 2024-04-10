#include "tripletlosslayer.h"

long double dist(const Tensor& expect, const Tensor& output) { // (L2 норма), сумма квадратов расстояний по измерению	x1^2+x2^2+x3^2...+xn^2
    long double ans = 0;
    for (uint64_t d = 0; d < output.size.depth; d++) {
        for (uint64_t i = 0; i < output.size.height; i++) {
            for (uint64_t j = 0; j < output.size.width; j++) {
                long double x1 = expect(d, i, j);
                long double x2 = output(d, i, j);
                ans += (x1 - x2) * (x1 - x2);
                //ans += sqr(x1) - 2 * x1 * x2 + sqr(x2);
                //res(d, i, j) = sqr(expect(d, i, j) - output(d, i, j)); // расстояние в одном из направлений
                //ans += sqr(expect(d, i, j) - output(d, i, j));
            }
        }
    }
    return ans;
}

TripletLosslayer::TripletLosslayer(TensorSize size, long double margin) {
    this->size = size;
    //mul = size.depth * size.height * size.width;
    this->margin = margin;
}

Tensor TripletLosslayer::Forward(const Tensor& X) {
    return X;
}

long double TripletLosslayer::Tripletloss(const Tensor& a, const Tensor& p, const Tensor& n) {
    long double dpr = dist(a, p);
    long double dpn = dist(a, n);
    long double res = dpr + margin - dpn;
    return std::max(res, (long double)0.0);
}

bool TripletLosslayer::check_in(const Tensor& a, const Tensor& p) {
    long double d = dist(a, p) - margin;
    return d <= 0;
}

std::vector<Tensor> TripletLosslayer::dxTripletloss(const Tensor& a, const Tensor& p, const Tensor& n) {
    // A - anchor P - positive N - negative
    std::vector<Tensor>out(3, Tensor(a));
    long double req = Tripletloss(a, p, n);
    {
        for (uint64_t d = 0; d < a.size.depth; ++d) {
            for (uint64_t i = 0; i < a.size.height; ++i) {
                for (uint64_t j = 0; j < a.size.width; ++j) {
                    if (req > 0) {
                        out[0](d, i, j) = 2.0 * (n(d, i, j) - p(d, i, j));
                    }
                    else {
                        out[0](d, i, j) = 0.0;
                    }
                }
            }
        }
    }// A
    {
        for (uint64_t d = 0; d < a.size.depth; ++d) {
            for (uint64_t i = 0; i < a.size.height; ++i) {
                for (uint64_t j = 0; j < a.size.width; ++j) {
                    if (req > 0) {
                        out[1](d, i, j) = 2.0 * (p(d, i, j) - a(d, i, j));
                    }
                    else {
                        out[1](d, i, j) = 0.0;
                    }
                }
            }
        }

    }// P
    {
        for (uint64_t d = 0; d < a.size.depth; ++d) {
            for (uint64_t i = 0; i < a.size.height; ++i) {
                for (uint64_t j = 0; j < a.size.width; ++j) {
                    if (req > 0) {
                        out[2](d, i, j) = 2.0 * (a(d, i, j) - n(d, i, j));
                    }
                    else {
                        out[2](d, i, j) = 0.0;
                    }
                }
            }
        }
    }// N
    return out;
}