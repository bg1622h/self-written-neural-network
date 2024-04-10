#include "layer.h"
#include <random>
class fullyconectedlayer : public layer { // скрытый слой
public:
    TensorSize inputsize;
    TensorSize outputsize;
    std::default_random_engine generator;
    std::normal_distribution<long double> distribution;
    uint64_t inputs; // число нейронов на входе
    uint64_t outputs; // число нейронов ны выходе

    long double eps = 1e-7;

    long double tahn(long double x);
    long double dxtahn(long double x);
    Tensor df; // значения производных
    Tensor w; // это матрица (по весам нейронов)
    Tensor dw; // матрица по градиентам

    Tensor sdw;
    std::vector<long double>b;
    std::vector<long double>db;
    struct req {
        uint64_t i, j, d;
        req(uint64_t vi, uint64_t vj, uint64_t vd) :i(vi), j(vj), d(vd) {

        }
    };
    req getX(const Tensor& X, uint64_t n); 
    fullyconectedlayer(TensorSize size, int outputs);
    void initweigts();
    void full_clear() override;
    Tensor Forward(const Tensor& X) override;
    Tensor Backward(const Tensor& dout, const Tensor& X) override;
    void updateweights(double nu);
};

