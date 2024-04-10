#include "fullconectLayer.h"

long double fullyconectedlayer::tahn(long double x) {
    return (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
}

long double fullyconectedlayer::dxtahn(long double x) {
    return 1.0 - (tahn(x) * tahn(x));
}

fullyconectedlayer::req fullyconectedlayer::getX(const Tensor& X, uint64_t n) {
    int i = n / X.dw;
    //n -= i * X.dw;
    n %= X.dw;
    int j = n / X.size.depth;
    int d = n % X.size.depth;
    return req(i, j, d);
}

fullyconectedlayer::fullyconectedlayer(TensorSize size, int outputs) :df(1, 1, outputs), distribution(0.0, sqrt(2.0 / (size.height * size.width * size.depth))) {
    inputsize = size; // в input по логике вещей, все уже тоже должно будет выровняться, до (1,1,k), или мы сами
    outputsize.width = 1;
    outputsize.height = 1;
    outputsize.depth = outputs;
    this->inputs = size.height * size.width * size.depth; // число входных
    this->outputs = outputs; // на выходе
    b = std::vector<long double>(outputs);
    db = std::vector<long double>(outputs);
    initweigts();
}
void fullyconectedlayer::initweigts() {
    w(inputs, outputs, 1);
    dw(inputs, outputs, 1);
    for (uint64_t i = 0; i < outputs; ++i) {
        for (uint64_t j = 0; j < inputs; ++j) {
            w(0, i, j) = distribution(generator);
        }
        b[i] = 0.01;
    }
}
void fullyconectedlayer::full_clear() {
    df(1, 1, outputs);
    db = std::vector<long double>(outputs, 0);
    dw(inputs, outputs, 1);
}

Tensor fullyconectedlayer::Forward(const Tensor& X) {
    full_clear();
    Tensor output(outputsize);
    for (uint64_t i = 0; i < outputs; ++i) {
        long double sum = b[i];
        for (uint64_t j = 0; j < inputs; ++j) {
            req r = getX(X, j);
            sum += w(0, i, j) * X(r.d, r.i, r.j);
        }
        output(i, 0, 0) = tahn(sum);
        df(i, 0, 0) = dxtahn(sum);
    }
    return output;
}
Tensor fullyconectedlayer::Backward(const Tensor& dout, const Tensor& X) {
    // db= dout*df, dw=(dout*df) * X, dx=wT * (dout * df)
    //int all = 0;
    for (uint64_t i = 0; i < outputs; ++i) {
        df(i, 0, 0) *= dout(i, 0, 0);
    }
    for (uint64_t i = 0; i < outputs; ++i) {
        uint64_t add = 0;
        for (uint64_t j = 0; j < inputs; ++j) {
            req r = getX(X, j);
            dw(0, i, j) += df(i, 0, 0) * X(r.d, r.i, r.j);
            //tdw(0, i, j) += dw(0, i, j);
            //dw(0, i, j) += df(i, 0, 0) * X(r.d, r.i, r.j) * (1.0 - beta);
        }
        //all++;
        db[i] += df(i, 0, 0);
        //tdb[i] += df(i, 0, 0);
        //db[i] += df(i, 0, 0) * (1.0 - beta);
    }
    //cout << "dead_neurons: " << calcdead << " from " << all << "\n";
    Tensor dx(inputsize);
    for (uint64_t j = 0; j < inputs; ++j) {
        long double sum = 0;
        for (uint64_t i = 0; i < outputs; ++i) {
            sum += w(0, i, j) * df(i, 0, 0);
        }
        req r = getX(dx, j);
        dx(r.d, r.i, r.j) = sum;
    }
    return dx;
}

void fullyconectedlayer::updateweights(double nu) { // используем градиент весов
    for (uint64_t i = 0; i < outputs; ++i) {
        for (uint64_t j = 0; j < inputs; ++j) {
            for (uint64_t d = 0; d < 1; ++d) {
                //sdw(d, i, j) += sqr(dw(d, i, j));

                w(d, i, j) = w(d, i, j) - (nu * dw(d, i, j));  // / sqrt(eps + sdw(d, i, j));

                dw(d, i, j) = 0;
                //gdw(d, i, j) = gamma * gdw(d, i, j) + (1.0 - gamma) * (dw(d, i, j)) * (dw(d, i, j));
                //w(d, i, j) -= (nu * dw(d, i, j)) / sqrt(eps + gdw(d, i, j));// + sumdx[k](d,i,j) * momentum;
                //dw(d, i, j) = 0;
                //w(0, i, j) -= alpha * dw(0, i, j);
                //dw(0, i, j) *= beta;
            }
        }
        uint64_t k = i;
        //sdb[k] += sqr(db[k]);
        b[k] = b[k] - (nu * db[k]); // / sqrt(eps + sdb[k]);

        db[k] = 0;
        //gdb[k] = gamma * gdb[k] + (1.0 - gamma) * (db[k]) * (db[k]);
        //b[k] -= (nu * db[k]) / sqrt(eps + gdb[k]);
        //db[k] = 0;
        //b[i] -= alpha * db[i];
        //db[i] *= beta;
        //RMSPROP
    }
}