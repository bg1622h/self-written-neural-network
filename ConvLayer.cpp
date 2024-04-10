#include "ConvLayer.h"

void ConvLayer::initweights() {
    for (uint64_t k = 0; k < fc; ++k) {
        for (uint64_t i = 0; i < fs; ++i) {
            for (uint64_t j = 0; j < fs; ++j) {
                for (uint64_t d = 0; d < fd; ++d) {
                    w[k](d, i, j) = distribution(generator);// инициализация
                }
            }
        }
        b[k] = 0.01; // смещение
    }
}

ConvLayer::ConvLayer(TensorSize size, uint64_t fc, uint64_t fs, uint64_t P, uint64_t S) : distribution(0.0, sqrt(2.0 / (fs * fs * size.depth))) {
    inputsize.width = size.width;
    inputsize.height = size.height;
    inputsize.depth = size.depth;
    outputsize.width = (size.width - fs + 2 * P) / S + 1;
    outputsize.height = (size.height - fs + 2 * P) / S + 1;
    outputsize.depth = fc;
    this->P = P; // пэддинг
    this->S = S; // шаг
    this->fc = fc; // число
    this->fs = fs; // размерность
    this->fd = size.depth; // глубина
    w = std::vector<Tensor>(fc, Tensor(fs, fs, fd));
    dw = std::vector<Tensor>(fc, Tensor(fs, fs, fd));
    b = std::vector<double>(fc, 0);
    db = std::vector<double>(fc, 0);
    initweights();
}

void ConvLayer::setw(uint64_t num, uint64_t d,uint64_t i, uint64_t j, long double x) {
    w[num](d, i, j) = x;
}

void ConvLayer::setb(uint64_t num, long double x) {
    b[num] = x;
}

Tensor ConvLayer::Forward(const Tensor& X) {
    Tensor output(outputsize);
    for (uint64_t f = 0; f < fc; ++f) {
        for (uint64_t y = 0; y < outputsize.height; ++y) {
            for (uint64_t x = 0; x < outputsize.width; ++x) {
                long double sum = b[f]; // добавляем +
                for (uint64_t i = 0; i < fs; ++i) {
                    for (uint64_t j = 0; j < fs; ++j) {
                        int i0 = S * y + i - P;
                        int j0 = S * x + j - P;
                        // поскольку вне границ входного тензора элементы нулевые, то просто игнорируем их
                        if (i0 < 0 || i0 >= inputsize.height || j0 < 0 || j0 >= inputsize.width)
                            continue;
                        for (uint64_t c = 0; c < fd; ++c) {
                            sum += X(c, i0, j0) * w[f](c, i, j);
                        }
                    }
                }
                output(f, y, x) = sum;
                /*
                что мы тут по-факту делаем:
                    создаём маленькие матрицы тензоров (проход по глубине) (неявно, когда проходимся по глубине)
                    объединяем их все сразуу (sum)
                    добавляем смещение филтра (sum=b[f], для каждой ячейки)
                */
            }
        }
    }
    return output;
}

Tensor ConvLayer::Backward(const Tensor& dout, const Tensor& X) {
    TensorSize size; // размер дельт

    // расчитываем размер для дельт
    size.height = S * (outputsize.height - 1) + 1;
    size.width = S * (outputsize.width - 1) + 1;
    size.depth = outputsize.depth;

    Tensor deltas(size); // создаём тензор для дельт

    // расчитываем значения дельт
    for (uint64_t d = 0; d < size.depth; ++d)
        for (uint64_t i = 0; i < outputsize.height; ++i)
            for (uint64_t j = 0; j < outputsize.width; ++j)
                deltas(d, i * S, j * S) = dout(d, i, j);

    // расчитываем градиенты весов фильтров и смещений
    // градиент фильтров - выполнить слегка модифицированную свёртку входного тензора и тензора градиентов
    for (uint64_t f = 0; f < fc; ++f) {
        for (uint64_t y = 0; y < size.height; ++y) {
            for (uint64_t x = 0; x < size.width; ++x) {
                long double delta = deltas(f, y, x); // запоминаем значение градиента

                for (uint64_t i = 0; i < fs; ++i) {
                    for (uint64_t j = 0; j < fs; ++j) {
                        uint64_t i0 = i + y - P;
                        uint64_t j0 = j + x - P;

                        // игнорируем выходящие за границы элементы
                        if (i0 < 0 || i0 >= inputsize.height || j0 < 0 || j0 >= inputsize.width)
                            continue;

                        // наращиваем градиент фильтра
                        for (uint64_t c = 0; c < fd; ++c) {
                            //dw[f](c, i, j) += (1.0 - beta) * delta * X(c, i0, j0);
                            dw[f](c, i, j) += delta * X(c, i0, j0);
                            //tdw[f](c, i, j) += delta * X(c, i0, j0);
                        }
                    }
                }
                db[f] += delta;
                //tdb[f] += delta;
                //db[f] += (1.0 - beta) * delta; // наращиваем градиент смещения
            }
        }
    }

    int pad = fs - 1 - P; // заменяем величину дополнения
    Tensor dX(inputsize); // создаём тензор градиентов по входу

    // расчитываем значения градиента
    for (uint64_t y = 0; y < inputsize.height; ++y) {
        for (uint64_t x = 0; x < inputsize.width; ++x) {
            for (uint64_t c = 0; c < fd; ++c) {
                long double sum = 0; // сумма для градиента

                // идём по всем весовым коэффициентам фильтров
                for (uint64_t i = 0; i < fs; ++i) {
                    for (uint64_t j = 0; j < fs; ++j) {
                        uint64_t i0 = y + i - pad;
                        uint64_t j0 = x + j - pad;

                        // игнорируем выходящие за границы элементы
                        if (i0 < 0 || i0 >= size.height || j0 < 0 || j0 >= size.width)
                            continue;

                        // суммируем по всем фильтрам
                        for (uint64_t f = 0; f < fc; ++f)
                            sum += w[f](c, fs - 1 - i, fs - 1 - j) * deltas(f, i0, j0); // добавляем произведение повёрнутых фильтров на дельты
                    }
                }

                dX(c, y, x) = sum; // записываем результат в тензор градиента
            }
        }
    }
    return dX; // возвращаем тензор градиентов
}
void ConvLayer::updateweights(double nu) { //nu - скорость обучения
    for (uint64_t k = 0; k < fc; ++k) {
        for (uint64_t i = 0; i < fs; ++i) {
            for (uint64_t j = 0; j < fs; ++j) {
                for (uint64_t d = 0; d < fd; ++d) {


                    //sdw[k](d, i, j) += sqr(dw[k](d, i, j));

                    w[k](d, i, j) = w[k](d, i, j) - (nu * dw[k](d, i, j)); //  / sqrt(eps + sdw[k](d, i, j));

                    dw[k](d, i, j) = 0;
                }
            }
        }

        //sdb[k] += sqr(db[k]);
        b[k] = b[k] - (nu * db[k]); // / sqrt(eps + sdb[k]);

        db[k] = 0;

    }
}