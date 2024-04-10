// инициализация сети c нормальным распределением
// RMSPROP на обоих слоях
// Конфигурация по методам обучения
//RMSPROP - для полносвязного
// RMSPROP - для сверточного
// NU - learning_rate
#include <random>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/photo.hpp>
#include <string>
#include <set>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>


#include "Tensor.h"
#include "init.h"
#include "layer.h"
#include "ConvLayer.h"
#include "ReluLayer.h"
#include "maxpooling.h"
#include "fullconectLayer.h"
#include "tripletlosslayer.h"
using namespace std;
using namespace cv;
using namespace filesystem;
namespace fs = std::filesystem;
mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count()); // это генератор рандома
class network {// класс нейронной сети, которая собирается по слоям
private:
    vector<Tensor>outlearn;
public:
    ConvLayer* c1;
    Relulayer* r1;
    maxpoolinglayer* m1;

    ConvLayer* c2;
    Relulayer* r2;
    maxpoolinglayer* m2;

    ConvLayer* c3;
    Relulayer* r3;


    ConvLayer* c4;
    Relulayer* r4;

    ConvLayer* c5;
    Relulayer* r5;
    maxpoolinglayer* m5;

    fullyconectedlayer* fc1;
    fullyconectedlayer* fc2;
    fullyconectedlayer* fc3;




    vector<layer*>net;
    TensorSize outputsize;

    /*vector<vector<Tensor>>outdw;
    vector<vector<Tensor>>outdb;*/

    // задали архитектуру, теперь обучение
    network() { // сборка конструктора (попроубем сделать Alexnet)
        //переделаем архитектуру под размер изображения 32 * 32 * 3
        TensorSize c1sz;
        c1sz.depth = 3;
        c1sz.height = 227;
        c1sz.width = 227;
        c1 = new ConvLayer(c1sz, 96, 11, 0, 4);
        net.push_back(c1);

        TensorSize m1sz;
        m1sz.depth = 96;
        m1sz.width = 55;
        m1sz.height = 55;



        TensorSize r1sz;
        r1sz = m1sz;
        r1 = new Relulayer(r1sz);
        net.push_back(r1);


        m1 = new maxpoolinglayer(m1sz, 2);
        net.push_back(m1);


        TensorSize c2sz;
        c2sz.depth = 96;
        c2sz.height = 27;
        c2sz.width = 27;
        c2 = new ConvLayer(c2sz, 256, 5, 2, 1);
        net.push_back(c2);

        TensorSize m2sz;
        m2sz.depth = 256;
        m2sz.width = 27;
        m2sz.height = 27;


        TensorSize r2sz;
        r2sz = m2sz;
        r2 = new Relulayer(r2sz);
        net.push_back(r2);


        m2 = new maxpoolinglayer(m2sz, 2);
        net.push_back(m2);

        TensorSize c3sz;
        c3sz.depth = 256;
        c3sz.height = 13;
        c3sz.width = 13;
        c3 = new ConvLayer(c3sz, 384, 3, 1, 1);
        net.push_back(c3);


        TensorSize r3sz;
        r3sz.depth = 384;
        r3sz.width = 13;
        r3sz.height = 13;
        r3 = new Relulayer(r3sz);
        net.push_back(r3);


        TensorSize c4sz;
        c4sz.depth = 384;
        c4sz.height = 13;
        c4sz.width = 13;
        c4 = new ConvLayer(c4sz, 384, 3, 1, 1);
        net.push_back(c4);


        TensorSize r4sz;
        r4sz.depth = 384;
        r4sz.width = 13;
        r4sz.height = 13;
        r4 = new Relulayer(r4sz);
        net.push_back(r4);

        TensorSize c5sz;
        c5sz.depth = 384;
        c5sz.height = 13;
        c5sz.width = 13;
        c5 = new ConvLayer(c5sz, 256, 3, 1, 1);
        net.push_back(c5);



        TensorSize r5sz;
        r5sz.depth = 256;
        r5sz.width = 13;
        r5sz.height = 13;
        r5 = new Relulayer(r5sz);
        net.push_back(r5);

        TensorSize m5sz;
        m5sz.depth = 256;
        m5sz.width = 13;
        m5sz.height = 13;
        m5 = new maxpoolinglayer(m5sz, 2);
        net.push_back(m5);

        TensorSize fc1sz;
        fc1sz.depth = 256;
        fc1sz.height = 6;
        fc1sz.width = 6;
        fc1 = new fullyconectedlayer(fc1sz, 4096);
        net.push_back(fc1);

        TensorSize fc2sz;
        fc2sz.depth = 4096;
        fc2sz.height = 1;
        fc2sz.width = 1;
        fc2 = new fullyconectedlayer(fc2sz, 4096);
        net.push_back(fc2);

        TensorSize fc3sz;
        fc3sz.depth = 4096;
        fc3sz.height = 1;
        fc3sz.width = 1;
        fc3 = new fullyconectedlayer(fc3sz, 1000);


        outputsize.depth = 1000;
        outputsize.height = 1;
        outputsize.width = 1;



    }
    Tensor Forward(const Tensor& X) {
         Tensor output = X;
        for (int i = 0; i < net.size(); ++i) {
            outlearn.push_back(output);
            output = net[i]->Forward(output);
            //cout << output;
            //cout << output << "\n";
        }
        //exit(0);
        outlearn.push_back(output);
        return output;
    }
    void Backward(const Tensor& res, const Tensor& expect) {
        Tensor dx = net.back()->Backward(res, expect);
        for (int i = net.size() - 2; i > -1; --i) {
            dx = net[i]->Backward(dx, outlearn[i]);
            //net[i]->updateweights(0.01);
        }
        outlearn.clear();
    }
    void updateweights(double alpha) {
        for (int i = (int)net.size() - 1; i > -1; --i) {
            net[i]->updateweights(alpha);
        }
        return;
    }
    void Backward(Tensor& dx) {
        for (int i = net.size() - 1; i > -1; --i) {
            dx = net[i]->Backward(dx, outlearn[i]);
            //net[i]->updateweights(0.01);
        }
        outlearn.clear();
    }
    void full_clear() {
        for (int i = 0; i < net.size(); ++i) {
            net[i]->full_clear();
        }
    }
    Tensor Test(const Tensor& X) {
        Tensor output = X;
        for (int i = 0; i < net.size(); ++i) {
            outlearn.push_back(output);
            output = net[i]->Test(output);
            //cout << output;
            //cout << output << "\n";
        }
        //exit(0);
        return output;
    }
    int get_ans(const Tensor& res) {
        double mx = -1;
        double it = 0;
        for (int d = 0; d < res.size.depth; ++d) {
            if (mx < res(d, 0, 0)) {
                it = d;
                mx = res(d, 0, 0);
            }
        }
        return it;
    }
    void hard_sampling(vector<Tensor>& data, vector<int>& mark) {
        vector<Tensor>req(data.size());
        TripletLosslayer L(outputsize, rad); // слой ошибки
        for (int i = 0; i < data.size(); ++i) {
            req[i] = Forward(data[i]);
            outlearn.clear();
        }
        full_clear();
        for (int a = 0; a < data.size(); ++a) {
            vector<pair<int, int>>all;
            for (int p = 0; p < data.size(); ++p) {
                for (int n = 0; n < data.size(); ++n) {
                    double dist1 = dist(req[a], req[p]);
                    double dist2 = dist(req[a], req[n]);
                    if (dist1 < dist2 && dist2 < dist1 + rad) {
                        all.push_back({ p,n });
                    }
                }
            }
            if (all.size() == 0) continue;
            int k = rnd() % all.size();
            int p = all[k].first;
            int n = all[k].second;
            vector<Tensor>res = L.dxTripletloss(req[a], req[p], req[n]);
            Forward(data[a]);
            Backward(res[0]);
            Forward(data[p]);
            Backward(res[1]);
            Forward(data[n]);
            Backward(res[2]);
        }
        updateweights(speed);
        full_clear();
    }//ищем полужесткие примеры

    void check_res(vector<Tensor>& data, vector<int>& mark) {
        vector<Tensor>req(data.size());
        TripletLosslayer L(outputsize, rad); // слой ошибки
        double output = 0; //  расстояние не может быть отрицательным
        for (int i = 0; i < data.size(); ++i) {
            req[i] = Forward(data[i]);
            outlearn.clear();
        }
        double tp = 0;
        double fp = 0;
        double tn = 0;
        double fn = 0;
        double error = 0;
        //double cnter = 0;
        for (int a = 0; a < (int)data.size(); ++a) {
            for (int p = 0; p < (int)data.size(); ++p) {
                for (int n = 0; n < (int)data.size(); ++n) {
                    if (a == p || a == n || n == p) continue;
                    if (mark[a] == mark[p] && mark[a] != mark[n]) {
                        error += L.Tripletloss(req[a], req[p], req[n]);
                        //cnter++;
                    }
                }
            }
        }
        //error /= cnter;


        for (int a = 0; a < (int)data.size(); ++a) {
            for (int p = 0; p < (int)data.size(); ++p) {
                if (a == p) continue;
                if (L.check_in(req[a], req[p])) {
                    if (mark[a] == mark[p]) {
                        tp++;
                    }
                    else {
                        fp++;
                    }
                }
                else {
                    if (mark[a] != mark[p]) {
                        tn++;
                    }
                    else {
                        fn++;
                    }
                }
            }
        }
        cout << "------  --------\n";
        cout << "Loss: " << fixed << setprecision(5) << error << "\n";
        cout << fixed << setprecision(0) << tp << " " << fn << "\n";
        cout << fixed << setprecision(0) << fp << " " << tn << "\n";
        cout << "error rate " << fixed << setprecision(5) << (fp + fn) / (tp + tn + fp + fn) << "\n";
        cout << "Accuracy " << fixed << setprecision(5) << (tp + tn) / (tp + tn + fp + fn) << "\n";
        cout << "precision " << fixed << setprecision(5) << (tp) / max(tp + fp, (double)1.0) << "\n";// доля правильных среди всех + классов
        cout << "recall " << fixed << setprecision(5) << (tp) / max(tp + fn, (double)1.0) << "\n";  // доля правильных, среди всех правильных пар
        cout << "------  --------\n";
        return;
    }
    void metric_train(const vector<Tensor>& train_data, vector<int>& train_mark,
        vector<Tensor>& test_data, vector<int>& test_mark, double border) {
        double speed0 = speed;
        int N = 5;
        double alph = 0.96;
        for (int ep = 1; ep < 256; ++ep) {
            check_res(test_data, test_mark);
            speed = speed0 * pow(alph, (ep / N));
            cerr << fixed << setprecision(10) << "speed_learning: " << speed << "\n";
            sampler(train_data, train_mark, 32); /* mini-batch gradient descent
            т.к в YALEB - 65 в одной пачке, эпохи потом подберешь и размер пакета тоже
            */
        }
    }
    void sampler(const vector<Tensor>& alldata, vector<int>& mark, int n_batch, int epoch = 1) { // alldata - сами изображения,

        // mark - к какому классу относится
        int cntop = 0;
        for (int ep = 0; ep < epoch; ++ep) {
            set<int>used;
            for (int i = 0; i < alldata.size(); i++) {
                used.insert(i);
            }
            while (!used.empty()) {
                vector<int>nums;
                if (used.size() <= n_batch) {
                    for (auto& c : used) {
                        nums.push_back(c);
                    }
                    used.clear();
                }
                else {
                    for (int i = 0; i < n_batch; ++i) {
                        int k = rnd() % alldata.size();
                        while (!used.count(k)) {
                            k = rnd() % alldata.size();
                        }
                        nums.push_back(k);
                        used.erase(k);
                    }
                }
                vector<Tensor>bdata;
                vector<int>bmark;
                for (auto& c : nums) {
                    bdata.push_back(alldata[c]);
                    bmark.push_back(mark[c]);
                }
                hard_sampling(bdata, bmark);
                ++cntop;
                cerr << "package number : " << cntop << " processed\n";
            }
        }
    }
};
int32_t main()
{
    network net;

    vector<Tensor>train_data;
    vector<int>train_mark;


    vector<Tensor>test_data;
    vector<int>test_mark;

    random_device rd;
    mt19937 g(rd());


    string tempStr;
    string inputPath = "./faces_data/";
    cout <<"there\n";
    path path;
    for (auto& p : directory_iterator(inputPath))
    {
        path = p;
        tempStr = path.path::generic_string();
        int c = tempStr.back() - '0';
        if (c == 2) break;
        int i = 0;
        for (auto& pp : directory_iterator(tempStr)) {
            path = pp;
            string way = path.path::generic_string();
            Mat image;
            image = imread(way, IMREAD_COLOR);
            cv::Size sz = { 227,227 };
            cv::resize(image, image, sz);

            Tensor nw(image.cols, image.rows, 3);
            for (uint64_t y = 0; y < image.rows; ++y) {
                for (uint64_t x = 0; x < image.cols; ++x) {
                    Vec3b pixel = image.at<Vec3b>(Point(x, y));
                    for (uint64_t d = 0; d < 3; d++) {
                        long double res = (long double)pixel.val[d];
                        res /= 255.0;
                        nw.set(d, y, x, res); //BGR
                    }
                }
            }
            if (i <= 7) { // в test
                test_mark.push_back(c);
                test_data.push_back(nw);

                train_mark.push_back(c);
                train_data.push_back(nw);
            }
            else {
                break;
            }
            i++;
        }
    }
    net.metric_train(train_data, train_mark, test_data, test_mark, 0.01);
}