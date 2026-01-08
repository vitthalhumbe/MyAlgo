#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using namespace std;

class LinearRegression
{
public:
    LinearRegression(string method = "gd", string regul = "none", double lr = 0.01, int epochs = 1000, double lambda_ = 0.0, double alpha = 0.5)
        : method(method), regul(regul), lr(lr), lambda_(lambda_), alpha(alpha), epochs(epochs) {};

    void fit(py::array_t<double> X, py::array_t<double> y) {
        auto Xbuffer = X.unchecked<2>();
        auto ybuffer = y.unchecked<1>();

        int m = Xbuffer.shape(0);       // rows
        int n = Xbuffer.shape(1) + 1;   // cols, +1 for bias

        // adding intercept

        Eigen::MatrixXd Xb(m,n);
        Eigen::VectorXd yv(m);
        for (int i = 0;i < m; ++i) {
            Xb(i, 0) = 1.0;
            for (int j = 1; j < n; ++j) {
                Xb(i, j) = Xbuffer(i, j-1);
            }
            yv(i) = ybuffer(i);
        }

        weights = Eigen::VectorXd::Zero(n);

        if (method == "normal_eq"){}
            // fit_normal_eq();
        else if (method == "gd") 
            fit_gd(Xb, yv);
        else if (method == "sgd") {}
            //fit_sgd();
        else
            throw runtime_error("Unknown method");
    }

    // todo : add this functions
    // py::array_t<double> predict();
    // vector<double> get_weights() const {};

private:
    string method, regul;
    double lr, lambda_, alpha;
    int epochs;
    Eigen::VectorXd weights;

    // double dot();
    void add_regularization(Eigen::VectorXd& grad) {
        for (int j = 1;j < grad.size(); ++j) {
            if (regul == "l2") {
                grad(j) += 2 * lambda_ * weights(j);
            } else if (regul == "l1") {
                grad(j) +=lambda_ * (weights(j) > 0 ? 1 : -1);
            } else if( regul == "elastic_net") {
                grad(j) += lambda_ * (alpha * (weights(j) > 0 ? 1: -1) + 2 * (1-alpha) * weights(j));
            }
        }
    }
    void fit_gd(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        int m = X.rows();

        for (int e = 0; e < epochs; e++) {
            Eigen::VectorXd error = X * weights -y;
            Eigen::VectorXd grad = (2.0 / m) * X.transpose() * error;

            add_regularization(grad);
            weights -= lr * grad;
            
        }
        cout << weights << endl;
    }
    //void fit_sgd();
    //void fit_normal_eq();
};

PYBIND11_MODULE(linreg_internal, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<string, string, double, int, double, double>())
        .def("fit", &LinearRegression::fit);
}