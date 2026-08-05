#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using namespace std;


Eigen::MatrixXd relu(const Eigen::MatrixXd& Z) {
    return Z.cwiseMax(0.0);
}

Eigen::MatrixXd relu_deriv(const Eigen::MatrixXd& Z) {
    return (Z.array() > 0.0).cast<double>();
}

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& Z) {
    return 1.0 / (1.0 + (-Z.array()).exp());
}

Eigen::MatrixXd sigmoid_deriv(const Eigen::MatrixXd& Z) {
    Eigen::MatrixXd s = sigmoid(Z);
    return s.array() * (1.0 - s.array());
}

Eigen::MatrixXd tanh_deriv(const Eigen::MatrixXd& Z) {
    return 1.0 - Z.array().tanh().square();
}

Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z) {
    Eigen::MatrixXd shifted = Z;
    for (int i = 0; i < Z.rows(); ++i) {
        shifted.row(i) = Z.row(i).array() - Z.row(i).maxCoeff();
    }

    Eigen::MatrixXd expZ = shifted.array().exp();
    Eigen::VectorXd row_sums = expZ.rowwise().sum();

    for (int i = 0; i < expZ.rows(); ++i) {
        expZ.row(i) /= row_sums(i);
    }

    return expZ;
}


class Dense {
public:
    Dense(int in_dim, int out_dim, string activation): activation(activation)
    {
        double scale = (activation == "relu")
            ? sqrt(2.0 / in_dim)
            : sqrt(1.0 / in_dim);

        W = Eigen::MatrixXd::Random(in_dim, out_dim) * scale;
        b = Eigen::RowVectorXd::Zero(out_dim);
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X) {
        X_cache = X;
        Z = (X * W).rowwise() + b;

        if (activation == "relu") return relu(Z);
        if (activation == "sigmoid") return sigmoid(Z);
        if (activation == "tanh") return Z.array().tanh();
        if (activation == "softmax") return softmax(Z);
        return Z;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad, double lr, bool softmax_ce=false) {
        Eigen::MatrixXd dZ;

        if (softmax_ce) {
            dZ = grad;
        } else if (activation == "relu") {
            dZ = grad.array() * relu_deriv(Z).array();
        } else if (activation == "sigmoid") {
            dZ = grad.array() * sigmoid_deriv(Z).array();
        } else if (activation == "tanh") {
            dZ = grad.array() * tanh_deriv(Z).array();
        } else {
            dZ = grad;
        }

        Eigen::MatrixXd dW = X_cache.transpose() * dZ;
        Eigen::RowVectorXd db = dZ.colwise().sum();
        Eigen::MatrixXd dX = dZ * W.transpose();

        W -= lr * dW;
        b -= lr * db;

        return dX;
    }

    string activation;

private:
    Eigen::MatrixXd W;
    Eigen::RowVectorXd b;

    Eigen::MatrixXd X_cache;
    Eigen::MatrixXd Z;
};

class NeuralNetwork {
public:
    NeuralNetwork() {}

    void add_dense(int in_dim, int out_dim, string activation) {
        layers.emplace_back(in_dim, out_dim, activation);
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X) {
        Eigen::MatrixXd out = X;
        for (auto& layer : layers)
            out = layer.forward(out);
        return out;
    }

    Eigen::MatrixXd loss_grad(
        const Eigen::MatrixXd& y,
        const Eigen::MatrixXd& y_hat,
        string loss
    ) {
        if (loss == "mse") {
            return (2.0 / y.rows()) * (y_hat - y);
        }
        if (loss == "categorical_crossentropy") {
            return (y_hat - y) / y.rows();
        }
        if (loss == "binary_crossentropy") {
            return (y_hat - y) / y.rows();
        }
        throw runtime_error("Unsupported loss");
    }

    double compute_loss(
        const Eigen::MatrixXd& y,
        const Eigen::MatrixXd& y_hat,
        string loss
    ) {
        const double eps = 1e-9;

        if (loss == "mse") {
            return (y - y_hat).array().square().mean();
        }
        if (loss == "categorical_crossentropy") {
            return -(y.array() * (y_hat.array() + eps).log()).rowwise().sum().mean();
        }
        if (loss == "binary_crossentropy") {
            return -(y.array() * (y_hat.array() + eps).log()
                   + (1 - y.array()) * (1 - y_hat.array() + eps).log()).mean();
        }
        throw runtime_error("Unsupported loss");
    }

    void backward(
        const Eigen::MatrixXd& loss_grad,
        double lr,
        string loss
    ) {
        Eigen::MatrixXd grad = loss_grad;

        for (int i = layers.size() - 1; i >= 0; --i) {
            bool softmax_ce =
                (i == layers.size() - 1 &&
                 layers[i].activation == "softmax" &&
                 loss == "categorical_crossentropy");

            grad = layers[i].backward(grad, lr, softmax_ce);
        }
    }

private:
    vector<Dense> layers;
};

PYBIND11_MODULE(_ann_cpp, m) {
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<>())
        .def("add_dense", &NeuralNetwork::add_dense)
        .def("forward", &NeuralNetwork::forward)
        .def("loss_grad", &NeuralNetwork::loss_grad)
        .def("compute_loss", &NeuralNetwork::compute_loss)
        .def("backward", &NeuralNetwork::backward);
}
