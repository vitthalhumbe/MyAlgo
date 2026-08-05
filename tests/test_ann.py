# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Vitthal Humbe

import pytest
import numpy as np
from bitlearn.ann import NeuralNetwork, Dense, Flatten


# CURRENT TEST STATUS : 24/24 PASSED

def make_xor_data():
    """Small XOR dataset for fast functional tests."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float64)
    return X, y

def make_random_data(n=200, in_dim=8, n_classes=3):
    """Random classification data with one-hot labels."""
    np.random.seed(42)
    X = np.random.randn(n, in_dim)
    labels = np.random.randint(0, n_classes, n)
    y = np.zeros((n, n_classes))
    y[np.arange(n), labels] = 1.0
    return X, y



def test_flatten_output_dim():
    f = Flatten((28, 28))
    assert f.output_dim == 784

def test_flatten_forward_shape():
    f = Flatten((4, 4))
    X = np.random.randn(10, 4, 4)
    out = f.forward(X)
    assert out.shape == (10, 16)

def test_flatten_backward_shape():
    f = Flatten((4, 4))
    X = np.random.randn(10, 4, 4)
    f.forward(X)
    grad = np.random.randn(10, 16)
    restored = f.backward(grad)
    assert restored.shape == (10, 4, 4)


def test_dense_stores_neurons():
    d = Dense(64, activation='relu')
    assert d.n_neurons == 64

def test_dense_stores_activation():
    d = Dense(32, activation='sigmoid')
    assert d.activation == 'sigmoid'

def test_dense_default_activation_is_none():
    d = Dense(16)
    assert d.activation is None



def test_compile_dense_before_flatten_raises():
    model = NeuralNetwork(
        Dense(64, activation='relu'),
        Dense(2, activation='softmax'),
    )
    with pytest.raises(ValueError):
        model.compile(loss='categorical_crossentropy', lr=0.01)

def test_compile_valid_architecture():
    model = NeuralNetwork(
        Flatten((8,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)

def test_compile_sets_loss_and_lr():
    model = NeuralNetwork(Flatten((4,)), Dense(2, activation='softmax'))
    model.compile(loss='mse', lr=0.05)
    assert model.loss == 'mse'
    assert model.lr == 0.05


def test_predict_output_shape():
    X, y = make_random_data(n=50, in_dim=8, n_classes=3)
    model = NeuralNetwork(
        Flatten((8,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    out = model.predict(X)
    assert out.shape == (50, 3)

def test_predict_softmax_sums_to_one():
    X, y = make_random_data(n=20, in_dim=4, n_classes=2)
    model = NeuralNetwork(
        Flatten((4,)),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    out = model.predict(X)
    row_sums = out.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(20), atol=1e-5)


def test_evaluate_returns_two_values():
    X, y = make_random_data(n=50, in_dim=8, n_classes=3)
    model = NeuralNetwork(
        Flatten((8,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    result = model.evaluate(X, y)
    assert len(result) == 2

def test_evaluate_accuracy_range():
    X, y = make_random_data(n=100, in_dim=8, n_classes=3)
    model = NeuralNetwork(
        Flatten((8,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    loss, accuracy = model.evaluate(X, y)
    assert 0.0 <= accuracy <= 1.0

def test_evaluate_loss_positive():
    X, y = make_random_data(n=50, in_dim=8, n_classes=3)
    model = NeuralNetwork(
        Flatten((8,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    loss, _ = model.evaluate(X, y)
    assert loss > 0.0


def test_train_returns_history_keys():
    X, y = make_random_data(n=100, in_dim=8, n_classes=3)
    model = NeuralNetwork(
        Flatten((8,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    history = model.train(X, y, epochs=3, batch_size=32)
    assert 'loss' in history
    assert 'accuracy' in history

def test_train_history_length():
    X, y = make_random_data(n=100, in_dim=8, n_classes=3)
    model = NeuralNetwork(
        Flatten((8,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    history = model.train(X, y, epochs=5, batch_size=32)
    assert len(history['loss']) == 5
    assert len(history['accuracy']) == 5

def test_train_loss_decreases():
    X, y = make_random_data(n=200, in_dim=8, n_classes=3)
    model = NeuralNetwork(
        Flatten((8,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    history = model.train(X, y, epochs=20, batch_size=32)
    mid = len(history['loss']) // 2
    assert np.mean(history['loss'][mid:]) < np.mean(history['loss'][:mid])

def test_train_fullbatch():
    """Full-batch mode (batch_size=None) should run without error."""
    X, y = make_xor_data()
    model = NeuralNetwork(
        Flatten((2,)),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    history = model.train(X, y, epochs=5)
    assert len(history['loss']) == 5



@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_hidden_activations(activation):
    X, y = make_random_data(n=50, in_dim=4, n_classes=2)
    model = NeuralNetwork(
        Flatten((4,)),
        Dense(8, activation=activation),
        Dense(2, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    out = model.predict(X)
    assert out.shape == (50, 2)



@pytest.mark.parametrize("loss", ["mse", "binary_crossentropy"])
def test_loss_functions(loss):
    np.random.seed(0)
    X = np.random.randn(50, 4)
    y = np.zeros((50, 2))
    y[np.arange(50), np.random.randint(0, 2, 50)] = 1.0

    model = NeuralNetwork(
        Flatten((4,)),
        Dense(8, activation='relu'),
        Dense(2, activation='sigmoid'),
    )
    model.compile(loss=loss, lr=0.01)
    history = model.train(X, y, epochs=3, batch_size=16)
    assert len(history['loss']) == 3



def test_mnist_smoke():
    """
    Loads a small MNIST subset (1000 samples) and checks that the model
    trains for 2 epochs without error and achieves above random accuracy.
    Random baseline for 10 classes is 0.10.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y_raw = mnist.data[:1000] / 255.0, mnist.target[:1000].astype(int)

    # one-hot encode
    y = np.zeros((1000, 10))
    y[np.arange(1000), y_raw] = 1.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = NeuralNetwork(
        Flatten((784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax'),
    )
    model.compile(loss='categorical_crossentropy', lr=0.01)
    model.train(X_train, y_train, epochs=2, batch_size=32)

    _, accuracy = model.evaluate(X_test, y_test)
    assert accuracy > 0.10  # above random baseline