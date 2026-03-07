# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Vitthal Humbe
#
# ann.py — Keras-style Python API over the C++  backend.
# layer definitions (Dense, Flatten) are pure Python; and  all forward/backward math runs inside ann.cpp via pybind11.

import numpy as np
from tqdm.auto import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import myalgo._ann_cpp as ann_cpp

class NeuralNetwork:
    """
    Feedforward neural network with a compiled C++ backend.

    Accepts a sequence of Flatten and Dense layers, mirrors the Keras compile/train/predict/evaluate API.

    @parameters
        *layers : Flatten or Dense. 
                  Layer definitions in order. Must start with Flatten, followed by
                  one or more Dense layers.

    @attributes
        history : dict, Contains 'loss' and 'accuracy' lists populated during training.

    @examples
        >>> model = NeuralNetwork(
                Flatten((28, 28)),
                Dense(128, activation='relu'),
                Dense(10, activation='softmax'),
            )
        >>> model.compile(loss='categorical_crossentropy', lr=0.01)
        >>> model.train(X_train, y_train, epochs=10, batch_size=32)
    
    """
    def __init__(self, *layers):
        self.layers = list(layers)
        self.loss = None
        self.cpp_model = ann_cpp.NeuralNetwork()
        self.lr = None
        self.history = {"loss": [], "accuracy": []}


    def train(self, X, y,epochs, batch_size=None):
        """
        Train the model on the given data.

        Shuffles data each epoch, supports full-batch and mini-batch modes,logs per-epoch loss and accuracy, and plots training curves after all epochs complete.

        @parameters
            X           :   ndarray of shape (n_samples, ...)
                            Input features. Will be flattened by the Flatten layer.
            
            y           :   ndarray of shape (n_samples, n_classes)
                            One-hot encoded target labels.
            
            epochs      :   int, Number of full passes over the training data.
            
            batch_size  :   int or None, default=None
                            Mini-batch size. If None, uses full-batch gradient descent.

        @returns
            history     : dict, Keys 'loss' and 'accuracy', each a list of per-epoch values.
        """

        n = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0

            if batch_size is None:
                batches = [(X_shuffled, y_shuffled)]
            else:
                batches = [
                    (X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]) for i in range(0, n, batch_size)
                ]

            print(f"Epoch {epoch+1}/{epochs}")
            with tqdm(total=len(batches), ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}') as pbar:
                pbar.set_postfix({"loss": "..."}) 
                for i, (X_batch, y_batch) in enumerate(batches):
                    y_hat = self.cpp_model.forward(X_batch)
                    loss = self.cpp_model.compute_loss(y_batch, y_hat, self.loss)
                    epoch_loss += loss
                    grad = self.cpp_model.loss_grad(y_batch, y_hat, self.loss)
                    self.cpp_model.backward(grad, self.lr, self.loss)


                    running_loss = epoch_loss / (i + 1)
                    pbar.set_postfix({"- loss": f"{running_loss}"})
                    pbar.update(1)
                
                epoch_loss = epoch_loss / len(batches)
                self.history["loss"].append(epoch_loss)

                y_pred = self.predict(X)
                y_pred_labels = np.argmax(y_pred, axis=1)
                y_true_labels = np.argmax(y, axis=1)

                accuracy = np.mean(y_pred_labels == y_true_labels)
                self.history["accuracy"].append(accuracy)
        self.plot_loss_accuracy(
            self.history["loss"],
            self.history["accuracy"]
        )
        return self.history

         
    def compile(self, loss, lr):
        """
        Configure the model for training.

        Registers all Dense layers with the C++ backend, inferring input dimensions from the preceding Flatten layer.

        @parameters
            loss :  {'mse', 'categorical_crossentropy', 'binary_crossentropy'}, Loss function to use during training.
            lr   :  float, Learning rate for gradient descent.

        @raises
            ValueError : If a Dense layer appears before a Flatten layer.
        """

        self.loss = loss
        self.lr = lr
        
        input_dim = None

        for layer in self.layers:
            if isinstance(layer, Flatten):
                input_dim = layer.output_dim
            elif isinstance(layer, Dense):
                if input_dim is None:
                    raise ValueError("Flatten must come before Dense")

                self.cpp_model.add_dense(
                    input_dim,
                    layer.n_neurons,
                    layer.activation or ""
                )
                input_dim = layer.n_neurons


    def predict(self, X):
        """
        Run a forward pass and return raw output activations.

        @parameters
            X : ndarray of shape (n_samples, ...)

        @returns
            y_pred : ndarray of shape (n_samples, n_outputs)
        
        """
        return self.cpp_model.forward(X)
    
    def plot_loss_accuracy(self,loss, accuracy):
        """
        Plot training loss and accuracy on a dual-axis chart.

        @parameters
            loss : list of float
            accuracy : list of float
        """
        
        epochs = np.arange(1, len(loss) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(epochs, loss, 'orange', label='Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color='orange')
        ax1.tick_params(axis='y', labelcolor='r')

        ax2 = ax1.twinx()
        ax2.plot(epochs, accuracy, 'b-', label='Accuracy')
        ax2.set_ylabel("Accuracy", color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        plt.title("Training Loss & Accuracy")

        plt.show()

    def evaluate(self, X, y):
        """
        Compute loss and accuracy on the given data.

        @parameters
            X : ndarray of shape (n_samples, ...)
            y : ndarray of shape (n_samples, n_classes)
                One-hot encoded labels.

        @returns
            loss : float
            accuracy : float
        
        """
        y_pred = self.cpp_model.forward(X)
        loss = self.cpp_model.compute_loss(y, y_pred, self.loss)

        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y, axis=1)

        accuracy = np.mean(y_pred_labels == y_true_labels)
        return loss, accuracy


        return loss, accuracy

    
class Flatten:
    """
    Reshapes input from (n_samples, *dims) to (n_samples, prod(dims)).

    @parameters
        input_shape :   tuple of int
                        Shape of a single sample, excluding the batch dimension.
                        Example: (28, 28) for MNIST images.

    @attributes
        output_dim : int
            Flattened feature size passed to the first Dense layer.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.original_shape = None
        self.output_dim = np.prod(input_shape)

    def forward(self, X):
        self.original_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, gradient):
        return gradient.reshape(self.original_shape)

class Dense:
    """
    Fully connected layer definition (registered with C++ backend on compile).

    @parameters
        n_neurons : int, Number of units in this layer.
        
        activation : {'relu', 'sigmoid', 'tanh', 'softmax'} or None
                     Activation function. None means linear (no activation).
    """
    def __init__(self, n_nuerons, activation=None):
        self.n_neurons = n_nuerons
        self.activation = activation
        self.input_dim = None
