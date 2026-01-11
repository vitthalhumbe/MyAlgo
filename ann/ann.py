import numpy as np

class NeuralNetwrok:
    pass

class Flatten:
    pass

class Dense:
    def __init__(self, n_nuerons, activation=None):
        self.n_neurons = n_nuerons
        self.activation = activation

        self.weights = None
        self.bias = None


        # cache (required for backpropogation)
        self.X = None       #  input's cache
        self.Z = None       #  weighted sum's cache

    def forward(self, X):
        self.X = X

        self.Z = X.dot(self.weights) + self.bias

        if self.activation is None:
            return self.Z
        
        return self._choose_activation(self.Z)

    def _choose_activation(self, Z):
        if self.activation == "relu":
            return np.maximum(0, Z)
        
        elif (self.activation == 'sigmoid'):
            return 1 / (1 + np.exp(-Z))
        
        elif (self.activation == 'tanh'):
            return np.tanh(Z)
        
        elif (self.activation == 'softmax'):
            exp_z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        else:
            raise ValueError("Unsupported Activation")


if __name__ == '__main__':
    np.random.seed(42)

    X = np.random.randn(5, 3)          # 5 rows and 3 columns dataset

    dense = Dense(4, activation='relu')
    dense.weights = np.random.randn(3, 4)
    dense.bias = np.zeros(4)

    out = dense.forward(X)
    print(out.shape)        # should be 5 rows and 4 cols
    print(out)

