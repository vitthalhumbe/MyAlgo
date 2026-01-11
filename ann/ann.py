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
        

    def backward(self, gradient):
        if self.activation is None:
            deriv_Z = gradient
        else:
            deriv_Z = gradient * self._activation_derivative(self.Z)

        deriv_weights = self.X.T.dot(deriv_Z)
        deriv_bias = np.sum(deriv_Z, axis=0)

        deriv_X = deriv_Z.dot(self.weights.T)

        self.weights -= self.lr * deriv_weights
        self.bias -= self.lr * deriv_bias

        return deriv_X

    def _activation_derivative(self, Z):
        if self.activation == 'relu':
            return (Z > 0).astype(float)        # relu'(x) = (0 if x < 0 and 1 if x >= 0)
        
        elif self.activation == 'sigmoid':
            sigmoid = 1 / (1 - np.exp(-Z))
            return sigmoid * (1 - sigmoid)
        
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z) ** 2
        
        elif (self.activation == 'softmax'):
            raise NotImplementedError( "softmax handled with cross entropy")
        
        else:
            raise ValueError("unsupported Activation")
if __name__ == '__main__':
    np.random.seed(42)

    X = np.random.randn(5, 3)          # 5 rows and 3 columns dataset
    deriv_A = np.random.randn(5, 3)

    dense = Dense(3, activation='relu')
    dense.weights = np.random.randn(3, 3)
    dense.bias = np.zeros(3)
    dense.lr = 0.01

    out = dense.forward(X)
    dX = dense.backward(deriv_A)
    print(dX.shape)        # should be 5 rows and 4 cols
    print(dX)

