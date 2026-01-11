import numpy as np
from tqdm.auto import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NeuralNetwrok:
    def __init__(self, *layers):
        self.layers = list(layers)
        self.loss = None
        self.lr = None
        self.history = {"loss": [], "accuracy": []}


    def train(self, X, y,epochs, batch_size=None):
        n = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0

            if batch_size is None:
                batches = [(X_shuffled, y_shuffled)]
            else:
                # creating batches using list comprehesion :
                batches = [
                    (X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]) for i in range(0, n, batch_size)
                ]

            print(f"Epoch {epoch+1}/{epochs}")
            with tqdm(total=len(batches), ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}') as pbar:
                pbar.set_postfix({"loss": "..."}) # Init postfix
            # training loop : calling forward -> compute loss -> calling backward (update wiehgts)
                for i, (X_batch, y_batch) in enumerate(batches):
                    y_hat = self.forward(X_batch)

                    loss = self._compute_loss(y_batch, y_hat)
                    epoch_loss += loss

                    loss_gradient = self._loss_derivative(y_batch, y_hat)
                    self.backward(loss_gradient)

                    running_loss = epoch_loss / (i + 1)
                    pbar.set_postfix({"- loss": f"{running_loss:.4f}"})
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

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        return out
    
    def backward(self, loss_gradient):
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def _compute_loss(self, y, y_hat):
        if self.loss == "mse":
            return self._mse(y, y_hat)
        elif self.loss == "categorical_crossentropy":
            return self._cce(y, y_hat)
        elif self.loss == "binary_crossentropy":
            return self._bce(y, y_hat)
        else:
            raise ValueError("Unsupported loss")


    def _loss_derivative(self, y, y_hat):
        if self.loss == "mse":
            return self._mse_grad(y, y_hat)
        elif self.loss == "categorical_crossentropy":
            return self._cce_grad(y, y_hat)
        elif self.loss == "binary_crossentropy":
            return self._bce_grad(y, y_hat)
        else:
            raise ValueError("Unsupported loss")

         
    def compile(self, loss, lr):
        self.loss = loss
        self.lr = lr
        
        input_dim = None

        for layer in self.layers:
            if isinstance(layer, Flatten):
                input_dim = layer.output_dim
            elif isinstance(layer, Dense):
                layer._initialize_parameters(input_dim)
                layer.lr = lr
                input_dim = layer.n_neurons

    def predict(self, X):
        return self.forward(X)
    
    def plot_loss_accuracy(self,loss, accuracy):
        epochs = np.arange(1, len(loss) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # ---- Loss (left y-axis)
        ax1.plot(epochs, loss, 'orange', label='Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color='orange')
        ax1.tick_params(axis='y', labelcolor='r')

        # ---- Accuracy (right y-axis)
        ax2 = ax1.twinx()
        ax2.plot(epochs, accuracy, 'b-', label='Accuracy')
        ax2.set_ylabel("Accuracy", color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        # ---- Title & grid
        plt.title("Training Loss & Accuracy")

        plt.show()

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        loss = self._compute_loss(y, y_pred)

        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y, axis=1)

        accuracy = np.mean(y_pred_labels == y_true_labels)
        return loss, accuracy


        return loss, accuracy

    def _mse(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def _mse_grad(self, y, y_hat):
        return 2 * (y_hat - y) / y.shape[0]
    
    def _cce(self, y, y_hat):
        eps = 1e-9
        return -np.mean(np.sum(y * np.log(y_hat + eps), axis=1))

    def _cce_grad(self, y, y_hat):
        return (y_hat - y) / y.shape[0]

    def _bce(self, y, y_hat):
        eps = 1e-9
        return -np.mean(
            y * np.log(y_hat + eps) +
            (1 - y) * np.log(1 - y_hat + eps)
        )

    def _bce_grad(self, y, y_hat):
        return (y_hat - y) / y.shape[0]


    
class Flatten:
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
        if self.activation == "softmax":
            deriv_Z = gradient
        elif self.activation is None:
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
            sigmoid = 1 / (1 + np.exp(-Z))
            return sigmoid * (1 - sigmoid)
        
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z) ** 2
        
        elif (self.activation == 'softmax'):
            raise NotImplementedError( "softmax handled with cross entropy")
        
        else:
            raise ValueError("unsupported Activation")
        
    def _initialize_parameters(self, input_dim):
        self.input_dim = input_dim

        if self.activation == 'relu':
            scale = np.sqrt( 2 / input_dim)         # He
        else:
            scale = np.sqrt(1 / input_dim)          # Xavier

        self.weights = np.random.randn(input_dim, self.n_neurons) * scale
        self.bias = np.zeros(self.n_neurons)


def main():
    X, y = fetch_openml("mnist_784",version=1,return_X_y=True,as_frame=False)

    X = X.astype(np.float32)
    y = y.astype(int)
    X = X / 255.0

    num_classes = 10
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    model = NeuralNetwrok(
        Flatten(input_shape=(784,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation='softmax')  
    )

    model.compile(loss="categorical_crossentropy", lr=0.1)

    history = model.train(
        X_train,
        y_train,
        epochs=11,
        batch_size=64
    )

    y_pred = model.predict(X_test)

    loss, accuracy = model.evaluate(X_test, y_test)
    print("\nFinal Test Accuracy:", accuracy, "\nFinal Test loss:", loss)

if __name__ == '__main__':
    np.random.seed(42)

    main()

    # X = np.random.randn(5, 3)          # 5 rows and 3 columns dataset
    # deriv_A = np.random.randn(5, 3)

    # dense = Dense(3, activation='relu')
    # dense.weights = np.random.randn(3, 3)
    # dense.bias = np.zeros(3)
    # dense.lr = 0.01

    # out = dense.forward(X)
    # dX = dense.backward(deriv_A)
    # print(dX.shape)        # should be 5 rows and 4 cols
    # print(dX)

    # model = NeuralNetwrok(
    #     Flatten(input_shape=(3,)),
    #     Dense(5, activation='relu'),
    #     Dense(1)
    # )

    # model.compile(loss='mse', lr=0.01)

    # X = np.random.randn(10, 3)
    # y_hat = model.predict(X)

    # print(y_hat)
    # print(y_hat.shape)

    # X = np.random.randn(100000, 1)
    # y = (X ** 2).reshape(-1, 1)

    # X = (X - X.mean()) / X.std()
    # y = (y - y.mean()) / y.std()

    # y_mean = y.mean()
    # y_std = y.std()

    # model = NeuralNetwrok(
    #     Flatten(input_shape=(1,)),
    #     Dense(16, activation='relu'),
    #     Dense(16, activation='tanh'),
    #     Dense(1)
    # )

    # model.compile(loss='mse', lr= 0.001)
    # history = model.train(X, y, epochs=11, batch_size=32)

    # print(history['loss'][-1])
    # y_pred = model.predict(X[:5])
    # y_pred_real = y_pred * y_std + y_mean

    # print(y_pred_real)
    # print(y[:5])

