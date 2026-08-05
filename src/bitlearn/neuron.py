import numpy as np

class Neuron:
    def __init__(self, n_inputs, lr = 0.01):
        self.w = np.random.randn(n_inputs)
        self.b = 0.0
        self.lr = lr


        #cache:
        self.x = None
        self.z = None
        self.a = None

    def train(self, X, y, epochs):
        for e in range(epochs):
            y_hat = self.forward(X)

            grad_loss = 2* (y_hat - y)
            self.backward(grad_loss)
            print("weights : ", self.w)
    def forward(self, x):

        self.x = x
        self.z = np.dot(self.w, x) + self.b              # z = wx + b
        self.a = self.z                                 # no activation for now

        return self.a           # output of forward pass
    
    def backward(self, grad_a):

        grad_z = grad_a
        
        grad_w = grad_z * self.x
        grad_b = grad_z

        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b


if __name__ == '__main__':
    
    x = np.array([2.0])
    y = 20.0

    neuron = Neuron(1, 0.1)
    # y_pred = neuron.forward(x)
    # loss = (y_pred - y) ** 2
    # print("predictin : ", y_pred)

    # grad_loss = 2* (y_pred - y)

    # neuron.backward(grad_loss)

    neuron.train(x, y, 30)
    print(neuron.a)

    # print("updated_weights : ", neuron.w)
    # print("updated bias :", neuron.b)