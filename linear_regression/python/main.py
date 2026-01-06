import numpy as np
class LinearRegression:
    def __init__(self, method="normal_eq", lr=0.01, epochs=1000):
        self.method = method
        self.epochs = epochs
        self.lr = lr
        
        # these are kept in hidden state (for updating value)
        self._weights = None

        # user will see this
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []


    def fit(self, X, y):
        y = y.ravel()
        X = self._add_intercept(X)
        self._initialize_params(X)

        if self.method == "normal_eq":
            self._fit_normal_eq(X,y)
        elif self.method == 'gd':
            self._fit_gd(X, y)
        elif self.method == 'sgd':
            self._fit_sgd(X, y)
        else:
            raise ValueError("unkwon method")
        
        self.coef_ = self._weights[1:]              # all are coeficeints of features
        self.intercept_ = self._weights[0]          # except the first one - that is bias's weight

        print("fitting successfull !!")

    def predict(self, X):
        X = self._add_intercept(X)
        return X.dot(self._weights)
    
    def score():
        pass

    def _validate_params():
        pass
    def _initialize_params(self, X):
        n_features = X.shape[1]                   # shape = (rows, cols) so shape[1] will return number of features.

        self._weights = np.zeros(n_features)




    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _fit_gd(self, X, y):
        m = X.shape[0]
        
        for _ in range(self.epochs):
            y_pred = X.dot(self._weights)
            error = y_pred - y

            gradient = self._compute_gradient(X, m , error)
            self._weights  -= self.lr * gradient
            self._compute_loss(error)

        
    def _fit_sgd(self, X, y):
        m = X.shape[0]
        

        for _ in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            per_epoch_error = []
            for i in range(m):
                xi = X_shuffled[i]
                yi = y_shuffled[i]

                y_pred = xi.dot(self._weights)
                error = y_pred - yi

                gradient = 2 * xi * error           # it's only one record from the data
                self._weights -= self.lr * gradient
                per_epoch_error.append(error)
            error = np.mean(per_epoch_error)
            per_epoch_error = np.array([])
            self._compute_loss(error)


    def _fit_normal_eq(self, X, y):
        self._weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def _compute_loss(self, error):
        loss = np.mean(error ** 2)
        self.loss_history.append(loss)

    def _compute_gradient(self, X, n_samples, error):
        return (2 / n_samples) * X.T.dot(error)

def main():
    print("Testing normal equation started\n\n")

    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    model = LinearRegression(method='sgd')
    model.fit(X, y)

    print(model.coef_)
    print(model.intercept_)

    print(model.predict(np.array([2])))
    # print(model.loss_history)

if __name__ == '__main__':
    main()