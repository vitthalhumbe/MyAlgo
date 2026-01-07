import numpy as np
class LinearRegression:
    def __init__(self, method="normal_eq", lr=0.01, epochs=1000, regul=None, lambda_=0.0, alpha=0.5):
        self.method = method
        self.epochs = epochs
        self.lr = lr
        self.regul = regul
        self.lambda_ = lambda_
        self.alpha = alpha
        
        # these are kept in hidden state (for updating value)
        self._weights = None

        # user will see this
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []
        self._validate_params()

    def fit(self, X, y):
        y = y.ravel()
        X = self._add_intercept(X)
        self._initialize_weights(X)

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

    def predict(self, X):
        X = self._add_intercept(X)
        return X.dot(self._weights)

    def score(self, X, y):
        y = y.ravel()
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot

    def _validate_params(self):
        if self.regul not in [None, 'l1', 'l2', 'elastic_net']:
            raise ValueError("Invalid regularization type!")
        if self.lambda_ < 0:
            raise ValueError("lambda_ must be >= 0")
        if self.regul == 'elastic_net' and not (0 <= self.alpha <= 1):
            raise ValueError("Aplha must be in between 0 and 1.")
        if self.method == 'normal_eq' and self.regul in ['l1', 'elastic_net']:
            raise ValueError("Normal equation supports only L2 regularization.")
        
    def _initialize_weights(self, X):
        n_features = X.shape[1]                   # shape = (rows, cols) so shape[1] will return number of features.

        self._weights = np.zeros(n_features)

    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _fit_gd(self, X, y):
        m = X.shape[0]
        
        for _ in range(self.epochs):
            gradient = self._compute_gradient(X, y)
            self._weights  -= self.lr * gradient
            self.loss_history.append(self._compute_loss(X, y))
 
    def _fit_sgd(self, X, y):
        m = X.shape[0]
        

        for _ in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            per_epoch_error = []
            for i in range(m):
                xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]              # extracting single row, but as a 2D array, for Gradient calculatoin.

                gradient = self._compute_gradient(xi, yi)           # it's only one record from the data
                self._weights -= self.lr * gradient
                error = xi.dot(self._weights) - yi
                per_epoch_error.append(error.item() ** 2)
            
            self.loss_history.append(np.mean(per_epoch_error))

    def _fit_normal_eq(self, X, y):
        self._weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def _compute_loss(self, X, y):
        m = X.shape[0]
        y_pred = X.dot(self._weights)
        mse = np.mean((y_pred - y) ** 2)            # base MSE cost, no penalty term added.


        if self.regul is None:
            return mse
        reg = 0.0
        w = self._weights[1:]                       # regul is applied only on features, not bias

        if self.regul == 'l2':
            reg = self.lambda_ * np.sum(w ** 2)
        elif self.regul == 'l2':
            reg = self.lambda_ * np.sum(np.abs(w))
        elif self.regul == "elastic_net":
            reg = self.lambda_ *(self.alpha * np.sum(w **2) + (1-self.alpha) * np.sum(np.abs(w)))      # just combine both l1 and l2

        return mse + reg         # here, actually penalty is added. 

    def _compute_gradient(self, X, y):
        m = X.shape[0]
        y_pred = X.dot(self._weights)
        error = y_pred - y

        grad = (2 / m) * X.T.dot(error)

        if self.regul is None:
            return grad
        
        w = self._weights.copy()
        w[0] = 0.0

        if self.regul == 'l2':
            grad += 2 * self.lambda_ * w

        elif self.regul == 'l1':
            grad += self.lambda_ * np.sign(w)
        
        elif self.regul == 'elastic_net':
            grad += self.lambda_ * (self.alpha * np.sign(w) + 2* (1 -self.alpha) * w)

        return grad

def main():
    np.random.seed(42)

    # sample test dataset creating, REF : Hands on Machine Learning book by Aurelien Geron ;
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    methods = ['normal_eq', 'gd', 'sgd']
    reguls = [None, 'l1', 'l2', 'elastic_net']

    print("\nLINEAR REGRESSION TEST RESULTS\n")
    print(f"{'Method':<10} {'Regul':<12} {'Coef':<12} {'Intercept':<12} {'R2':<8}")
    print("-" * 50)

    for method in methods:
        for regul in reguls:

            if method == 'normal_eq' and regul in ['l1', 'elastic_net']:
                continue

            try:
                model = LinearRegression(method=method,  regul=regul,lr=0.01, epochs=1000,lambda_=0.1, alpha=0.5)

                model.fit(X, y)
                r2 = model.score(X, y)
                coef = model.coef_[0]
                intercept = model.intercept_

                print(
                    f"{method:<10} {str(regul):<12} "
                    f"{coef:<12.4f} {intercept:<12.4f} {r2:<8.4f}"
                )

            except Exception as e:
                print(
                    f"{method:<10} {str(regul):<12} ERROR: {e}"
                )

if __name__ == '__main__':
    main()