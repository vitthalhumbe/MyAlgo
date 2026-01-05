import numpy as np
class LinearRegression:
    def __init__(self, method="normal_eq"):
        self.method = method

        # these are kept in hidden state (for updating value)
        self._weights = None

        # user will see this
        self.coef_ = None
        self.intercept_ = None


    def fit(self, X, y):
        X = self._add_intercept(X)
        self._initialize_params(X)

        if self.method == "normal_eq":
            self._fit_normal_eq(X,y)
        else:
            raise ValueError("unkwon method")
        
        self.coef_ = self._weights[1:]              # all are coeficeints of features
        self.intercept_ = self._weights[0]          # except the first one - that is bias's weight

        print("fitting successfull !!")

    def predict():
        pass
    def score():
        pass

    def _validate_params():
        pass
    def _initialize_params(self, X):
        n_features = X.shape[1]                   # shape = (rows, cols) so shape[1] will return number of features.

        self._weights = np.zeros(n_features)




    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _fit_gd():
        pass
    def _fit_sgd():
        pass
    def _fit_normal_eq(self, X, y):
        self._weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def _compute_loss():
        pass
    def _compute_gradient():
        pass

def main():
    print("Testing normal equation started\n\n")

    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    model = LinearRegression(method='normal_eq')
    model.fit(X, y)

    print(model.coef_)
    print(model.intercept_)



if __name__ == '__main__':
    main()