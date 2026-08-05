# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Vitthal Humbe
#
# linear_regression.py — scikit-learn style Python wrapper over the C++ backend.
# The actual math (normal equation, GD, SGD, regularization) lives in linear_regression.cpp.


import numpy as np
from myalgo._linreg_cpp import LinearRegression as _CppImplimentation
class LinearRegression:

    """
    Supports normal equation, gradient descent, and stochastic gradient descent,
    with optional L1, L2, and Elastic Net regularization.

    @Parameters :
        method  :   {'normal_eq', 'gd', 'sgd'}, default='normal_eq'
                    Optimization method. 'normal_eq' solves statistically, 'gd' and 'sgd' use iterative gradient descent.
        
        lr      :   float, default=0.01
                    Learning rate. Used only when method is 'gd' or 'sgd'.
        
        epochs  :   int, default=1000
                    Number of training iterations. Used only when method is 'gd' or 'sgd'.
        
        regul   :   {'l1', 'l2', 'elastic_net'} or None, default=None
                    Regularization type. 'normal_eq' supports only 'l2'.
        
        lambda_ :   float, default=0.0
                    Regularization strength. Must be >= 0.
        
        alpha   :   float, default=0.5
                    Elastic Net mixing parameter between L1 and L2. Must be in [0, 1].

    @Attributes:
        coef_       :   ndarray of shape (n_features,)
                        Feature weights after fitting.
        
        intercept_  :   float
                        Bias term after fitting.

    @examples
        >>> import numpy as np
        >>> from myalgo import LinearRegression
        >>> X = np.random.rand(100, 1)
        >>> y = 3 * X + 2 + np.random.randn(100, 1)
        >>> model = LinearRegression(method='gd', lr=0.01, epochs=1000)
        >>> model.fit(X, y)
        >>> model.score(X, y)
    
    """

    def __init__(self, method="normal_eq", lr=0.01, epochs=1000, regul=None, lambda_=0.0, alpha=0.5):
        self.method = method
        self.epochs = epochs
        self.lr = lr
        self.regul = regul
        self.lambda_ = lambda_
        self.alpha = alpha
        
        self.coef_ = None
        self.intercept_ = None
        self._validate_params()

        self._internal_egine = _CppImplimentation()

    def fit(self, X, y):
        
        """
        fit the model to trraining data.

        @parameters
            X : ndarray of shape (n_samples, n_features) - Training feature matrix.
            y : ndarray of shape (n_samples,) or (n_samples, 1) - Target values.

        @returns 
            self : LinearRegression : Fitted estimator.
        
        """

        y = y.ravel()
        self._internal_egine.fit(X, y, self.method, self.regul if self.regul else "none", self.lr, self.epochs, self.lambda_, self.alpha)
        weights = self._internal_egine.get_weights()
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

        return self

    def predict(self, X):
        """
        predict target values for input X.

        @parameters
            X : ndarray of shape (n_samples, n_features)

        @returns
            y_pred : ndarray of shape (n_samples,)
        
        """
        return self._internal_egine.predict(X)

    def get_loss_history(self):
        """
        Return per-epoch loss values recorded during training.

        @returns
            loss_history : ndarray of shape (epochs,) : Only populated when method is 'gd' or 'sgd'.
        """

        return self._internal_egine.get_loss_history()

    def score(self, X, y):

        """
        Compute R² score on the given data.

        @parameters
            X : ndarray of shape (n_samples, n_features)
            y : ndarray of shape (n_samples,) or (n_samples, 1)

        @returns
            r^2 : float, R^2 score. 1.0 is perfect prediction.
        
        """

        y = y.ravel()
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot

    def _validate_params(self):
        """raise ValueError early if any constructor argument is invalid."""
        if self.regul not in [None, 'l1', 'l2', 'elastic_net']:
            raise ValueError("Invalid regularization type!")
        if self.lambda_ < 0:
            raise ValueError("lambda_ must be >= 0")
        if self.regul == 'elastic_net' and not (0 <= self.alpha <= 1):
            raise ValueError("Aplha must be in between 0 and 1.")
        if self.method == 'normal_eq' and self.regul in ['l1', 'elastic_net']:
            raise ValueError("Normal equation supports only L2 regularization.")


def main():
    np.random.seed(42)

    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    methods = ["normal_eq", "gd", "sgd"]
    reguls = [None, "l1", "l2", "elastic_net"]

    print("\nLINEAR REGRESSION (C++ BACKEND) TEST RESULTS\n")
    print(f"{'Method':<10} {'Regul':<12} {'Coef':<12} {'Intercept':<12} {'R2':<8}")
    print("-" * 55)

    for method in methods:
        for regul in reguls:

            if method == "normal_eq" and regul in ["l1", "elastic_net"]:
                continue

            try:
                model = LinearRegression(
                    method=method,
                    regul=regul,
                    lr=0.01,
                    epochs=1000,
                    lambda_=0.1,
                    alpha=0.5,
                )

                model.fit(X, y)
                r2 = model.score(X, y)

                print(
                    f"{method:<10} {str(regul):<12} "
                    f"{model.coef_[0]:<12.4f} {model.intercept_:<12.4f} {r2:<8.4f}"
                )

            except Exception as e:
                print(f"{method:<10} {str(regul):<12} ERROR: {e}")


if __name__ == "__main__":
    main()

