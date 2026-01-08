from linreg_internal import LinearRegression
import numpy as np

# generate data
X = 2 * np.random.rand(100, 1)
y = (4 + 3 * X + np.random.randn(100, 1)).ravel()

# create GD model
model = LinearRegression(
    "gd",          # method
    "none",        # regul
    0.01,          # learning rate
    1000,          # epochs
    0.0,           # lambda
    0.5            # alpha (unused here)
)

# train
model.fit(X, y)
