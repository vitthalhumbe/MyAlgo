# MyAlgo
MyAlgo is my Machine Learning project which is focused on building Linear Regression and Artificial Neural Network from scratch without using high level machine learning libraries such as Tensorflow, PyTorch and scikit-learn.

The main goal of this project is to combining
- **python** as a user facing python API
- **C++** as a high performance computational backend (only for numerical calculations)

Read More : ![documentations](Documentation/MyAlgo.pdf)


## Implemented Algorithms
### Linear Regression
|Feature|Supported values|
|-------|----------------|
|Methods|Normal Equation, GD, SGD|
|Regularization|Lasso, Ridge, Elastic net|

Example usage:
```python
model = LinearRegression(method='gd', lr=0.01, epochs=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 2. ANN
|Feature|Supported values|
|-------|----------------|
|Layers|Flatten, Dense|  
|Activations|ReLU, sigmoid, Tanh, Softmax|
|Loss functions|MSE, cross-entropy|

Example usage:
```python
model = NeuralNetwork(
  Flatten((28,28)),
  Dense(128, activation='relu'),
  Dense(10, activation='softmax')
)
model.compile(loss='categorical_crossentropy', lr=0.01)
model.train(X_train, y_train, epochs=11, batch_size=32)
```

## Results of MINST dataset training 
![image alt](https://github.com/vitthalhumbe/MyAlgo/blob/3ac8685bc65afd62119edb26d3aa7988723c90a9/results/MNIST%20loss%20vs%20Accuracy.png)

## Author

Vitthal Humbe (Kuron) 
B.tech second year student


