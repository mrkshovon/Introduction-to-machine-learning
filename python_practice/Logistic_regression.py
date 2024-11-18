import numpy as np
import pandas as pd
from numpy import matrix, zeros
from matplotlib import pyplot as plt

# Load data from file
def loadDataSet():
    X = []
    y = []
    with open('Question4.txt') as f:
        for line in f.readlines():
            nline = line.strip().split()
            X.append([float(nline[0]), float(nline[1])])
            y.append(int(nline[2]))
    return matrix(X).T, matrix(y).T

X, y = loadDataSet()

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def logistic_regression(X, y, W, b, alpha, iterations):
    m, n = X.shape
    J = zeros((iterations, 1))

    for i in range(iterations):
        # Step 1: Forward propagation
        Z = np.dot(W.T, X) + b
        A = sigmoid(Z)
        
        # Compute cost function
        #cost = - (1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        cost = - (1 / m) * np.sum(np.multiply(y, np.log(A)) + np.multiply(1 - y, np.log(1 - A)))

        J[i] = cost
        
        # Step 2: Backpropagation
        dZ = A - y
        dW = (1 / m) * np.dot(X, dZ.T)
        db = (1 / m) * np.sum(dZ)
        
        # Step 3: Gradient descent
        W = W - alpha * dW
        b = b - alpha * db
        
    return W, b, J

def plotBestFit(X, y, J, W, b):
    # Plot cost function
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(J)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function over Iterations')
    
    # Plot decision boundary
    plt.subplot(1, 2, 2)
    plt.scatter(X[0, y.flatten() == 0], X[1, y.flatten() == 0], color='red', label='Class 0')
    plt.scatter(X[0, y.flatten() == 1], X[1, y.flatten() == 1], color='blue', label='Class 1')

    # Plot decision boundary
    x1_min, x1_max = X[0, :].min(), X[0, :].max()
    x2_min, x2_max = X[1, :].min(), X[1, :].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    Z = sigmoid(np.dot(np.c_[xx1.ravel(), xx2.ravel()], W) + b)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='viridis')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()

# Initialize parameters
num = X.shape[0]  # number of features
n = X.shape[1]    # number of samples
W = np.random.randn(num, 1)  # Random initialization of weights
b = np.random.randn()       # Random initialization of bias
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Train logistic regression model
W, b, J = logistic_regression(X, y, W, b, alpha, iterations)

# Plot results
plotBestFit(X, y, J, W, b)

# Experiment with different learning rates
alphas = [0.1, 0.01, 0.001]
plt.figure(figsize=(12, 6))
for alpha in alphas:
    W, b, J = logistic_regression(X, y, W, b, alpha, iterations)
    plt.plot(J, label=f'Alpha = {alpha}')

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Effect of Learning Rate on Cost Function')
plt.legend()
plt.grid(True)
plt.show()
