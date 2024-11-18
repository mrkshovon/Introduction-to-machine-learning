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

# Step 1: Load data from a file
def load_data(file_path):
    """Load data from a CSV or TXT file."""
    data = np.loadtxt(file_path)  # Load data assuming it's space-separated
    X = data[:, :2]  # First two columns as features
    y = data[:, 2]   # Last column as labels
    return X, y

# Step 2: Load the dataset
a, b = load_data('Question4.txt')

# Step 3: Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(a[:, 0], a[:, 1], c=b, cmap='bwr', alpha=0.5, edgecolor='k')

# Step 4: Customize the plot
plt.title('Scatter Plot of Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')  # Color bar to indicate class labels
plt.grid(True)

# Step 5: Save and show the plot
plt.savefig("scatter_x_y.png")
plt.show()

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
        cost = - (1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        #cost = - (1 / m) * np.sum(np.multiply(y, np.log(A)) + np.multiply(1 - y, np.log(1 - A)))

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
    plt.savefig("Cost_Function_over_Iterations.png")

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
plt.savefig("Effect_of_Learning_Rate_on_Cost_Function.png")
