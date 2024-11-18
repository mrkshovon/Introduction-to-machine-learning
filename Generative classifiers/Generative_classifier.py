import pandas as pd
import numpy as np
from collections import Counter

# Load dataset
file_path = 'heart_disease_uci.csv'
data = pd.read_csv(file_path)

# Assuming the last column is the target (classification label)
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target

# Split dataset (using ratios for Naive Bayes and k-NN)
def split_dataset(X, y, train_size, val_size=None):
    # Shuffle the data
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    if val_size:
        train_idx = int(len(X) * train_size)
        val_idx = int(len(X) * (train_size + val_size))
        return X[:train_idx], y[:train_idx], X[train_idx:val_idx], y[train_idx:val_idx], X[val_idx:], y[val_idx:]
    else:
        train_idx = int(len(X) * train_size)
        return X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

# 1. Naive Bayes split: 80% training, 20% test
X_nb_train, y_nb_train, X_nb_test, y_nb_test = split_dataset(X, y, train_size=0.8)

# 2. k-NN split: 60% training, 20% validation, 20% test
X_knn_train, y_knn_train, X_knn_val, y_knn_val, X_knn_test, y_knn_test = split_dataset(X, y, train_size=0.6, val_size=0.2)

# Gaussian Naive Bayes Classifier
class GaussianNB:
    def fit(self, X, y):
        # Separate data by class
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
    
    def gaussian_pdf(self, x, mean, var):
        eps = 1e-9  # Small constant to avoid division by zero
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coeff * exponent
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                posterior = np.sum(np.log(self.gaussian_pdf(x, self.mean[c], self.var[c])))
                posteriors.append(prior + posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

# Train Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_nb_train, y_nb_train)
nb_predictions = nb_model.predict(X_nb_test)

# Calculate accuracy for Naive Bayes
nb_accuracy = np.mean(nb_predictions == y_nb_test)
print(f"Naive Bayes Classifier Accuracy: {nb_accuracy:.4f}")

# k-NN Classifier
class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

# Fine-tune k on the validation set
best_k = 1
best_accuracy = 0

for k in range(1, 21):
    knn_model = KNearestNeighbors(k=k)
    knn_model.fit(X_knn_train, y_knn_train)
    knn_val_predictions = knn_model.predict(X_knn_val)
    val_accuracy = np.mean(knn_val_predictions == y_knn_val)
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_k = k

print(f"Best k found: {best_k} with validation accuracy: {best_accuracy:.4f}")

# Train final k-NN model using the best k and test it
final_knn_model = KNearestNeighbors(k=best_k)
final_knn_model.fit(X_knn_train, y_knn_train)
knn_test_predictions = final_knn_model.predict(X_knn_test)

# Calculate accuracy for k-NN
knn_accuracy = np.mean(knn_test_predictions == y_knn_test)
print(f"k-NN Classifier Accuracy with k={best_k}: {knn_accuracy:.4f}")
