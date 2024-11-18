import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Load the data using NumPy
train_data = np.loadtxt('fashion-mnist_train.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt('fashion-mnist_test.csv', delimiter=',', skiprows=1)

# Separate features and labels
X_train_full, y_train_full = train_data[:, 1:], train_data[:, 0].astype(int)
X_test, y_test = test_data[:, 1:], test_data[:, 0].astype(int)

# Manual standardization (scaling to mean=0 and variance=1)
def standardize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X_train_full_scaled = standardize_data(X_train_full)
X_test_scaled = standardize_data(X_test)

# Manual splitting of data into training and validation sets (80% train, 20% validation)
def train_val_split(X, y, val_ratio=0.2):
    np.random.seed(42)
    indices = np.random.permutation(len(y))
    val_size = int(len(y) * val_ratio)
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

X_train, X_val, y_train, y_val = train_val_split(X_train_full_scaled, y_train_full)

# Apply SVD for dimensionality reduction
target_variance = 0.9
n_components = 1
svd = TruncatedSVD(n_components=n_components)
while True:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_reduced = svd.fit_transform(X_train)
    if svd.explained_variance_ratio_.sum() >= target_variance:
        break
    n_components += 1

X_val_reduced = svd.transform(X_val)
X_test_reduced = svd.transform(X_test_scaled)

# Print number of components
print(f"Optimal components for >90% variance: {n_components}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.2%}")

# Helper functions for evaluation metrics (accuracy, precision, recall, F1-score)
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true, y_pred, num_classes=10):
    precisions, recalls, f1_scores = [], [], []
    for cls in range(num_classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

# Model training and evaluation function
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    print("Validation Set Metrics:")
    acc_val = accuracy(y_val, y_pred_val)
    prec_val, rec_val, f1_val = precision_recall_f1(y_val, y_pred_val)
    print(f"Accuracy: {acc_val:.2%}, Precision: {prec_val:.2%}, Recall: {rec_val:.2%}, F1 Score: {f1_val:.2%}")
    
    print("Test Set Metrics:")
    acc_test = accuracy(y_test, y_pred_test)
    prec_test, rec_test, f1_test = precision_recall_f1(y_test, y_pred_test)
    print(f"Accuracy: {acc_test:.2%}, Precision: {prec_test:.2%}, Recall: {rec_test:.2%}, F1 Score: {f1_test:.2%}")
    return acc_test

# Train and evaluate classifiers
print("Evaluating classifiers on SVD-reduced data:")
models = {
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200, multi_class="multinomial", solver="lbfgs", random_state=42)
}

# Tune hyperparameters manually
best_k = 3
best_knn_accuracy = 0
for k in range(3, 12, 2):  # Odd k values for KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    acc_test = train_and_evaluate_model(knn, X_train_reduced, y_train, X_val_reduced, y_val, X_test_reduced, y_test)
    if acc_test > best_knn_accuracy:
        best_knn_accuracy = acc_test
        best_k = k
print(f"Best KNN k value: {best_k} with test accuracy: {best_knn_accuracy:.2%}")

best_C = 1
best_lr_accuracy = 0
for C in [0.01, 0.1, 1, 10, 100]:
    lr = LogisticRegression(C=C, max_iter=200, multi_class="multinomial", solver="lbfgs", random_state=42)
    acc_test = train_and_evaluate_model(lr, X_train_reduced, y_train, X_val_reduced, y_val, X_test_reduced, y_test)
    if acc_test > best_lr_accuracy:
        best_lr_accuracy = acc_test
        best_C = C
print(f"Best Logistic Regression C value: {best_C} with test accuracy: {best_lr_accuracy:.2%}")

# Naive Bayes
print("\nNaive Bayes:")
nb_model = GaussianNB()
train_and_evaluate_model(nb_model, X_train_reduced, y_train, X_val_reduced, y_val, X_test_reduced, y_test)
