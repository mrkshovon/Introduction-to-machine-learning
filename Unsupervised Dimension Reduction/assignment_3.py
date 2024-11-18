import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the datasets
train_data = pd.read_csv('/mnt/data/fashion-mnist_train.csv')
test_data = pd.read_csv('/mnt/data/fashion-mnist_test.csv')

# Separate features and labels
X_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

# Split the training data further to create a validation set
X_train_full, X_val, y_train_full, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Find the number of SVD components that explain just above 90% variance
def find_svd_components(X, target_variance=0.9):
    for n_components in range(1, X.shape[1] + 1):
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(X)
        cumulative_variance = svd.explained_variance_ratio_.sum()
        if cumulative_variance >= target_variance:
            return n_components, svd

# Apply SVD to reduce dimensions
optimal_components, svd_model = find_svd_components(X_train_full_scaled, target_variance=0.9)
X_train_reduced = svd_model.transform(X_train_full_scaled)
X_val_reduced = svd_model.transform(X_val_scaled)
X_test_reduced = svd_model.transform(X_test_scaled)

# Report results for SVD
print(f"Optimal components to explain >90% variance: {optimal_components}")
print(f"Explained variance ratio: {svd_model.explained_variance_ratio_.sum():.2%}")

# Define models and parameters for tuning
models = {
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200, multi_class="multinomial", solver="lbfgs", random_state=42)
}

# Hyperparameters for Grid Search
params = {
    "KNN": {"n_neighbors": range(3, 12, 2)},
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]}  # Inverse of regularization strength
}

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    print("Validation Set Classification Report:")
    print(classification_report(y_val, y_pred_val))
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred_test))
    return accuracy_score(y_test, y_pred_test)

# Train and evaluate models on both reduced and original data
print("Evaluating classifiers on SVD-reduced data:")
for name, model in models.items():
    print(f"\n{name} on SVD-reduced data:")
    if name in params:
        grid = GridSearchCV(model, params[name], scoring="accuracy", cv=3)
        grid.fit(X_train_reduced, y_train_full)
        best_model = grid.best_estimator_
        print(f"Best parameters for {name}: {grid.best_params_}")
    else:
        best_model = model
    accuracy = evaluate_model(best_model, X_train_reduced, y_train_full, X_val_reduced, y_val, X_test_reduced, y_test)
    print(f"{name} Test Accuracy on SVD-reduced data: {accuracy:.2%}")

print("\nEvaluating classifiers on original data:")
for name, model in models.items():
    print(f"\n{name} on original data:")
    if name in params:
        grid = GridSearchCV(model, params[name], scoring="accuracy", cv=3)
        grid.fit(X_train_full_scaled, y_train_full)
        best_model = grid.best_estimator_
        print(f"Best parameters for {name}: {grid.best_params_}")
    else:
        best_model = model
    accuracy = evaluate_model(best_model, X_train_full_scaled, y_train_full, X_val_scaled, y_val, X_test_scaled, y_test)
    print(f"{name} Test Accuracy on original data: {accuracy:.2%}")
