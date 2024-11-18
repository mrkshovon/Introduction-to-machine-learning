import numpy as np
from collections import Counter

# Confusion matrix function
def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return matrix

# Metrics calculation functions
def calculate_metrics(conf_matrix):
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (FP + FN + TP)
    
    accuracy = np.sum(TP) / np.sum(conf_matrix)
    precision = np.mean(TP / (TP + FP))
    recall = np.mean(TP / (TP + FN))
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1_score

# After training and predicting using Naive Bayes and k-NN (using the previous code)
# Example: Let's assume `nb_predictions` and `knn_test_predictions` are the predicted labels from Naive Bayes and k-NN.

# Confusion matrix for Naive Bayes
nb_conf_matrix = confusion_matrix(y_nb_test, nb_predictions)
print("Naive Bayes Confusion Matrix:")
print(nb_conf_matrix)

# Confusion matrix for k-NN
knn_conf_matrix = confusion_matrix(y_knn_test, knn_test_predictions)
print("k-NN Confusion Matrix:")
print(knn_conf_matrix)

# Calculate metrics for Naive Bayes
nb_accuracy, nb_precision, nb_recall, nb_f1 = calculate_metrics(nb_conf_matrix)
print(f"Naive Bayes - Accuracy: {nb_accuracy:.4f}, Precision: {nb_precision:.4f}, Recall: {nb_recall:.4f}, F1-Score: {nb_f1:.4f}")

# Calculate metrics for k-NN
knn_accuracy, knn_precision, knn_recall, knn_f1 = calculate_metrics(knn_conf_matrix)
print(f"k-NN - Accuracy: {knn_accuracy:.4f}, Precision: {knn_precision:.4f}, Recall: {knn_recall:.4f}, F1-Score: {knn_f1:.4f}")
