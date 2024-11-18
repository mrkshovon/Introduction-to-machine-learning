import numpy as np
from collections import defaultdict

# Function to construct the confusion matrix
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return TP, TN, FP, FN

# Function to calculate accuracy, precision, recall, F1-score
def calculate_metrics(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    
    return accuracy, precision, recall, f1_score

# Assuming  already predicted `y_pred_nb` and `y_pred_knn` from the classifiers

# Naive Bayes results (for example)
TP_nb, TN_nb, FP_nb, FN_nb = confusion_matrix(y_test_nb, y_pred_nb)

# k-NN results (for example)
TP_knn, TN_knn, FP_knn, FN_knn = confusion_matrix(y_test_knn, y_pred_knn)

# Calculate metrics for Naive Bayes
nb_accuracy, nb_precision, nb_recall, nb_f1_score = calculate_metrics(TP_nb, TN_nb, FP_nb, FN_nb)
print(f"Naive Bayes Confusion Matrix: TP={TP_nb}, TN={TN_nb}, FP={FP_nb}, FN={FN_nb}")
print(f"Naive Bayes Metrics: Accuracy={nb_accuracy:.4f}, Precision={nb_precision:.4f}, Recall={nb_recall:.4f}, F1 Score={nb_f1_score:.4f}")

# Calculate metrics for k-NN
knn_accuracy, knn_precision, knn_recall, knn_f1_score = calculate_metrics(TP_knn, TN_knn, FP_knn, FN_knn)
print(f"k-NN Confusion Matrix: TP={TP_knn}, TN={TN_knn}, FP={FP_knn}, FN={FN_knn}")
print(f"k-NN Metrics: Accuracy={knn_accuracy:.4f}, Precision={knn_precision:.4f}, Recall={knn_recall:.4f}, F1 Score={knn_f1_score:.4f}")
