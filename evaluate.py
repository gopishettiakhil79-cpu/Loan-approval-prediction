from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model(y_test, pred, prob):

    print("Accuracy :", accuracy_score(y_test, pred))

    print("Precision :", precision_score(y_test, pred))

    print("Recall :", recall_score(y_test, pred))

    print("F1 Score :", f1_score(y_test, pred))

    print("ROC-AUC :", roc_auc_score(y_test, prob))

    print("\nConfusion Matrix\n")

    print(confusion_matrix(y_test, pred))