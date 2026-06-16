from preprocess import load_data
from evaluate import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

# Load Data
X_train, X_test, y_train, y_test = load_data()

# Logistic Regression
print("\n===== Logistic Regression =====\n")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_prob = lr.predict_proba(X_test)[:, 1]

evaluate_model(
    y_test,
    lr_pred,
    lr_prob
)

# Random Forest
print("\n===== Random Forest =====\n")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_prob = rf.predict_proba(X_test)[:, 1]

evaluate_model(
    y_test,
    rf_pred,
    rf_prob
)

# Cross Validation
cv_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=5
)

print(
    "\nCross Validation Accuracy:",
    cv_scores.mean()
)

# Model Comparison Plot
models = [
    "Logistic Regression",
    "Random Forest"
]

scores = [
    lr.score(X_test, y_test),
    rf.score(X_test, y_test)
]

plt.figure(figsize=(8, 5))

plt.bar(models, scores)

plt.ylabel("Accuracy")

plt.title("Loan Approval Prediction Model Comparison")

plt.show()