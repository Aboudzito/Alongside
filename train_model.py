"""
Train and evaluate a model that predicts whether a student understood a session.
Runs on the synthetic demo data (the real survey data is private).
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# 1. Load data and prepare features
df = pd.read_csv("synthetic_student_responses.csv")
X = pd.get_dummies(df.drop(columns=["student_id", "understood"]), columns=["topic"])
y = df["understood"]
print("Class balance:", y.value_counts(normalize=True).round(3).to_dict())

# 2. Train/test split (stratify keeps the same class balance in both parts)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Baseline: always predict the most common class
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(x_train, y_train)
print(f"\nBaseline accuracy: {accuracy_score(y_test, baseline.predict(x_test))*100:.2f}%")

# 4. Hyperparameter tuning with GridSearchCV (5-fold cross-validation)
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 6, 12],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, scoring="f1_macro")
grid.fit(x_train, y_train)
rf = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# 5. Evaluate on the held-out test set
y_pred = rf.predict(x_test)
print(f"\nRandom Forest test accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["not understood", "understood"]))

# 6. Confusion matrix and feature importance
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion matrix (test set)")
plt.show()

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().tail(10).plot(kind="barh")
plt.title("Top 10 features")
plt.show()
