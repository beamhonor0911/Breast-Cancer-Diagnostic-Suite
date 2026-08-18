import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def run_benchmark(data_path="breast-cancer.csv"):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns], inplace=True)
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Support Vector Machine (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Calibrated Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model Architecture": name,
            "CV Recall (5-Fold)": f"{np.mean(cv_res['test_recall']):.4f}",
            "CV ROC-AUC": f"{np.mean(cv_res['test_roc_auc']):.4f}",
            "Test Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
            "Test Precision": f"{precision_score(y_test, y_pred):.4f}",
            "Test Recall": f"{recall_score(y_test, y_pred):.4f}",
            "Test ROC-AUC": f"{roc_auc_score(y_test, y_prob):.4f}"
        })

    report_df = pd.DataFrame(results)
    print("\n================ MODEL BENCHMARK RESULTS ================")
    print(report_df.to_string(index=False))
    report_df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to 'benchmark_results.csv'")

if __name__ == "__main__":
    run_benchmark()