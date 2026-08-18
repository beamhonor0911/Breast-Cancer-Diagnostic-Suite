import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def load_data(filepath="breast-cancer.csv"):
    df = pd.read_csv(filepath)
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
    if "Unnamed: 32" in df.columns:
        df.drop(columns=["Unnamed: 32"], inplace=True)
        
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y

def train_production_pipeline(X, y):
    # Leak-free stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scikit-Learn Pipeline guarantees scaler is only fit on training folds
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
    ])
    
    # 5-fold stratified cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    print("=== MODEL PERFORMANCE ===")
    print(f"5-Fold CV Mean Recall : {np.mean(cv_scores['test_recall']):.4f}")
    print(f"Test Accuracy         : {pipeline.score(X_test, y_test):.4f}")
    print(f"Test ROC-AUC Score    : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
    
    # Save the serialized production model artifact
    joblib.dump(pipeline, "cancer_model.joblib")
    print("\n[SUCCESS] Model artifact saved to 'cancer_model.joblib'")
    
    return pipeline

if __name__ == "__main__":
    if os.path.exists("breast-cancer.csv"):
        X, y = load_data("breast-cancer.csv")
        train_production_pipeline(X, y)
    else:
        print("Error: 'breast-cancer.csv' not found in working directory.")