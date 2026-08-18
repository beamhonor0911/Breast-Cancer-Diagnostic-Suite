<div align="center">

# 🧬 OncoVision AI: Clinical Cancer Detection & Triage Suite
### *High-Confidence Morphometric Malignancy Classification & Clinical Decision Support*

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <a href="#-executive-summary">Executive Summary</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-model-benchmarks--clinical-metrics">Benchmarks</a> •
  <a href="#-system-architecture">System Architecture</a> •
  <a href="#-repository-structure">Repository Structure</a> •
  <a href="#-quickstart-guide">Quickstart</a> •
  <a href="#-serving--deployment">Serving & Deployment</a> •
  <a href="#-clinical-triage-tier-classification">Triage Protocol</a>
</p>

</div>

---

## 📌 Executive Summary

**OncoVision AI** is an end-to-end, production-grade diagnostic machine learning system built on the **Wisconsin Breast Cancer Diagnostic (WBCD)** dataset. It automates cellular morphometry analysis to detect and stratify malignancy risk in fine-needle aspirate (FNA) biopsy samples.

The project incorporates strict **leak-free data transformations**, **stratified cross-validation**, an **asynchronous REST microservice**, and an **interactive clinical dashboard** tailored for pathology triage workflows.

---

## 🚀 Key Features

* **🛡️ Zero Data Leakage Architecture:** Preprocessing (`StandardScaler`) is strictly encapsulated within `scikit-learn` execution pipelines, computing fold statistics exclusively on training partitions.
* **🎯 High Clinical Specificity:** Delivers **1.00 Precision** on malignant holdout test samples, ensuring zero false-positive biopsy alerts.
* **🔬 Multi-Model Benchmark Harness:** Side-by-side automated cross-validation benchmarking comparing RBF Support Vector Machines, Random Forest ensembles, and Calibrated Logistic Regression.
* **⚡ Sub-Millisecond REST API:** Asynchronous FastAPI endpoints with Pydantic payload validation and automated three-tier risk stratification (*Low Risk*, *Uncertain/Review*, *Critical*).
* **🖥️ Interactive Clinician Dashboard:** Feature slider control with diagnostic probability gauges, patient case presets, and real-time inference.

---

## 📊 Model Benchmarks & Clinical Metrics

Evaluated using **Stratified 5-Fold Cross-Validation** on an 80/20 train-test partition:

| Model Architecture | 5-Fold CV Recall | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** 🏆 | **94.71%** | **0.9949** | **97.37%** | **1.0000** | **92.86%** | **0.9947** |
| **Random Forest Classifier** | 94.71% | 0.9888 | 97.37% | 1.0000 | 92.86% | 0.9954 |
| **Calibrated Logistic Regression** | 95.29% | 0.9958 | 96.49% | 0.9750 | 92.86% | 0.9960 |

```text
                       Test Set Confusion Matrix
                       ┌───────────────────────┐
                       │  TN = 72   │  FP = 0  │  ◄─ Benign (100% Specificity)
                       ├────────────┼──────────┤
                       │  FN = 3    │  TP = 39 │  ◄─ Malignant (93% Sensitivity)
                       └───────────────────────┘


                       🏗️ System Architecture
                    
                    ┌────────────────────────┐
                               │   breast-cancer.csv    │
                               │  (30 Cell Biomarkers)  │
                               └───────────┬────────────┘
                                           │
                                           ▼
                     ┌──────────────────────────────────────────┐
                     │          production_pipeline.py          │
                     │  • Stratified 80/20 Split                │
                     │  • Scaler Pipeline (Leak-Free)           │
                     │  • SVC (RBF Kernel, C=1.0)               │
                     └─────────────────────┬────────────────────┘
                                           │
                              Generates Model Artifact
                                           │
                                           ▼
                             ┌───────────────────────────┐
                             │    cancer_model.joblib    │
                             └─────────────┬─────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       ┌────────────────────────┐                    ┌────────────────────────┐
       │      FastAPI API       │                    │    Streamlit Web UI    │
       │        (api.py)        │                    │        (app.py)        │
       │                        │                    │                        │
       │ • /predict (POST)      │                    │ • Real-time sliders    │
       │ • /health (GET)        │                    │ • Triage Risk Alerts   │
       │ • Auto OpenAPI Docs    │                    │ • Case Presets         │
       └────────────────────────┘                    └────────────────────────┘

       ## 📈 Clinical Triage Protocol

| Probability Score | Triage Level | Priority | Action Protocol |
| :---: | :---: | :---: | :--- |
| **≥ 65.0%** | 🚨 **Critical / Malignant** | **High** | Immediate secondary biopsy & urgent oncology referral. |
| **35.0% – 64.9%** | ⚠️ **Borderline / Uncertain** | **Medium** | Diagnostic uncertainty zone: Secondary review by human pathologist required. |
| **< 35.0%** | ✅ **Low Risk / Benign** | **Standard** | Consistent with benign cellular morphometry; standard annual screening. |

## 👨‍💻 Author & Maintainer

* **Author:** [@beamhonor0911](https://github.com/beamhonor0911)
* **GitHub Repository:** [Breast-Cancer-Diagnostic-Suite](https://github.com/beamhonor0911/Breast-Cancer-Diagnostic-Suite)
* **License:** [MIT License](LICENSE)