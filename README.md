# Student Status Prediction via Specialist Expert Pool & DCS-LCA

This repository contains the data science research project implementation for the multi-class classification of student academic statuses (_Dropout_, _Enrolled_, and _Graduate_). The **Proposed Method** utilizes a novel combination of a **Class-Specific Expert Pool** (hyper-optimized using **Optuna**) dynamically synthesized via **Dynamic Classifier Selection - Local Class Accuracy (DCS-LCA)**.

## Experimental Results Summary

Empirical experiments demonstrate that dynamic decision delegation based on local competence effectively mitigates the _destructive noise_ inherent in over-specialized base models. As a result, the proposed framework achieves the highest overall performance compared to both individual single models and traditional static ensemble combinations:

| Model Architecture         | Approach / Algorithm                  | F1-Score Macro | Evaluation Status           |
| :------------------------- | :------------------------------------ | :------------: | :-------------------------- |
| **Single Model (Class 0)** | Naive Bayes (Dropout Specialist)      |     0.7916     | Baseline 1                  |
| **Single Model (Class 2)** | SVM (Graduate Specialist)             |     0.8101     | Baseline 2                  |
| **Single Model (Class 1)** | XGBoost (Enrolled Specialist)         |     0.8608     | Baseline 3                  |
| **Static Ensemble**        | Probability Harvesting / Soft Pooling |     0.8215     | Combination Baseline        |
| **Dynamic Ensemble**       | **DCS-LCA (K=7)**                     |   **0.8638**   | **Proposed Method (Ours) ** |

---

## Project Pipeline Structure

The codebase is structured completely modularly and divided into 5 distinct Jupyter Notebook stages:

1. **`1_data_preprocessing.ipynb`**
   - Raw data cleaning, data type casting, and the implementation of _KNN Overlap Cleaning_ to sharpen class decision boundaries.
2. **`2_hyperparameter_tuning.ipynb`**
   - Class-specific feature selection using a _One-vs-Rest_ (OvR) approach and extreme hyperparameter tuning via Optuna per target class to establish 3 independent expert models (_XGBoost, SVM, and GaussianNB_).
3. **`3_synthesis_evaluation.ipynb`**
   - Implementation of the proposed DCS-LCA dynamic selection algorithm, local competence neighborhood parameter exploration ($K$-Neighbors tuning), and final ensemble model export.
4. **`4_probability_harvesting.ipynb`**
   - Implementation of the traditional baseline model (_Probability Harvesting_) serving as a comparative engine for static ensemble pooling configurations.
5. **`5_full_comparison.ipynb`**
   - A comprehensive cross-method evaluation hub featuring side-by-side confusion matrix visualizations, multi-class One-vs-Rest ROC-AUC curves, and ensemble decision diversity/disagreement analysis.

---

## Getting Started & Installation

1. Clone this repository to your local machine.
2. Ensure that your virtual environment (`venv`) is activated.
3. Install the required in the requirements.txt
