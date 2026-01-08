# 🛡️ Financial Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project focuses on detecting fraudulent mobile money transactions using advanced Machine Learning techniques. The dataset is highly imbalanced, mimicking real-world financial data where fraud cases are rare but costly.

The goal was to build a robust pipeline that prioritizes **Precision and Recall (F1-Score)** over simple Accuracy, ensuring that actual fraud cases are caught while minimizing false alarms.

**Author:** Shaury Pratap Singh  
**Institution:** New Jersey Institute of Technology (NJIT) - Ying Wu College of Computing  
**Degree:** M.S. in Data Science

---

## 🚀 Key Features & Methodologies

### 1. **Rigorous Data Preprocessing**
* **Data Leakage Prevention:** Strictly applied the "Split-First, Scale-Later" methodology. Feature selection (`SelectKBest`) and Scaling (`StandardScaler`) were fit **only** on the training set to prevent information from the test set leaking into the model.
* **Feature Engineering:**
    * Derived `hour_of_day` from the time-step column to capture temporal fraud patterns.
    * Calculated `balance_error` to flag discrepancies in origin and destination accounts.

### 2. **Handling Class Imbalance**
* Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data, allowing models to learn the "Fraud" class effectively without bias.
* *Note: SMOTE was applied strictly to the training set, keeping the test set pure.*

### 3. **Automated Model Comparison**
* Implemented an automated pipeline to train and evaluate **9 different algorithms**:
    * **Ensemble Methods:** Random Forest, XGBoost, LightGBM, CatBoost.
    * **Neural Networks:** Multi-Layer Perceptron (MLP).
    * **Traditional ML:** Logistic Regression, KNN, Naive Bayes, Decision Trees.
* Used `GridSearchCV` for hyperparameter tuning (optimizing for F1-Score).

---

## 📊 Model Performance Results

We evaluated models based on **ROC-AUC** and **F1-Score**, as Accuracy is misleading for imbalanced datasets.

| Model | Accuracy | F1 Score | ROC AUC |
| :--- | :--- | :--- | :--- |
| **MLP (ANN)** 🏆 | **99.66%** | **0.4383** | **0.9989** |
| XGBoost | 99.42% | 0.3067 | 0.9996 |
| Random Forest | 98.01% | 0.1141 | 0.9909 |
| Logistic Regression | 94.87% | 0.0464 | 0.9222 |

> **Winner:** The **MLP (Artificial Neural Network)** was selected as the final model due to its superior balance of Precision and Recall, achieving the highest ROC-AUC score of ~0.999.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE)
* **Boosting Algorithms:** XGBoost, LightGBM, CatBoost
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Repository Structure
├── data/ # Dataset files (not included in repo due to size) ├── notebooks/ # Jupyter Notebooks for EDA and Modeling ├── models/ # Saved models (.pkl files) ├── images/ # Plots and confusion matrices ├── README.md # Project documentation └── requirements.txt # List of dependencies

Here is a professional, recruiter-ready **README.md** file. I have written this specifically to highlight your status as a **Data Science Graduate Student at NJIT**, emphasizing technical rigor (like fixing data leakage and handling imbalance).

You can copy the raw code block below and paste it directly into a file named `README.md` on your GitHub repository.

---

### **Copy This Code Block:**

```markdown
# 🛡️ Financial Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project focuses on detecting fraudulent mobile money transactions using advanced Machine Learning techniques. The dataset is highly imbalanced, mimicking real-world financial data where fraud cases are rare but costly.

The goal was to build a robust pipeline that prioritizes **Precision and Recall (F1-Score)** over simple Accuracy, ensuring that actual fraud cases are caught while minimizing false alarms.

**Author:** Shaury Pratap Singh  
**Institution:** New Jersey Institute of Technology (NJIT) - Ying Wu College of Computing  
**Degree:** M.S. in Data Science

---

## 🚀 Key Features & Methodologies

### 1. **Rigorous Data Preprocessing**
* **Data Leakage Prevention:** Strictly applied the "Split-First, Scale-Later" methodology. Feature selection (`SelectKBest`) and Scaling (`StandardScaler`) were fit **only** on the training set to prevent information from the test set leaking into the model.
* **Feature Engineering:**
    * Derived `hour_of_day` from the time-step column to capture temporal fraud patterns.
    * Calculated `balance_error` to flag discrepancies in origin and destination accounts.

### 2. **Handling Class Imbalance**
* Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data, allowing models to learn the "Fraud" class effectively without bias.
* *Note: SMOTE was applied strictly to the training set, keeping the test set pure.*

### 3. **Automated Model Comparison**
* Implemented an automated pipeline to train and evaluate **9 different algorithms**:
    * **Ensemble Methods:** Random Forest, XGBoost, LightGBM, CatBoost.
    * **Neural Networks:** Multi-Layer Perceptron (MLP).
    * **Traditional ML:** Logistic Regression, KNN, Naive Bayes, Decision Trees.
* Used `GridSearchCV` for hyperparameter tuning (optimizing for F1-Score).

---

## 📊 Model Performance Results

We evaluated models based on **ROC-AUC** and **F1-Score**, as Accuracy is misleading for imbalanced datasets.

| Model | Accuracy | F1 Score | ROC AUC |
| :--- | :--- | :--- | :--- |
| **MLP (ANN)** 🏆 | **99.66%** | **0.4383** | **0.9989** |
| XGBoost | 99.42% | 0.3067 | 0.9996 |
| Random Forest | 98.01% | 0.1141 | 0.9909 |
| Logistic Regression | 94.87% | 0.0464 | 0.9222 |

> **Winner:** The **MLP (Artificial Neural Network)** was selected as the final model due to its superior balance of Precision and Recall, achieving the highest ROC-AUC score of ~0.999.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE)
* **Boosting Algorithms:** XGBoost, LightGBM, CatBoost
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Repository Structure


```

├── data/                   # Dataset files (not included in repo due to size)
├── notebooks/              # Jupyter Notebooks for EDA and Modeling
├── models/                 # Saved models (.pkl files)
├── images/                 # Plots and confusion matrices
├── README.md               # Project documentation
└── requirements.txt        # List of dependencies

---

## ⚙️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Fraud-Detection-ML.git](https://github.com/YOUR_USERNAME/Fraud-Detection-ML.git)
    cd Fraud-Detection-ML
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Notebook:**
    Open `Fraud_Detection_ML_Project.ipynb` in Jupyter Notebook or Google Colab to reproduce the training and evaluation pipeline.

---

## 🔮 Future Scope
* **Deployment:** Wrap the final MLP model in a Flask/FastAPI backend for real-time inference.
* **Deep Learning:** Experiment with LSTM (Long Short-Term Memory) networks to capture sequential transaction patterns.
* **Explainability:** Implement SHAP values to explain *why* a specific transaction was flagged as fraud.

---

### 📬 Contact
* **LinkedIn:** [Your LinkedIn Profile Link Here]
* **Email:** [Your Email Here]
