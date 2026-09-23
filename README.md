# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved based on applicant demographics, financial information, and credit history.

## 📋 Project Overview 
This project builds and evaluates classification models to predict loan approval outcomes using the Loan_Data.csv dataset. It covers the complete data science workflow: data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

* Goal: Predict `Loan_Status` (Approved / Rejected) from applicant features.

## 📊 Dataset

* File: `Loan_Data.csv`

* Records: 614

* Features: 13

## 📊 Dataset Structure

| Category | Features |
| :--- | :--- |
| **Applicant Details** | Gender, Married, Dependents, Education, Self_Employed |
| **Financial Information** | ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History |
| **Loan Information** | Property_Area |
| **Target Variable** | Loan_Status (Y = Approved, N = Rejected) |

## 🔧 Project Workflow

---

### 1. Data Handling & Preprocessing
* **Feature Cleaning:** Cleaned the `Dependents` column by replacing `'3+'` with `3` and converting the data to a numeric type.
* **Missing Value Imputation:**
  * **Numerical Columns** (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`) → Imputed using **Median Imputation**.
  * **Categorical Columns** (`Gender`, `Married`, `Dependents`, `Self_Employed`) → Imputed using **Mode Imputation**.
* **Categorical Encoding:** Applied **Label Encoding** to convert categorical variables into numerical values.
* **Feature Dropping:** Excluded `Loan_ID` from training as it provides no predictive power.

---

### 2. Exploratory Data Analysis (EDA)
* **Target Distribution:** Evaluated the `Loan_Status` spread, finding roughly **~69% Approved** and **31% Rejected** loans.
* **Predictor Impact:** Identified `Credit_History` as the single strongest statistical predictor for approval status.
* **Outlier Detection:** Utilized boxplots to detect and isolate skewness outliers in `ApplicantIncome`, `LoanAmount`, and `TotalIncome`.
* **Multivariate Analysis:** Constructed a correlation heatmap alongside feature scatter plots to monitor structural data trends (e.g., *Income vs. Loan Amount*).

---

### 3. Feature Engineering
* **Feature Creation:** Generated a comprehensive composite metric: 
  \[\text{TotalIncome} = \text{ApplicantIncome} + \text{CoapplicantIncome}\]
* **Skewness Correction:** Applied **Log Transformations** to normalize data distributions across skewed numerical predictors:
  * `ApplicantIncome_log`
  * `LoanAmount_log`
  * `TotalIncome_log`

---

### 4. Model Building & Implementation
The system trains three distinct classification algorithms using a structured **80/20 train-test split**:

| Model Architecture | Hyperparameters & Configuration Notes |
| :--- | :--- |
| **Logistic Regression** | Features fully scaled using a standard `StandardScaler` pipeline |
| **Decision Tree** | Stratified max depth constrained to `max_depth=5` |
| **Random Forest** | Built using an ensemble configuration of `n_estimators=100`, `max_depth=7` |

---

### 5. Model Evaluation
Model performance benchmarks were measured using baseline accuracy, precision-recall breakdowns, and confusion matrices:

| Model Algorithm | Evaluation Accuracy |
| :--- | :--- |
| 🥇 **Logistic Regression** | **~79%** (Best Performing Model) |
| 🥈 **Random Forest** | **~78%** |
| 🥉 **Decision Tree** | **~77%** |

---

## 🔑 Key Insights
* **Credit Integrity:** A history of healthy credit stands out as the ultimate indicator of credit risk.
* **Demographic Variables:** Applicants from *Semiurban* geographic profiles show higher baseline approval metrics than *Rural* profiles.
* **Income & Status:** Higher cumulative `TotalIncome` tracks closely with successful underwriting status.
* **Education & Cohort:** Candidates holding a *Graduate* degree or listed as *Married* exhibit statistically stronger application profiles.
* **Data Bias Warning:** The original application pool exhibits structural class imbalance (~69% approved).

---

## 🛠️ Technologies & Libraries Used
* **Language Environment:** `Python 3`
* **Data Core Frameworks:** `Pandas` (Data manipulation) & `NumPy` (Numerical matrix math)
* **Visualization Layer:** `Matplotlib` & `Seaborn` (Statistical visualization engineering)
* **Machine Learning Engine:** `Scikit-learn` (Model preprocessing, model training, and metrics extraction)
* **Development Workspace:** `Jupyter Notebook` / `Google Colab`

---

## 🚀 How to Run the Project

### Interactive Notebook
* **View the Notebook:** Open via [nbviewer](https://nbviewer.org) by pasting your repository file URL for a quick, beautifully rendered static preview.

### Run Locally
Clone the code library and run the processing files in a local terminal window:

```bash
# 1. Clone the repository framework
git clone https://github.com/BismaNisar/loan-approval-prediction.git
cd loan-approval-prediction

# 2. Setup production requirements
pip install -r requirements.txt

# 3. Launch pipeline interface
jupyter notebook
```
*Note: Make sure your raw data asset file (`Loan_Data.csv`) remains inside the root execution directory.*

---

## 📁 Project Structure
```text
loan-approval-prediction/
│
├── Loan Approval Prediction.ipynb  # Core execution pipeline file
├── Loan_Data.csv                  # Anonymized data source matrix
├── requirements.txt               # Mandatory execution libraries list
├── .gitignore                     # Git tracking safety filter file
└── README.md                      # Production system documentation
```

---

## ⚠️ Known System Limitations
* **Generalization Limits:** The small sample scale (**614 records**) restricts the model's predictive scope for outside banking environments.
* **Class Skew:** Imbalance profile trends may natively bias classifiers toward the majority application pool.
* **Validation Profile:** The initial baseline launch bypasses structural cross-validation parameters.

---

## 🔮 Future System Improvements
* Implement cross-validation frameworks (e.g., **Stratified K-Fold** validation profiles).
* Deploy exhaustive grid search frameworks to perform programmatic hyperparameter tuning.

---

## 👥 Authors & Collaborators
* **Bisma Nisar**
* **Rameen Khan**
* **Amna Azeem**

---
*If you find this project's machine learning evaluation architecture helpful, please consider leaving a ⭐ on GitHub!*
