# Loan-Approval-Prediction
A collaborative data science project focused on predicting credit risk and automated loan approvals. Developed a machine learning pipe line to classify applicants based on financial history. personally owned the exploratory data analysis, and model evaluation phases to optimize prediction accuracy.
📋 Project Overview
This project builds and evaluates classification models to predict loan approval outcomes using the Loan_Data.csv dataset. It covers the complete data science workflow: data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

Goal: Predict Loan_Status (Approved / Rejected) from applicant features.

📊 Dataset
File: Loan_Data.csv

Records: 614

Features: 13

Category	Features
Applicant Details	Gender, Married, Dependents, Education, Self_Employed
Financial Information	ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
Loan Information	Property_Area
Target	Loan_Status (Y = Approved, N = Rejected)
🔧 Workflow
1. Data Handling & Preprocessing
Fixed the Dependents column by replacing '3+' with 3 and converting to numeric.

Handled missing values:

Numerical columns (LoanAmount, Loan_Amount_Term, Credit_History) → median imputation.

Categorical columns (Gender, Married, Dependents, Self_Employed) → mode imputation.

Applied Label Encoding to categorical variables.

Dropped Loan_ID (not useful for prediction).

2. Exploratory Data Analysis (EDA)
Loan Status distribution (~69% approved, 31% rejected).

Credit History vs. Loan Status (strongest predictor).

Outlier detection using boxplots (ApplicantIncome, LoanAmount, TotalIncome).

Correlation heatmap.

Categorical features vs. Loan Status (Property_Area, Self_Employed).

Income vs. Loan Amount scatter plot by Loan Status.

3. Feature Engineering
Created TotalIncome = ApplicantIncome + CoapplicantIncome.

Applied log transformations to skewed features:

ApplicantIncome_log

LoanAmount_log

TotalIncome_log

4. Model Building
Trained three classification models on an 80/20 train-test split:

Model	Notes
Logistic Regression	Features scaled with StandardScaler
Decision Tree	max_depth=5
Random Forest	n_estimators=100, max_depth=7
5. Model Evaluation
Evaluated using accuracy, confusion matrix, and classification report.

Model	Accuracy
Logistic Regression	~79%
Decision Tree	~77%
Random Forest	~78%
Best Model: Logistic Regression

🔑 Key Insights
Credit History is the single most important predictor of loan approval.

Semiurban applicants have higher approval rates than Rural applicants.

Higher TotalIncome correlates with loan approval.

Married and Graduate applicants show higher approval rates.

The dataset has class imbalance (~69% approved).

🛠️ Technologies & Libraries
Python 3

Pandas – Data manipulation

NumPy – Numerical operations

Matplotlib & Seaborn – Visualization

Scikit-learn – Machine learning models & evaluation

Jupyter Notebook / Google Colab

🚀 How to Run
View the Notebook
Open in nbviewer (paste the notebook URL) — recommended for the best rendered view.

Run Locally
Clone the repository:

bash
git clone https://github.com/BismaNisar/loan-approval-prediction.git
cd loan-approval-prediction
Install dependencies:

bash
Open Loan Approval Prediction.ipynb and run all cells.

Make sure Loan_Data.csv is in the same directory as the notebook.

📁 Project Structure
text
loan-approval-prediction/
│
├── Loan Approval Prediction.ipynb   # Main notebook
├── Loan_Data.csv                    # Dataset
├── requirements.txt                 # Dependencies
├── .gitignore                       # Ignored files
└── README.md                        # Project documentation
⚠️ Limitations
Small dataset (614 records) limits generalization.

Class imbalance may bias models toward the majority class.

No cross-validation performed in the initial version.

🔮 Future Improvements
Apply k-fold cross-validation for robust evaluation.

Perform hyperparameter tuning.

Deploy the best model using Flask or Streamlit.

👤 Author
Bisma Nisar , Rameen Khan & Amna Azeem

If you found this project useful, consider giving it a ⭐ on GitHub!


