
Project: Customer Churn Analysis (DataCore Solution). Goal: understand why customers
leave (churn) and predict who is likely to churn. Target = Churn (1 = churned, 0 = retained).
Numeric features: Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend,
Last Interaction. Categorical: Gender, Subscription Type, Contract Length.
Files: Churn.ipynb (EDA, encoding, scaling, Logistic Regression), Statistical_Testing.ipynb
(hypothesis tests), streamlit_dashboard.py (dashboard), chatbot.py (this assistant).
Dashboard pages: Overview & EDA, Statistical Tests, Model Fitting, Predict Churn.
Tests: Welch's t-test (numeric vs churn), Chi-square + Cramer's V (categorical vs churn),
one-way ANOVA + Levene's test; p-values corrected with Benjamini-Hochberg (FDR).
Models: Logistic Regression and Random Forest (150 trees, max depth 12) in a scikit-learn
Pipeline (StandardScaler + OneHotEncoder), 80/20 stratified split, random_state=42.
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix.

LIVE RESULTS: 64,374 customers, 0 missing values, overall churn rate 47.4%.
Mean of each numeric feature (0 = retained, 1 = churned):
Churn                  0       1
Age                41.13   42.90
Tenure             28.83   35.52
Usage Frequency    16.04   14.01
Support Calls       4.50    6.40
Payment Delay      12.45   22.33
Total Spend       560.54  519.34
Last Interaction   15.52   15.47
Welch t-test (churned vs retained):
  Age: t=16.1, p=2.67e-58
  Tenure: t=50.8, p=0
  Usage Frequency: t=-29.3, p=4.75e-187
  Support Calls: t=81.8, p=0
  Payment Delay: t=173.0, p=0
  Total Spend: t=-20.1, p=2.62e-89
  Last Interaction: t=-0.7, p=0.475
Chi-square (categorical vs churn) and churn rate per category:
  Gender: p=0, Cramer's V=0.165; churn rate: Female=55.0%, Male=38.6%
  Subscription Type: p=0.00108, Cramer's V=0.015; churn rate: Basic=48.3%, Premium=46.5%, Standard=47.3%
  Contract Length: p=7.34e-58, Cramer's V=0.064; churn rate: Annual=46.2%, Monthly=51.6%, Quarterly=44.0%
Logistic Regression (test 20%): Accuracy=0.827, Precision=0.814, Recall=0.823, F1=0.819, ROC-AUC=0.903; confusion matrix [[TN,FP],[FN,TP]]=[[5628, 1148], [1077, 5022]]
Random Forest (test 20%): Accuracy=0.998, Precision=0.999, Recall=0.997, F1=0.998, ROC-AUC=1.000; confusion matrix [[TN,FP],[FN,TP]]=[[6769, 7], [20, 6079]]
Random Forest top features: num__Payment Delay=0.456, num__Support Calls=0.169, num__Tenure=0.103, num__Usage Frequency=0.080, cat__Gender_Male=0.072, num__Total Spend=0.041, num__Age=0.038, cat__Contract Length_Monthly=0.026