# 💳 Credit Card Fraud Detection Dashboard

This Streamlit app analyzes credit card transactions to detect fraudulent behavior using exploratory data analysis (EDA) and machine learning models. The app includes interactive filters, visualizations, ML models, and prediction capabilities.  



##  Features

- **Data Preview:** View filtered transaction data with quick statistics.  
- **EDA Charts:** Visualize fraud vs normal transactions, transaction amount distributions, and more.  
- **Correlation Heatmap:** Explore correlations between transaction features.  
- **ML Models:** Train **Random Forest** and **Logistic Regression** models.  
- **Feature Importance:** Identify which features most impact predictions.  
- **Predictions:** Predict fraud on sample transactions and download results as CSV.  



##  Dataset

- **Original dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Reduced dataset for app (smaller file):** `creditcard_sample.csv`  
  - Contains all fraud transactions + 10x normal transactions.  
  - Used to reduce file size and improve performance on Streamlit Cloud.  



##  How to Run

1. Clone the repository:
   <PRE>bash
   git clone [<your-repo-url>](https://github.com/muizcal/creditcard-fraud-dashboard)
   cd https://github.com/muizcal/creditcard-fraud-dashboard </PRE>
