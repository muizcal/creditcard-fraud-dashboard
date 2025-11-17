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
   git clone https://github.com/muizcal/creditcard-fraud-dashboard
   cd https://github.com/muizcal/creditcard-fraud-dashboard </PRE>

2. Install dependencies:

    <pre>pip install -r requirements.txt </pre>

3. Run the app:

    <pre>streamlit run app.py</pre>

4. Open the app in your browser at the link provided by Streamlit

# Sidebar Filters

Transaction Type: All / Normal / Fraud

Transaction Amount Range: Filter by transaction value

# Machine Learning Models

-Random Forest Classifier

-Logistic Regression

Both models are trained on a sample of the dataset and evaluated with:

-Accuracy

-Classification Report

-Confusion Matrix

-Feature Importance (for Random Forest)


# Visualizations

Fraud vs Normal Transactions (count plot)

Transaction Amount Distribution (histogram)

Correlation Heatmap (feature correlation)

# Predictions

Predict fraud on first 20 transactions from test set.

Download prediction results as CSV for further analysis.

# Requirements

Python 3.9+

Streamlit

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Install dependencies with pip install -r requirements.txt

# Links

Original dataset on Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Streamlit Dashboard: [Add your deployed app link here](https://creditcard-fraud-dashboard-w8ekr7mhacmnlz2k83appvq.streamlit.app/)
