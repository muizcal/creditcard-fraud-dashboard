# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("""
Analyze credit card transactions to detect fraudulent behavior.  
Filters, EDA charts, ML models, and predictions included.
""")


@st.cache_data
def load_data():
    df = pd.read_csv("creditcard_sample.csv")
    return df

df = load_data()


st.sidebar.header("Filters")
transaction_type = st.sidebar.selectbox("Transaction Type", ["All", "Normal", "Fraud"])
amount_range = st.sidebar.slider(
    "Transaction Amount Range", 
    float(df['Amount'].min()), float(df['Amount'].max()),
    (float(df['Amount'].min()), float(df['Amount'].max()))
)

filtered_df = df[
    (df['Amount'] >= amount_range[0]) & 
    (df['Amount'] <= amount_range[1])
]

if transaction_type == "Normal":
    filtered_df = filtered_df[filtered_df['Class'] == 0]
elif transaction_type == "Fraud":
    filtered_df = filtered_df[filtered_df['Class'] == 1]


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Preview", "EDA Charts", "Correlation Heatmap", "ML Models", "Predictions"
])


with tab1:
    st.subheader("Filtered Data")
    st.dataframe(filtered_df.head(20))
    st.markdown("### Quick Statistics")
    st.write(filtered_df.describe())


with tab2:
    st.subheader("Fraud vs Normal Transactions")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=filtered_df, palette='coolwarm', ax=ax)
    ax.set_xticklabels(["Normal","Fraud"])
    st.pyplot(fig)

    st.subheader("Transaction Amount Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df['Amount'], bins=50, kde=True, ax=ax2)
    st.pyplot(fig2)


with tab3:
    st.subheader("Correlation Heatmap")
    corr = filtered_df.corr()
    fig3, ax3 = plt.subplots(figsize=(12,8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)


with tab4:
    st.subheader("Train ML Models")
    st.markdown("Random Forest and Logistic Regression on a sample of 5000 rows.")

    sample_df = filtered_df.sample(n=min(5000, len(filtered_df)), random_state=42)
    X = sample_df.drop(['Class'], axis=1)
    y = sample_df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    st.write("**Random Forest Accuracy:**", accuracy_score(y_test, y_pred_rf))
    st.text(classification_report(y_test, y_pred_rf))

    st.subheader("Random Forest Confusion Matrix")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fig_cm_rf, ax_cm_rf = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax_cm_rf)
    st.pyplot(fig_cm_rf)

    st.subheader("Feature Importance (Random Forest)")
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_fi, ax_fi = plt.subplots(figsize=(10,6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis', ax=ax_fi)
    st.pyplot(fig_fi)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    st.write("**Logistic Regression Accuracy:**", accuracy_score(y_test, y_pred_lr))
    st.text(classification_report(y_test, y_pred_lr))

    st.subheader("Logistic Regression Confusion Matrix")
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    fig_cm_lr, ax_cm_lr = plt.subplots()
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds', ax=ax_cm_lr)
    st.pyplot(fig_cm_lr)


with tab5:
    st.subheader("Predict Fraud on Sample Transactions")
    demo_df = X_test[:20]
    rf_pred_demo = rf.predict(demo_df)
    lr_pred_demo = lr.predict(demo_df)

    result_df = pd.DataFrame({
        "Transaction Index": demo_df.index,
        "Random Forest Prediction": rf_pred_demo,
        "Logistic Regression Prediction": lr_pred_demo
    })
    st.dataframe(result_df)

    st.download_button(
        label="Download Predictions as CSV",
        data=result_df.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )
