import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv("labeled_transactions_5000.csv")
    df = df.drop(columns=["Transaction_ID", "User_Name", "Transaction_Date"])
    df["Transaction_Value_Label"] = df["Transaction_Value_Label"].map({"High": 1, "Low": 0})
    df = pd.get_dummies(df, drop_first=True)
    return df

@st.cache_data
def load_transformers():
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return scaler, pca

def preprocess(X, scaler, pca):
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca

def main():
    st.title("E-Commerce Transaction Classifier")

    df = load_data()
    X = df.drop("Transaction_Value_Label", axis=1)
    y = df["Transaction_Value_Label"]
    scaler, pca = load_transformers()

    X_processed = preprocess(X, scaler, pca)

    rf = joblib.load("rf_model.pkl")
    knn = joblib.load("knn_model.pkl")

    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a page", [
        "Home", "Accuracy Comparison", "Confusion Matrices",
        "Classification Reports", "Predict New Transaction"
    ])

    if options == "Home":
        st.subheader("Welcome to the E-Commerce Transaction Classifier!")
        st.markdown("Compare models and make predictions using your own inputs.")

    elif options == "Accuracy Comparison":
        st.subheader("Accuracy Comparison")
        rf_pred = rf.predict(X_processed)
        knn_pred = knn.predict(X_processed)
        rf_acc = (rf_pred == y).mean()
        knn_acc = (knn_pred == y).mean()
        st.write(f"**Random Forest Accuracy**: `{rf_acc:.2f}`")
        st.write(f"**KNN Accuracy**: `{knn_acc:.2f}`")
        fig, ax = plt.subplots()
        ax.bar(["Random Forest", "KNN"], [rf_acc, knn_acc], color=["skyblue", "lightgreen"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)

    elif options == "Confusion Matrices":
        st.subheader("Confusion Matrix - Random Forest")
        rf_pred = rf.predict(X_processed)
        fig_rf, ax_rf = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y, rf_pred, ax=ax_rf)
        st.pyplot(fig_rf)
        st.subheader("Confusion Matrix - KNN")
        knn_pred = knn.predict(X_processed)
        fig_knn, ax_knn = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y, knn_pred, ax=ax_knn)
        st.pyplot(fig_knn)

    elif options == "Classification Reports":
        st.subheader("Classification Report - Random Forest")
        rf_pred = rf.predict(X_processed)
        st.text(classification_report(y, rf_pred))
        st.subheader("Classification Report - KNN")
        knn_pred = knn.predict(X_processed)
        st.text(classification_report(y, knn_pred))

    elif options == "Predict New Transaction":
        st.subheader("📥 Enter Transaction Details")

        input_dict = {}

        # Fixed category/prefix matching
        country_cols = [col for col in X.columns if col.startswith("Country_")]
        category_cols = [col for col in X.columns if col.startswith("Product_Category_")]
        payment_cols = [col for col in X.columns if col.startswith("Payment_Method_")]

        country_options = [col.split("Country_")[1] for col in country_cols]
        category_options = [col.split("Product_Category_")[1] for col in category_cols]
        payment_options = [col.split("Payment_Method_")[1] for col in payment_cols]

        selected_country = st.selectbox("Country", country_options)
        selected_category = st.selectbox("Product Category", category_options)
        selected_payment = st.selectbox("Payment Method", payment_options)

        for col in country_cols:
            input_dict[col] = 1 if selected_country in col else 0
        for col in category_cols:
            input_dict[col] = 1 if selected_category in col else 0
        for col in payment_cols:
            input_dict[col] = 1 if selected_payment in col else 0

        for col in X.columns:
            if col in input_dict:
                continue
            elif X[col].nunique() == 2 and set(X[col].unique()) <= {0, 1}:
                input_dict[col] = st.selectbox(f"{col}", [0, 1])
            else:
                input_dict[col] = st.number_input(f"{col}", value=float(X[col].mean()), step=1.0)

        # Ensure input_df columns match training columns
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        if st.button("Predict"):
            input_transformed = preprocess(input_df, scaler, pca)
            rf_pred = rf.predict(input_transformed)[0]
            knn_pred = knn.predict(input_transformed)[0]

            st.success(f"Random Forest Prediction: {'High' if rf_pred == 1 else 'Low'}")
            st.success(f"KNN Prediction: {'High' if knn_pred == 1 else 'Low'}")

if __name__ == "__main__":
    main()
