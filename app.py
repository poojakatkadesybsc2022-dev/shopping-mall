import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# =========================================
# TITLE
# =========================================

st.title("🧠 AI Customer Intelligence System")

# =========================================
# FILE UPLOAD
# =========================================

file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if file:

    # =========================================
    # LOAD DATA
    # =========================================

    df = pd.read_csv(file)

    st.subheader("📊 Raw Dataset")
    st.write(df.head())

    # =========================================
    # FEATURE SELECTION
    # =========================================

    features = df[['Age', 'Annual_Income_k$', 'Spending_Score']]

    # =========================================
    # FEATURE SCALING
    # =========================================

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(features)

    # =========================================
    # KMEANS CLUSTERING
    # =========================================

    kmeans = KMeans(n_clusters=5, random_state=42)

    df['Cluster'] = kmeans.fit_predict(scaled_data)

    st.subheader("🎯 KMeans Customer Segmentation")

    fig1, ax1 = plt.subplots()

    sns.scatterplot(
        x='Annual_Income_k$',
        y='Spending_Score',
        hue='Cluster',
        palette='Set1',
        data=df,
        ax=ax1
    )

    st.pyplot(fig1)

    # =========================================
    # DBSCAN CLUSTERING
    # =========================================

    dbscan = DBSCAN(eps=0.8, min_samples=5)

    df['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_data)

    st.subheader("🔍 DBSCAN Outlier Detection")

    fig2, ax2 = plt.subplots()

    sns.scatterplot(
        x='Annual_Income_k$',
        y='Spending_Score',
        hue='DBSCAN_Cluster',
        palette='Set2',
        data=df,
        ax=ax2
    )

    st.pyplot(fig2)

    st.info("""
    DBSCAN detects:
    - abnormal customers
    - rare spending behavior
    - outliers
    """)

    # =========================================
    # PCA VISUALIZATION
    # =========================================

    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(
        pca_data,
        columns=['PC1', 'PC2']
    )

    pca_df['Cluster'] = df['Cluster']

    st.subheader("📉 PCA Visualization")

    fig3, ax3 = plt.subplots()

    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='Cluster',
        palette='viridis',
        data=pca_df,
        ax=ax3
    )

    st.pyplot(fig3)

    # =========================================
    # SILHOUETTE SCORE
    # =========================================

    score = silhouette_score(
        scaled_data,
        df['Cluster']
    )

    st.subheader("📊 KMeans Evaluation")

    st.write(f"Silhouette Score: {score:.2f}")

    # =========================================
    # CREATE LABELS FOR SVM
    # =========================================

    df['Customer_Type'] = df['Cluster'].apply(
        lambda x: 1 if x in [0, 3] else 0
    )

    # =========================================
    # TRAIN TEST SPLIT
    # =========================================

    X = scaled_data

    y = df['Customer_Type']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =========================================
    # SVM MODEL
    # =========================================

    svm_model = SVC(kernel='rbf')

    svm_model.fit(X_train, y_train)

    predictions = svm_model.predict(X_test)

    # =========================================
    # CLASSIFICATION REPORT
    # =========================================

    st.subheader("🤖 SVM Classification Report")

    report = classification_report(
        y_test,
        predictions
    )

    st.text(report)

    st.success("""
    SVM predicts whether a customer is:
    - Premium Customer
    - Normal Customer
    """)

    # =========================================
    # CLUSTER SUMMARY
    # =========================================

    st.subheader("📌 Cluster Summary")

    summary = df.groupby('Cluster')[
        ['Age', 'Annual_Income_k$', 'Spending_Score']
    ].mean()

    st.write(summary)

    # =========================================
    # BUSINESS INSIGHTS
    # =========================================

    st.subheader("💡 Business Insights")

    st.markdown("""
    - Cluster 0 → Premium Customers  
    - Cluster 1 → Low Value Customers  
    - Cluster 2 → Average Customers  
    - Cluster 3 → Young High Spenders  
    - Cluster 4 → Target Customers  

    ### Algorithms Used

    ✅ KMeans → Customer Segmentation  
    ✅ DBSCAN → Outlier Detection  
    ✅ SVM → Customer Type Prediction  
    """)
