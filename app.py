import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.title("🛍️ Shopping Mall Customer Segmentation")

# Upload file
file = st.file_uploader("Upload Mall Customer CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Raw Data")
    st.write(df.head())

    # -------------------------------
    # Preprocessing
    # -------------------------------
    features = df[['Age', 'Annual Income', 'Spending Score']]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    # -------------------------------
    # KMeans Clustering
    # -------------------------------
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📈 Customer Segments")

    fig, ax = plt.subplots()
    sns.scatterplot(
        x='Annual Income',
        y='Spending Score',
        hue='Cluster',
        palette='Set1',
        data=df,
        ax=ax
    )
    st.pyplot(fig)

    # -------------------------------
    # PCA Visualization
    # -------------------------------
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df['Cluster']

    st.subheader("🔍 PCA Visualization")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='Cluster',
        palette='Set2',
        data=pca_df,
        ax=ax2
    )
    st.pyplot(fig2)

    # -------------------------------
    # Silhouette Score
    # -------------------------------
    score = silhouette_score(scaled_data, df['Cluster'])

    st.subheader("📊 Model Evaluation")
    st.write(f"Silhouette Score: {score:.2f}")

    # -------------------------------
    # Cluster Insights
    # -------------------------------
    st.subheader("📌 Cluster Summary")

    summary = df.groupby('Cluster')[['Age', 'Annual Income', 'Spending Score']].mean()
    st.write(summary)

    st.subheader("💡 Business Insights")

    st.markdown("""
    - Cluster 0 → High income & high spending → Premium customers  
    - Cluster 1 → Low income & low spending → Low value customers  
    - Cluster 2 → High income & low spending → Target customers  
    - Cluster 3 → Young high spenders → Trendy buyers  
    - Cluster 4 → Average customers  
    """)