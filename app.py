import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Customer Intelligence System",
    layout="wide"
)

# =========================================
# TITLE
# =========================================

st.title("🧠 AI Customer Intelligence System")

st.markdown("""
This system performs:

✅ Customer Segmentation  
✅ Outlier Detection  
✅ Customer Prediction  
✅ Business Insights  
✅ AI Chatbot Analysis  
""")

# =========================================
# FILE UPLOADER
# =========================================

file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)

# =========================================
# MAIN APP
# =========================================

if file is not None:

    try:

        # =========================================
        # LOAD DATA
        # =========================================

        df = pd.read_csv(file)

        df.columns = df.columns.str.strip()

        st.subheader("📌 Dataset Preview")

        st.dataframe(df.head())

        # =========================================
        # REQUIRED COLUMNS CHECK
        # =========================================

        required_columns = [
            'Age',
            'Annual Income',
            'Spending Score'
        ]

        missing_cols = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_cols:

            st.error(f"""
            Missing Required Columns:
            {missing_cols}
            """)

            st.stop()

        # =========================================
        # FEATURE SELECTION
        # =========================================

        features = df[
            ['Age', 'Annual Income', 'Spending Score']
        ]

        # =========================================
        # SCALING
        # =========================================

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(features)

        # =========================================
        # KMEANS
        # =========================================

        kmeans = KMeans(
            n_clusters=5,
            random_state=42,
            n_init=10
        )

        df['Cluster'] = kmeans.fit_predict(
            scaled_data
        )

        # =========================================
        # KMEANS VISUALIZATION
        # =========================================

        st.subheader("🎯 KMeans Customer Segmentation")

        fig1, ax1 = plt.subplots(figsize=(8, 5))

        sns.scatterplot(
            x='Annual Income',
            y='Spending Score',
            hue='Cluster',
            palette='Set1',
            data=df,
            ax=ax1
        )

        st.pyplot(fig1)

        # =========================================
        # DBSCAN
        # =========================================

        dbscan = DBSCAN(
            eps=0.8,
            min_samples=5
        )

        df['DBSCAN_Cluster'] = dbscan.fit_predict(
            scaled_data
        )

        # =========================================
        # DBSCAN VISUALIZATION
        # =========================================

        st.subheader("🔍 DBSCAN Outlier Detection")

        fig2, ax2 = plt.subplots(figsize=(8, 5))

        sns.scatterplot(
            x='Annual Income',
            y='Spending Score',
            hue='DBSCAN_Cluster',
            palette='Set2',
            data=df,
            ax=ax2
        )

        st.pyplot(fig2)

        # =========================================
        # OUTLIERS
        # =========================================

        outliers = df[
            df['DBSCAN_Cluster'] == -1
        ]

        num_outliers = len(outliers)

        st.subheader("🚨 Outlier Detection")

        st.write(
            f"Total Outliers Detected: {num_outliers}"
        )

        st.dataframe(outliers)

        # =========================================
        # PCA
        # =========================================

        pca = PCA(n_components=2)

        pca_data = pca.fit_transform(
            scaled_data
        )

        pca_df = pd.DataFrame(
            pca_data,
            columns=['PC1', 'PC2']
        )

        pca_df['Cluster'] = df['Cluster']

        st.subheader("📉 PCA Visualization")

        fig3, ax3 = plt.subplots(figsize=(8, 5))

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

        st.subheader("📊 Model Evaluation")

        st.write(
            f"Silhouette Score: {score:.2f}"
        )

        # =========================================
        # SVM
        # =========================================

        df['Customer_Type'] = df['Cluster'].apply(
            lambda x: 1 if x in [0, 3] else 0
        )

        X = scaled_data

        y = df['Customer_Type']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        svm_model = SVC(kernel='rbf')

        svm_model.fit(
            X_train,
            y_train
        )

        predictions = svm_model.predict(
            X_test
        )

        report = classification_report(
            y_test,
            predictions
        )

        st.subheader("🤖 SVM Classification Report")

        st.text(report)

        # =========================================
        # CLUSTER SUMMARY
        # =========================================

        st.subheader("📌 Cluster Summary")

        summary = df.groupby('Cluster')[
            ['Age', 'Annual Income', 'Spending Score']
        ].mean()

        st.dataframe(summary)

        # =========================================
        # BUSINESS INSIGHTS
        # =========================================

        st.subheader("💡 Business Insights")

        st.markdown("""
        ✅ Cluster 0 → Premium Customers  
        ✅ Cluster 1 → Low Value Customers  
        ✅ Cluster 2 → Average Customers  
        ✅ Cluster 3 → Young High Spenders  
        ✅ Cluster 4 → Target Customers  
        """)

        # =========================================
        # AI CHATBOT
        # =========================================

        st.subheader("🤖 AI Smart Chatbot")

        user_question = st.text_input(
            "Ask questions about dataset/project:"
        )

        if user_question:

            question = user_question.lower()

            # ROWS

            if "rows" in question:

                st.success(
                    f"Total Rows: {df.shape[0]}"
                )

            # COLUMNS

            elif "columns" in question:

                st.success(
                    f"Total Columns: {df.shape[1]}"
                )

                st.write(list(df.columns))

            # NULL VALUES

            elif "null" in question or "missing" in question:

                st.write(df.isnull().sum())

            # DUPLICATES

            elif "duplicate" in question:

                duplicates = df.duplicated().sum()

                st.success(
                    f"Duplicate Rows: {duplicates}"
                )

            # OUTLIERS

            elif "outlier" in question:

                st.success(
                    f"Total Outliers: {num_outliers}"
                )

                st.dataframe(outliers)

            # SHAPE

            elif "shape" in question:

                st.success(
                    f"Dataset Shape: {df.shape}"
                )

            # STATISTICS

            elif "statistics" in question or "summary" in question:

                st.dataframe(df.describe())

            # KMEANS

            elif "kmeans" in question:

                st.success("""
                KMeans is used for customer segmentation.
                """)

            # DBSCAN

            elif "dbscan" in question:

                st.success("""
                DBSCAN is used for outlier detection.
                """)

            # PCA

            elif "pca" in question:

                st.success("""
                PCA reduces dimensions for visualization.
                """)

            # SVM

            elif "svm" in question:

                st.success("""
                SVM predicts customer category.
                """)

            # SILHOUETTE SCORE

            elif "silhouette" in question:

                st.success(
                    f"Silhouette Score: {score:.2f}"
                )

            # CLUSTERS

            elif "cluster" in question:

                st.write(
                    df['Cluster'].value_counts()
                )

            # DEFAULT

            else:

                st.warning("""
                Ask questions like:

                - Total rows
                - Columns
                - Null values
                - Duplicate rows
                - Outliers
                - KMeans
                - DBSCAN
                - PCA
                - SVM
                - Statistics
                - Cluster summary
                """)

    except Exception as e:

        st.error("Error loading dataset or processing app.")

        st.exception(e)

else:

    st.info("Please upload a CSV file to continue.")
