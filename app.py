# =========================================
# CHATBOT
# =========================================

try:

    st.subheader("🤖 AI Smart Dataset Chatbot")

    user_question = st.text_input(
        "Ask questions about your dataset/project:"
    )

    if user_question:

        question = user_question.lower()

        # =========================================
        # ROWS
        # =========================================

        if "rows" in question:

            st.success(f"Total Rows: {df.shape[0]}")

        # =========================================
        # COLUMNS
        # =========================================

        elif "columns" in question:

            st.success(f"Total Columns: {df.shape[1]}")

            st.write(list(df.columns))

        # =========================================
        # NULL VALUES
        # =========================================

        elif "null" in question or "missing" in question:

            st.success("Null Values")

            st.write(df.isnull().sum())

        # =========================================
        # DUPLICATES
        # =========================================

        elif "duplicate" in question:

            duplicates = df.duplicated().sum()

            st.success(f"Duplicate Rows: {duplicates}")

        # =========================================
        # OUTLIERS
        # =========================================

        elif "outlier" in question:

            st.success(f"Total Outliers: {num_outliers}")

            st.write(outliers)

        # =========================================
        # SHAPE
        # =========================================

        elif "shape" in question:

            st.success(f"Dataset Shape: {df.shape}")

        # =========================================
        # STATISTICS
        # =========================================

        elif "statistics" in question or "summary" in question:

            st.write(df.describe())

        # =========================================
        # AGE
        # =========================================

        elif "age" in question:

            st.write(df['Age'].describe())

        # =========================================
        # INCOME
        # =========================================

        elif "income" in question:

            st.write(df['Annual Income'].describe())

        # =========================================
        # SPENDING SCORE
        # =========================================

        elif "spending" in question:

            st.write(df['Spending Score'].describe())

        # =========================================
        # CLUSTERS
        # =========================================

        elif "cluster" in question:

            st.success("""
            Cluster 0 → Premium Customers
            Cluster 1 → Low Value Customers
            Cluster 2 → Average Customers
            Cluster 3 → Young High Spenders
            Cluster 4 → Target Customers
            """)

            st.write(df['Cluster'].value_counts())

        # =========================================
        # KMEANS
        # =========================================

        elif "kmeans" in question:

            st.success("""
            KMeans is used for customer segmentation.
            """)

        # =========================================
        # DBSCAN
        # =========================================

        elif "dbscan" in question:

            st.success("""
            DBSCAN is used for outlier detection.
            """)

        # =========================================
        # PCA
        # =========================================

        elif "pca" in question:

            st.success("""
            PCA reduces dimensions for visualization.
            """)

        # =========================================
        # SVM
        # =========================================

        elif "svm" in question:

            st.success("""
            SVM predicts customer category.
            """)

        # =========================================
        # ALGORITHMS
        # =========================================

        elif "algorithms" in question:

            st.success("""
            Algorithms Used:

            ✅ KMeans
            ✅ DBSCAN
            ✅ PCA
            ✅ SVM
            """)

        # =========================================
        # DEFAULT
        # =========================================

        else:

            st.warning("""
            Ask questions like:

            - Total rows
            - Null values
            - Outliers
            - KMeans
            - DBSCAN
            - SVM
            - PCA
            - Cluster info
            - Statistics
            """)

except Exception as e:

    st.error("Chatbot could not load properly.")
