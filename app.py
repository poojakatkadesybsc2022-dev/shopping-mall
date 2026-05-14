# =========================================
# AI SMART CHATBOT
# =========================================

st.subheader("🤖 AI Smart Dataset Chatbot")

user_question = st.text_input(
    "Ask questions about your dataset/project:"
)

if user_question:

    question = user_question.lower()

    # =========================================
    # BASIC DATASET INFO
    # =========================================

    if "rows" in question or "records" in question:

        st.success(f"""
        Total Rows in Dataset:
        {df.shape[0]}
        """)

    elif "columns" in question:

        st.success(f"""
        Total Columns in Dataset:
        {df.shape[1]}

        Column Names:
        {list(df.columns)}
        """)

    elif "shape" in question:

        st.success(f"""
        Dataset Shape:
        {df.shape}
        """)

    elif "null" in question or "missing" in question:

        null_values = df.isnull().sum()

        st.success("Null Values In Dataset")

        st.write(null_values)

    elif "duplicate" in question:

        duplicates = df.duplicated().sum()

        st.success(f"""
        Total Duplicate Rows:
        {duplicates}
        """)

    # =========================================
    # OUTLIERS
    # =========================================

    elif "outlier" in question:

        st.success(f"""
        Total Outliers Detected:
        {num_outliers}

        DBSCAN detected unusual customers
        with abnormal spending behavior.
        """)

        st.write(outliers)

    # =========================================
    # STATISTICS
    # =========================================

    elif "statistics" in question or "summary" in question:

        st.success("Dataset Statistical Summary")

        st.write(df.describe())

    elif "average income" in question:

        avg_income = df['Annual Income'].mean()

        st.success(f"""
        Average Annual Income:
        {avg_income:.2f}
        """)

    elif "highest spending" in question:

        highest = df['Spending Score'].max()

        st.success(f"""
        Highest Spending Score:
        {highest}
        """)

    elif "lowest spending" in question:

        lowest = df['Spending Score'].min()

        st.success(f"""
        Lowest Spending Score:
        {lowest}
        """)

    elif "average age" in question:

        avg_age = df['Age'].mean()

        st.success(f"""
        Average Customer Age:
        {avg_age:.2f}
        """)

    # =========================================
    # CLUSTER QUESTIONS
    # =========================================

    elif "cluster" in question:

        st.success("""
        Cluster Insights:

        Cluster 0 → Premium Customers
        Cluster 1 → Low Value Customers
        Cluster 2 → Average Customers
        Cluster 3 → Young High Spenders
        Cluster 4 → Target Customers
        """)

        cluster_count = df['Cluster'].value_counts()

        st.write(cluster_count)

    elif "premium customers" in question:

        premium = df[df['Customer_Type'] == 1]

        st.success(f"""
        Total Premium Customers:
        {len(premium)}
        """)

        st.write(premium.head())

    # =========================================
    # ML ALGORITHMS
    # =========================================

    elif "kmeans" in question:

        st.success("""
        KMeans Clustering Algorithm:

        - Type → Unsupervised ML
        - Purpose → Customer Segmentation
        - Clusters Created → 5
        """)

    elif "dbscan" in question:

        st.success("""
        DBSCAN Algorithm:

        - Type → Density Based Clustering
        - Purpose → Outlier Detection
        - Detects abnormal customer behavior
        """)

    elif "svm" in question:

        st.success("""
        SVM Algorithm:

        - Type → Supervised ML
        - Kernel Used → RBF
        - Purpose → Customer Prediction
        """)

    elif "pca" in question:

        st.success("""
        PCA reduces dimensions for visualization.

        It converts multiple features into
        2D graphs for easier understanding.
        """)

    elif "silhouette" in question:

        st.success(f"""
        Silhouette Score:
        {score:.2f}

        Higher score means better clustering.
        """)

    elif "algorithms" in question:

        st.success("""
        Algorithms Used:

        ✅ KMeans
        ✅ DBSCAN
        ✅ PCA
        ✅ SVM
        """)

    # =========================================
    # COLUMN SPECIFIC QUESTIONS
    # =========================================

    elif "age" in question:

        st.success("Age Column Information")

        st.write(df['Age'].describe())

    elif "income" in question:

        st.success("Annual Income Information")

        st.write(df['Annual Income'].describe())

    elif "spending" in question:

        st.success("Spending Score Information")

        st.write(df['Spending Score'].describe())

    # =========================================
    # DEFAULT RESPONSE
    # =========================================

    else:

        st.warning("""
        I can answer questions related to:

        ✅ Dataset
        ✅ Null Values
        ✅ Outliers
        ✅ KMeans
        ✅ DBSCAN
        ✅ PCA
        ✅ SVM
        ✅ Statistics
        ✅ Clusters
        ✅ Spending Score
        ✅ Income
        ✅ Age
        ✅ Premium Customers
        """)
