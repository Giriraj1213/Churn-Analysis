import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def main():
    # Load scaled features
    X = pd.read_csv("/Users/rajsilwal/Documents/GitHub/Churn-Analysis/data/X_scaled.csv")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Save PCA clusters
    pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
    pca_df["Cluster"] = clusters
    pca_df.to_csv("/Users/rajsilwal/Documents/GitHub/Churn-Analysis/data/pca_clusters.csv", index=False)

    # Save clustered dataset
    clustered_df = X.copy()
    clustered_df["Cluster"] = clusters
    clustered_df.to_csv("/Users/rajsilwal/Documents/GitHub/Churn-Analysis/data/clustered_data.csv", index=False)

    print("Clustering complete.")


if __name__ == "__main__":
    main()