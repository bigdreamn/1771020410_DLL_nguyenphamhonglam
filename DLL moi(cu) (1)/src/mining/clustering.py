import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def run_clustering(df_scaled, n_clusters=3, random_state=42):
    """
    Run K-Means clustering on scaled employee data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Add cluster label to data
    df_clustered = df_scaled.copy()
    df_clustered['cluster'] = clusters
    
    # Calculate silhouette score
    score = silhouette_score(df_scaled, clusters)
    
    return df_clustered, kmeans, score

def profile_clusters(df_original, cluster_labels):
    """
    Profile clusters based on the original (unscaled) features.
    """
    df_profile = df_original.copy()
    df_profile['cluster'] = cluster_labels
    
    # Group by cluster and calculate mean for numeric features
    # and mode for categorical if possible
    profile = df_profile.groupby('cluster').mean(numeric_only=True)
    
    # Also include the count of employees in each cluster
    profile['count'] = df_profile['cluster'].value_counts()
    
    return profile
