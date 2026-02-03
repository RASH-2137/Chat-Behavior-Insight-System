"""
Clustering module.
Scales features and performs KMeans clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Tuple


def prepare_features_for_clustering(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale features for clustering.
    
    Args:
        features_df: DataFrame with user features
        
    Returns:
        Tuple of (scaled_features_df, scaler)
    """
    # These are the features we use for clustering
    feature_columns = [
        'avg_length', 'median_length',
        'messages_per_day', 'avg_response_time_hours', 'night_activity_ratio',
        'avg_exclamations', 'avg_questions', 'avg_emojis', 'uppercase_ratio',
        'link_sharing_ratio', 'initiation_ratio'
    ]
    
    # Only use columns that actually exist
    available_columns = [col for col in feature_columns if col in features_df.columns]
    
    if not available_columns:
        raise ValueError("No valid feature columns found for clustering")
    
    # Get the feature matrix
    X = features_df[available_columns].values
    
    # Scale everything (important for KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Put it back in a DataFrame
    scaled_df = pd.DataFrame(X_scaled, columns=available_columns, index=features_df.index)
    
    return scaled_df, scaler


def perform_clustering(scaled_features: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """
    Perform KMeans clustering on scaled features.
    
    Args:
        scaled_features: Scaled feature DataFrame
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (cluster_labels, kmeans_model)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features.values)
    
    return cluster_labels, kmeans


def assign_clusters(features_df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Assign cluster labels to users.
    
    Args:
        features_df: DataFrame with user features
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with cluster labels added
    """
    # Scale the features first
    scaled_features, scaler = prepare_features_for_clustering(features_df)
    
    # Run KMeans
    cluster_labels, kmeans = perform_clustering(scaled_features, n_clusters, random_state)
    
    # Add cluster assignments to the dataframe
    features_df = features_df.copy()
    features_df['cluster'] = cluster_labels
    
    return features_df
