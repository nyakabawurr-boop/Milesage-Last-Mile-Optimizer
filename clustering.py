"""
Clustering utilities for multi-depot and region-based routing.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from sklearn.cluster import KMeans


def apply_kmeans_clustering(df: pd.DataFrame, n_clusters: int, seed: int = 42) -> pd.DataFrame:
    """
    Apply K-means clustering on lat/lon coordinates.
    
    Args:
        df: DataFrame with 'lat' and 'lon' columns
        n_clusters: Number of clusters (K)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with added 'cluster_id' column
    """
    df_clustered = df.copy()
    
    # Extract coordinates
    coords = df_clustered[['lat', 'lon']].values
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    
    df_clustered['cluster_id'] = cluster_labels
    
    # Find depot for each cluster (closest to centroid)
    for cluster_id in range(n_clusters):
        cluster_mask = df_clustered['cluster_id'] == cluster_id
        cluster_coords = coords[cluster_mask]
        centroid = kmeans.cluster_centers_[cluster_id]
        
        if len(cluster_coords) > 0:
            # Calculate distances to centroid
            distances = np.sqrt(
                np.sum((cluster_coords - centroid) ** 2, axis=1)
            )
            closest_idx = np.argmin(distances)
            
            # Get the actual index in the dataframe
            cluster_indices = df_clustered.index[cluster_mask].tolist()
            closest_df_idx = cluster_indices[closest_idx]
            
            # Mark as depot if not already
            df_clustered.loc[closest_df_idx, 'is_depot'] = True
    
    return df_clustered


def apply_column_based_clustering(df: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    """
    Create clusters based on an existing column (e.g., region, depot_id).
    
    Args:
        df: DataFrame
        cluster_column: Column name to use for clustering
        
    Returns:
        DataFrame with added 'cluster_id' column (numeric mapping of cluster_column values)
    """
    df_clustered = df.copy()
    
    # Map unique values to numeric cluster IDs
    unique_values = df_clustered[cluster_column].unique()
    value_to_cluster = {val: idx for idx, val in enumerate(sorted(unique_values))}
    
    df_clustered['cluster_id'] = df_clustered[cluster_column].map(value_to_cluster)
    
    # For each cluster, mark the first row as depot (or keep existing depot markers)
    for cluster_id in df_clustered['cluster_id'].unique():
        cluster_mask = df_clustered['cluster_id'] == cluster_id
        
        # If no depot exists in cluster, mark first row as depot
        if not df_clustered.loc[cluster_mask, 'is_depot'].any():
            first_idx = df_clustered.index[cluster_mask][0]
            df_clustered.loc[first_idx, 'is_depot'] = True
    
    return df_clustered


def split_dataframe_by_cluster(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Split a DataFrame into separate DataFrames by cluster_id.
    
    Args:
        df: DataFrame with 'cluster_id' column
        
    Returns:
        Dictionary mapping cluster_id to DataFrame
    """
    clusters = {}
    
    if 'cluster_id' not in df.columns:
        # No clustering applied, return single "cluster"
        clusters[0] = df.copy()
        return clusters
    
    for cluster_id in sorted(df['cluster_id'].unique()):
        clusters[cluster_id] = df[df['cluster_id'] == cluster_id].copy().reset_index(drop=True)
    
    return clusters


def aggregate_cluster_solutions(cluster_solutions: Dict[int, Dict]) -> Dict:
    """
    Aggregate solutions from multiple clusters into a single solution.
    
    Args:
        cluster_solutions: Dict mapping cluster_id to solution dict
        
    Returns:
        Aggregated solution dictionary
    """
    total_distance = 0
    total_time = 0
    total_vehicles = 0
    all_route_details = []
    
    for cluster_id, solution in cluster_solutions.items():
        if 'error' in solution:
            continue
        
        total_distance += solution.get('total_distance', 0)
        total_time += solution.get('total_time', 0)
        total_vehicles += solution.get('n_vehicles_used', 0)
        
        # Prefix vehicle IDs with cluster ID
        for route_detail in solution.get('route_details', []):
            route_copy = route_detail.copy()
            original_vehicle_id = route_copy.get('vehicle_id', 0)
            route_copy['vehicle_id'] = f"C{cluster_id}-V{original_vehicle_id}"
            route_copy['cluster_id'] = cluster_id
            all_route_details.append(route_copy)
    
    return {
        'total_distance': total_distance,
        'total_time': total_time,
        'n_vehicles_used': total_vehicles,
        'route_details': all_route_details,
        'n_clusters': len(cluster_solutions),
    }

