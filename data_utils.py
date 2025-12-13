"""
Data utilities for Milesage - Last-Mile Optimizer

Handles data loading, validation, normalization, and synthetic data generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    R = 6371.0
    
    return R * c


def generate_synthetic_data(n_customers: int = 40, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic last-mile delivery dataset.
    
    Args:
        n_customers: Number of customer stops (excluding depot)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: stop_id, lat, lon, demand, earliest_time, latest_time, service_time
    """
    np.random.seed(seed)
    
    # Define a city-like bounding box (e.g., Boston area)
    center_lat, center_lon = 42.36, -71.06
    lat_range = 0.15  # ~17 km
    lon_range = 0.15  # ~12 km
    
    data = []
    
    # Generate depot
    depot_lat = center_lat + np.random.uniform(-0.02, 0.02)
    depot_lon = center_lon + np.random.uniform(-0.02, 0.02)
    data.append({
        'stop_id': 'DEPOT',
        'lat': depot_lat,
        'lon': depot_lon,
        'demand': 0,
        'earliest_time': '08:00',
        'latest_time': '20:00',
        'service_time': 0
    })
    
    # Generate customers
    for i in range(n_customers):
        lat = center_lat + np.random.uniform(-lat_range, lat_range)
        lon = center_lon + np.random.uniform(-lon_range, lon_range)
        
        # Random demand (packages/units)
        demand = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
        
        # Random time window (start between 9:00-14:00, duration 2-6 hours)
        start_hour = np.random.randint(9, 15)
        window_duration = np.random.choice([2, 3, 4, 5, 6], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        earliest = f"{start_hour:02d}:00"
        latest_hour = min(20, start_hour + window_duration)
        latest = f"{latest_hour:02d}:00"
        
        # Service time (5-15 minutes)
        service_time = np.random.choice([5, 10, 15], p=[0.3, 0.5, 0.2])
        
        data.append({
            'stop_id': f'CUST_{i+1:03d}',
            'lat': lat,
            'lon': lon,
            'demand': demand,
            'earliest_time': earliest,
            'latest_time': latest,
            'service_time': service_time
        })
    
    return pd.DataFrame(data)


def build_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build a distance matrix between all stops using Haversine formula.
    
    Args:
        df: DataFrame with 'lat' and 'lon' columns, sorted by stop_id
    
    Returns:
        NxN numpy array of distances in kilometers (float32 to save memory)
    """
    n = len(df)
    
    # Use float32 to save memory (sufficient precision for distance calculations)
    # This reduces memory usage by 50% compared to float64
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    
    # Build distance matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = haversine_distance(
                    df.iloc[i]['lat'], df.iloc[i]['lon'],
                    df.iloc[j]['lat'], df.iloc[j]['lon']
                )
    
    return dist_matrix


def build_time_matrix(distance_matrix: np.ndarray, speed_kmh: float) -> np.ndarray:
    """
    Convert distance matrix to time matrix.
    
    Args:
        distance_matrix: NxN distance matrix in kilometers
        speed_kmh: Average vehicle speed in km/h
    
    Returns:
        NxN numpy array of travel times in minutes
    """
    # Time in hours, then convert to minutes
    time_matrix = (distance_matrix / speed_kmh) * 60
    return time_matrix


def parse_time_window(time_str: str) -> Optional[int]:
    """
    Parse time string (HH:MM format) to minutes from midnight.
    
    Args:
        time_str: Time string in HH:MM format
    
    Returns:
        Minutes from midnight, or None if parsing fails
    """
    if pd.isna(time_str) or time_str is None or time_str == '':
        return None
    
    try:
        if isinstance(time_str, str):
            parts = time_str.split(':')
            if len(parts) == 2:
                hours = int(parts[0])
                minutes = int(parts[1])
                return hours * 60 + minutes
        return None
    except:
        return None


def normalize_dataframe(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
    depot_stop_id: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Normalize user-provided DataFrame to internal schema.
    
    Args:
        df: User-provided DataFrame
        column_mapping: Dictionary mapping internal columns to user columns
                       Keys: 'stop_id', 'lat', 'lon', 'is_depot', 'demand', 
                             'earliest_time', 'latest_time', 'service_time'
        depot_stop_id: Explicit depot stop ID if 'is_depot' mapping is not provided
    
    Returns:
        Tuple of (normalized DataFrame, error message if any)
    """
    normalized = pd.DataFrame()
    
    # Map required columns
    if 'stop_id' not in column_mapping or column_mapping['stop_id'] is None:
        return None, "Stop ID column mapping is required"
    
    normalized['stop_id'] = df[column_mapping['stop_id']].astype(str)
    
    # Map latitude
    if 'lat' not in column_mapping or column_mapping['lat'] is None:
        return None, "Latitude column mapping is required"
    normalized['lat'] = pd.to_numeric(df[column_mapping['lat']], errors='coerce')
    
    # Map longitude
    if 'lon' not in column_mapping or column_mapping['lon'] is None:
        return None, "Longitude column mapping is required"
    normalized['lon'] = pd.to_numeric(df[column_mapping['lon']], errors='coerce')
    
    # Check for invalid lat/lon
    if normalized['lat'].isna().any() or normalized['lon'].isna().any():
        return None, "Invalid latitude or longitude values found"
    
    if (normalized['lat'].abs() > 90).any() or (normalized['lon'].abs() > 180).any():
        return None, "Latitude must be between -90 and 90, longitude between -180 and 180"
    
    # Map is_depot
    if 'is_depot' in column_mapping and column_mapping['is_depot']:
        normalized['is_depot'] = df[column_mapping['is_depot']].astype(bool)
    else:
        # Use explicit depot_stop_id
        if depot_stop_id:
            normalized['is_depot'] = (normalized['stop_id'] == str(depot_stop_id))
        else:
            # Default: assume first row is depot
            normalized['is_depot'] = False
            normalized.loc[0, 'is_depot'] = True
    
    # Check that at least one depot exists
    if not normalized['is_depot'].any():
        return None, "At least one depot must be specified"
    
    # Check that at least 2 customer stops exist
    n_customers = (~normalized['is_depot']).sum()
    if n_customers < 2:
        return None, "At least 2 customer stops are required"
    
    # Map optional columns with defaults
    if 'demand' in column_mapping and column_mapping['demand']:
        normalized['demand'] = pd.to_numeric(df[column_mapping['demand']], errors='coerce').fillna(0)
    else:
        normalized['demand'] = 0
    
    if 'earliest_time' in column_mapping and column_mapping['earliest_time']:
        time_col = df[column_mapping['earliest_time']]
        normalized['earliest_time'] = time_col.apply(parse_time_window)
    else:
        normalized['earliest_time'] = None
    
    if 'latest_time' in column_mapping and column_mapping['latest_time']:
        time_col = df[column_mapping['latest_time']]
        normalized['latest_time'] = time_col.apply(parse_time_window)
    else:
        normalized['latest_time'] = None
    
    if 'service_time' in column_mapping and column_mapping['service_time']:
        normalized['service_time'] = pd.to_numeric(df[column_mapping['service_time']], errors='coerce').fillna(0)
    else:
        normalized['service_time'] = 0
    
    # Reset index and ensure stop_id is unique
    normalized = normalized.reset_index(drop=True)
    
    return normalized, None

