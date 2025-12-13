"""
Utilization and Fairness Metrics - Route balance analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict
import streamlit as st


def compute_utilization_metrics(solution: Dict) -> Dict:
    """
    Compute utilization and fairness metrics for routes.
    
    Args:
        solution: Solution dictionary with route_details
        
    Returns:
        Dictionary with utilization metrics
    """
    route_details = solution.get('route_details', [])
    
    if not route_details:
        return {
            'route_count': 0,
            'distance_stats': {},
            'duration_stats': {},
            'load_stats': {},
            'fairness_index_distance': None,
            'fairness_index_load': None,
        }
    
    # Extract metrics per route
    distances = []
    durations = []
    loads = []
    
    for route in route_details:
        distances.append(route.get('distance', 0))
        durations.append(route.get('time', 0) / 60)  # Convert to hours
        loads.append(route.get('demand', 0))
    
    distances = np.array(distances)
    durations = np.array(durations)
    loads = np.array(loads)
    
    # Compute statistics
    distance_stats = {
        'min': float(np.min(distances)) if len(distances) > 0 else 0,
        'max': float(np.max(distances)) if len(distances) > 0 else 0,
        'mean': float(np.mean(distances)) if len(distances) > 0 else 0,
        'std': float(np.std(distances)) if len(distances) > 0 else 0,
    }
    
    duration_stats = {
        'min': float(np.min(durations)) if len(durations) > 0 else 0,
        'max': float(np.max(durations)) if len(durations) > 0 else 0,
        'mean': float(np.mean(durations)) if len(durations) > 0 else 0,
        'std': float(np.std(durations)) if len(durations) > 0 else 0,
    }
    
    load_stats = {
        'min': float(np.min(loads)) if len(loads) > 0 else 0,
        'max': float(np.max(loads)) if len(loads) > 0 else 0,
        'mean': float(np.mean(loads)) if len(loads) > 0 else 0,
        'std': float(np.std(loads)) if len(loads) > 0 else 0,
    }
    
    # Fairness index (coefficient of variation = std/mean)
    fairness_index_distance = None
    if distance_stats['mean'] > 0:
        fairness_index_distance = distance_stats['std'] / distance_stats['mean']
    
    fairness_index_load = None
    if load_stats['mean'] > 0:
        fairness_index_load = load_stats['std'] / load_stats['mean']
    
    return {
        'route_count': len(route_details),
        'distance_stats': distance_stats,
        'duration_stats': duration_stats,
        'load_stats': load_stats,
        'fairness_index_distance': fairness_index_distance,
        'fairness_index_load': fairness_index_load,
    }


def render_utilization_ui(solution: Dict):
    """
    Render utilization and fairness metrics UI.
    
    Args:
        solution: Solution dictionary
    """
    if not solution or 'route_details' not in solution:
        return
    
    metrics = compute_utilization_metrics(solution)
    
    st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
    st.markdown("### ⚖️ Utilization & Fairness Metrics")
    st.markdown("Route balance analytics to assess workload distribution across vehicles.")
    
    # Summary table
    summary_data = []
    summary_data.append({
        'Metric': 'Route Count',
        'Value': f"{metrics['route_count']}",
        'Unit': 'routes'
    })
    
    # Distance metrics
    dist_stats = metrics['distance_stats']
    summary_data.append({
        'Metric': 'Distance - Min',
        'Value': f"{dist_stats['min']:.2f}",
        'Unit': 'km'
    })
    summary_data.append({
        'Metric': 'Distance - Max',
        'Value': f"{dist_stats['max']:.2f}",
        'Unit': 'km'
    })
    summary_data.append({
        'Metric': 'Distance - Mean',
        'Value': f"{dist_stats['mean']:.2f}",
        'Unit': 'km'
    })
    summary_data.append({
        'Metric': 'Distance - Std Dev',
        'Value': f"{dist_stats['std']:.2f}",
        'Unit': 'km'
    })
    
    # Fairness index
    if metrics['fairness_index_distance'] is not None:
        fairness_label = "Good" if metrics['fairness_index_distance'] < 0.3 else \
                        "Moderate" if metrics['fairness_index_distance'] < 0.5 else "Poor"
        summary_data.append({
            'Metric': 'Distance Fairness Index',
            'Value': f"{metrics['fairness_index_distance']:.3f} ({fairness_label})",
            'Unit': 'coefficient of variation'
        })
    
    # Load metrics
    load_stats = metrics['load_stats']
    summary_data.append({
        'Metric': 'Load - Min',
        'Value': f"{load_stats['min']:.1f}",
        'Unit': 'units'
    })
    summary_data.append({
        'Metric': 'Load - Max',
        'Value': f"{load_stats['max']:.1f}",
        'Unit': 'units'
    })
    summary_data.append({
        'Metric': 'Load - Mean',
        'Value': f"{load_stats['mean']:.1f}",
        'Unit': 'units'
    })
    
    if metrics['fairness_index_load'] is not None:
        fairness_label = "Good" if metrics['fairness_index_load'] < 0.3 else \
                        "Moderate" if metrics['fairness_index_load'] < 0.5 else "Poor"
        summary_data.append({
            'Metric': 'Load Fairness Index',
            'Value': f"{metrics['fairness_index_load']:.3f} ({fairness_label})",
            'Unit': 'coefficient of variation'
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Per-vehicle breakdown
    st.markdown("#### Per-Vehicle Breakdown")
    vehicle_data = []
    for route in solution['route_details']:
        vehicle_data.append({
            'Vehicle ID': route.get('vehicle_id', 'Unknown'),
            'Stops': route.get('n_stops', 0),
            'Distance (km)': f"{route.get('distance', 0):.2f}",
            'Duration (hrs)': f"{route.get('time', 0) / 60:.2f}",
            'Load': f"{route.get('demand', 0):.1f}",
        })
    
    vehicle_df = pd.DataFrame(vehicle_data)
    st.dataframe(vehicle_df, use_container_width=True, hide_index=True)
    
    # Interpretation
    st.info(
        "💡 **Fairness Index**: Lower is better (0 = perfectly balanced). "
        "Values < 0.3 indicate good balance, > 0.5 indicates significant imbalance."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

