"""
Visualization utilities for Milesage - Last-Mile Optimizer

Handles map visualization using Folium and summary plots.
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from typing import Dict, List, Optional


def get_route_colors(n_routes: int) -> List[List[int]]:
    """
    Generate distinct colors for routes matching dark theme dashboard.
    
    Args:
        n_routes: Number of routes
    
    Returns:
        List of RGB color tuples [R, G, B] where each value is 0-255
    """
    # Color palette matching dark theme dashboard (yellow, orange, blues, greens, purples)
    colors = [
        [255, 193, 7],    # Yellow/Amber (FFC107) - Primary accent
        [255, 152, 0],    # Orange (FF9800)
        [33, 150, 243],   # Light Blue (2196F3)
        [76, 175, 80],    # Green (4CAF50)
        [156, 39, 176],   # Purple (9C27B0)
        [244, 67, 54],    # Red (F44336)
        [0, 188, 212],    # Cyan (00BCD4)
        [255, 235, 59],   # Bright Yellow (FFEB3B)
        [255, 87, 34],    # Deep Orange (FF5722)
        [63, 81, 181],    # Indigo (3F51B5)
        [0, 150, 136],    # Teal (009688)
        [233, 30, 99],    # Pink (E91E63)
    ]
    
    # Repeat colors if needed
    while len(colors) < n_routes:
        colors.extend(colors)
    
    return colors[:n_routes]


def create_route_map_folium(
    df: pd.DataFrame,
    solution: Dict,
    solution_name: str = 'Solution'
) -> folium.Map:
    """
    Create an interactive Folium map showing routes with real street tiles.
    
    Args:
        df: Normalized DataFrame with stops
        solution: Solution dictionary with 'routes' and 'route_details'
        solution_name: Name for the solution (for legend)
    
    Returns:
        folium.Map object
    """
    # Get center of map
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    # Create base map with CartoDB positron tiles (clean, data analyst friendly)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='CartoDB positron'  # Clean, modern light basemap with streets
    )
    
    # Add fullscreen button plugin
    Fullscreen(
        position="topleft",
        title="Full screen",
        title_cancel="Exit full screen",
        force_separate_button=True
    ).add_to(m)
    
    # Color palette for routes (matching dark theme accent colors)
    route_colors = [
        "#FBBF24",  # Yellow/Amber
        "#34D399",  # Green
        "#60A5FA",  # Blue
        "#F97316",  # Orange
        "#EC4899",  # Pink
        "#8B5CF6",  # Purple
        "#14B8A6",  # Teal
        "#F59E0B",  # Dark Yellow
        "#10B981",  # Emerald
        "#3B82F6",  # Indigo
        "#EF4444",  # Red
        "#06B6D4",  # Cyan
    ]
    
    # Add depot marker (larger, yellow/amber color)
    depot_df = df[df['is_depot']].copy()
    for idx, row in depot_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=12,
            popup=folium.Popup(f"<b>Depot</b><br>Stop ID: {row['stop_id']}", max_width=200),
            tooltip=f"Depot: {row['stop_id']}",
            color="#1A3A3F",  # Dark border
            weight=2,
            fillColor="#FBBF24",  # Yellow/Amber accent
            fillOpacity=0.9
        ).add_to(m)
    
    # Add routes and customer stops
    for route_detail in solution['route_details']:
        vehicle_id = route_detail['vehicle_id']
        route_stops = route_detail['stops']
        
        if len(route_stops) < 3:  # Skip empty routes (only depot -> depot)
            continue
        
        # Get color for this vehicle
        color = route_colors[vehicle_id % len(route_colors)]
        
        # Create route path coordinates
        route_coords = []
        for stop_idx in route_stops:
            route_coords.append([df.loc[stop_idx, 'lat'], df.loc[stop_idx, 'lon']])
        
        # Draw route line (PolyLine)
        folium.PolyLine(
            locations=route_coords,
            color=color,
            weight=4,
            opacity=0.7,
            tooltip=f"Vehicle {vehicle_id} Route"
        ).add_to(m)
        
        # Add customer stop markers for this route
        for stop_idx in route_stops[1:-1]:  # Exclude depot at start and end
            stop_row = df.loc[stop_idx]
            if not stop_row['is_depot']:
                folium.CircleMarker(
                    location=[stop_row['lat'], stop_row['lon']],
                    radius=6,
                    popup=folium.Popup(
                        f"<b>Vehicle {vehicle_id}</b><br>Stop ID: {stop_row['stop_id']}<br>"
                        f"Demand: {stop_row['demand']}", 
                        max_width=200
                    ),
                    tooltip=f"Vehicle {vehicle_id} - {stop_row['stop_id']}",
                    color=color,
                    weight=2,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(m)
    
    # Add legend using folium's HTML template (Jinja2)
    from branca.element import MacroElement, Template
    
    # Build legend items HTML
    legend_items_html = ''
    for route_detail in solution['route_details']:
        if len(route_detail['stops']) >= 3:
            vehicle_id = route_detail['vehicle_id']
            color = route_colors[vehicle_id % len(route_colors)]
            n_stops = route_detail['n_stops']
            legend_items_html += f'<div style="margin: 4px 0;"><span style="color: {color}; font-size: 18px; font-weight: bold;">●</span> Vehicle {vehicle_id} ({n_stops} stops)</div>'
    
    # Build template string - avoid f-string conflict with Jinja2 syntax
    template_start = '{% macro html(this, kwargs) %}'
    template_end = '{% endmacro %}'
    legend_content = f'''
    <div style="position: fixed; 
                bottom: 50px; 
                right: 50px; 
                width: 220px; 
                background-color: #243D42; 
                border: 2px solid #2F4F54; 
                padding: 12px; 
                border-radius: 8px; 
                z-index:9999; 
                font-size:12px; 
                color: #FFFFFF;
                font-family: Arial, sans-serif;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px;">{solution_name} - Vehicle Routes</div>
        {legend_items_html}
    </div>
    '''
    
    # Combine template parts
    legend_html = template_start + legend_content + template_end
    
    # Create and add legend as MacroElement
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)
    
    return m


def create_summary_dataframe(naive_solution: Dict, optimized_solution: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame comparing naive vs optimized solutions.
    
    Args:
        naive_solution: Naive solution dictionary
        optimized_solution: Optimized solution dictionary
    
    Returns:
        DataFrame with comparison metrics
    """
    metrics = {
        'Metric': [
            'Total Distance (km)',
            'Total Travel Time (hours)',
            'Number of Vehicles Used',
            'Average Distance per Vehicle (km)',
            'Average Time per Vehicle (hours)'
        ],
        'Naive Solution': [
            f"{naive_solution['total_distance']:.2f}",
            f"{naive_solution['total_time'] / 60:.2f}",
            naive_solution['n_vehicles_used'],
            f"{naive_solution['total_distance'] / max(1, naive_solution['n_vehicles_used']):.2f}",
            f"{naive_solution['total_time'] / (60 * max(1, naive_solution['n_vehicles_used'])):.2f}"
        ],
        'Optimized Solution': [
            f"{optimized_solution['total_distance']:.2f}",
            f"{optimized_solution['total_time'] / 60:.2f}",
            optimized_solution['n_vehicles_used'],
            f"{optimized_solution['total_distance'] / max(1, optimized_solution['n_vehicles_used']):.2f}",
            f"{optimized_solution['total_time'] / (60 * max(1, optimized_solution['n_vehicles_used'])):.2f}"
        ]
    }
    
    # Calculate improvements
    improvements = []
    naive_dist = naive_solution['total_distance']
    opt_dist = optimized_solution['total_distance']
    naive_time = naive_solution['total_time']
    opt_time = optimized_solution['total_time']
    naive_veh = naive_solution['n_vehicles_used']
    opt_veh = optimized_solution['n_vehicles_used']
    
    if naive_dist > 0:
        dist_improvement = ((naive_dist - opt_dist) / naive_dist) * 100
        improvements.append(f"{dist_improvement:.1f}%")
    else:
        improvements.append("N/A")
    
    if naive_time > 0:
        time_improvement = ((naive_time - opt_time) / naive_time) * 100
        improvements.append(f"{time_improvement:.1f}%")
    else:
        improvements.append("N/A")
    
    improvements.append(f"{naive_veh - opt_veh}")
    
    if naive_veh > 0 and naive_dist > 0:
        avg_dist_improvement = ((naive_dist / naive_veh - opt_dist / max(1, opt_veh)) / (naive_dist / naive_veh)) * 100
        improvements.append(f"{avg_dist_improvement:.1f}%")
    else:
        improvements.append("N/A")
    
    if naive_veh > 0 and naive_time > 0:
        avg_time_improvement = ((naive_time / naive_veh - opt_time / max(1, opt_veh)) / (naive_time / naive_veh)) * 100
        improvements.append(f"{avg_time_improvement:.1f}%")
    else:
        improvements.append("N/A")
    
    metrics['Improvement'] = improvements
    
    return pd.DataFrame(metrics)


def generate_business_summary(
    naive_solution: Dict,
    optimized_solution: Dict,
    cost_per_km: float,
    fixed_cost_per_vehicle: float,
    cost_per_hour: float = 0
) -> str:
    """
    Generate a natural-language business summary of the optimization results.
    
    Args:
        naive_solution: Naive solution dictionary
        optimized_solution: Optimized solution dictionary
        cost_per_km: Cost per kilometer
        fixed_cost_per_vehicle: Fixed cost per vehicle used
        cost_per_hour: Optional cost per hour of driver time
    
    Returns:
        Formatted summary string
    """
    naive_dist = naive_solution['total_distance']
    opt_dist = optimized_solution['total_distance']
    naive_time = naive_solution['total_time'] / 60  # Convert to hours
    opt_time = optimized_solution['total_time'] / 60
    naive_veh = naive_solution['n_vehicles_used']
    opt_veh = optimized_solution['n_vehicles_used']
    
    # Calculate costs
    naive_cost = (naive_dist * cost_per_km + 
                  naive_veh * fixed_cost_per_vehicle + 
                  naive_time * cost_per_hour)
    opt_cost = (opt_dist * cost_per_km + 
                opt_veh * fixed_cost_per_vehicle + 
                opt_time * cost_per_hour)
    
    cost_savings = naive_cost - opt_cost
    cost_reduction_pct = (cost_savings / naive_cost * 100) if naive_cost > 0 else 0
    dist_reduction_pct = ((naive_dist - opt_dist) / naive_dist * 100) if naive_dist > 0 else 0
    
    summary = f"""
    **Optimization Summary**
    
    For this scenario, Milesage reduced total distance from {naive_dist:.1f} km to {opt_dist:.1f} km 
    (a {dist_reduction_pct:.1f}% reduction), and estimated daily cost from ${naive_cost:.2f} to ${opt_cost:.2f}. 
    This translates to approximately ${cost_savings:.2f} in daily savings and {naive_dist - opt_dist:.1f} fewer 
    kilometers driven per day, which can significantly reduce fuel consumption, labor costs, and emissions.
    """
    
    if naive_veh != opt_veh:
        summary += f"\n\nAdditionally, the optimized solution uses {opt_veh} vehicle(s) compared to {naive_veh} in the naive plan, "
        summary += f"potentially reducing fleet management complexity."
    
    return summary

