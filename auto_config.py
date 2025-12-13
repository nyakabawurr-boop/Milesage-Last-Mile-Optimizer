"""
Automatic configuration module for Milesage - Last-Mile Optimizer

Provides functions to automatically determine reasonable VRP model parameters
based on dataset characteristics.
"""

import pandas as pd
from typing import Optional, Dict


def auto_configure_parameters(
    df: pd.DataFrame,
    demand_column: Optional[str] = None,
    earliest_time_column: Optional[str] = None,
    latest_time_column: Optional[str] = None,
) -> Dict:
    """
    Given the mapped dataset df and which optional columns are present,
    return a dictionary of suggested model configuration parameters.
    
    Args:
        df: Normalized DataFrame with stops
        demand_column: Name of demand column (usually 'demand')
        earliest_time_column: Name of earliest time column (usually 'earliest_time')
        latest_time_column: Name of latest time column (usually 'latest_time')
    
    Returns:
        Dictionary with suggested configuration parameters:
        - n_vehicles: Number of vehicles
        - vehicle_capacity: Vehicle capacity
        - max_route_duration_hours: Maximum route duration in hours
        - depot_open_hour: Depot opening hour (0-23)
        - depot_close_hour: Depot closing hour (0-23)
        - speed_kmh: Average vehicle speed (km/h)
        - use_capacity: Whether to use capacity constraints
        - use_time_windows: Whether to use time window constraints
    """
    n_stops = len(df)
    
    # Check if demand column exists and has data
    if demand_column and demand_column in df.columns:
        total_demand = df[demand_column].sum()
        max_demand = df[demand_column].max()
        has_demand_data = total_demand > 0
    else:
        total_demand = n_stops  # Treat each stop as demand 1
        max_demand = 1.0
        has_demand_data = False
    
    # Check if time windows exist
    has_time_windows = (
        earliest_time_column is not None and 
        latest_time_column is not None and
        earliest_time_column in df.columns and 
        latest_time_column in df.columns and
        (df[earliest_time_column].notna().any() or df[latest_time_column].notna().any())
    )
    
    # Auto-suggested parameters
    
    # Number of vehicles: roughly 20-30 stops per vehicle
    num_vehicles = max(1, min(50, int((n_stops / 25.0) + 0.5)))
    
    # Vehicle capacity
    if has_demand_data and demand_column and demand_column in df.columns:
        avg_demand_per_stop = total_demand / max(n_stops, 1)
        base_capacity = avg_demand_per_stop * 30  # ~30 stops worth of load
        vehicle_capacity = max(max_demand * 1.5, base_capacity)
        use_capacity = True
    else:
        vehicle_capacity = 10**9  # effectively unlimited
        use_capacity = False
    
    # Max route duration: assume 8-10 hour day; longer if time windows exist
    max_route_hours = 9.0 if not has_time_windows else 10.0
    
    # Depot hours
    depot_open_hour = 8
    if not has_time_windows:
        depot_close_hour = 18
    else:
        depot_close_hour = 20
    
    # Average speed: urban-ish default
    avg_speed_kmh = 45
    
    return {
        "n_vehicles": num_vehicles,
        "vehicle_capacity": float(vehicle_capacity),
        "max_route_duration_hours": max_route_hours,
        "depot_open_hour": depot_open_hour,
        "depot_close_hour": depot_close_hour,
        "speed_kmh": avg_speed_kmh,
        "use_capacity": use_capacity,
        "use_time_windows": has_time_windows,
    }


def run_vrp_with_auto_relaxation(
    run_ortools_func,
    df: pd.DataFrame,
    distance_matrix,
    time_matrix,
    initial_config: Dict
) -> tuple:
    """
    Run VRP solver with automatic constraint relaxation if infeasible.
    
    Args:
        run_ortools_func: Function to call OR-Tools solver
        df: Normalized DataFrame with stops
        distance_matrix: Distance matrix
        time_matrix: Time matrix
        initial_config: Initial configuration dictionary
    
    Returns:
        Tuple of (solution_dict, final_config_dict, relaxation_info)
        where relaxation_info is None if no relaxation was needed,
        or a dict describing what was relaxed.
    """
    # Create a copy of config for modification
    config = initial_config.copy()
    relaxation_info = None
    
    # Helper to run solver with current config
    def try_solve():
        return run_ortools_func(
            df,
            distance_matrix,
            time_matrix,
            config['n_vehicles'],
            config.get('vehicle_capacity') if config.get('use_capacity') else None,
            config.get('max_route_duration_hours'),
            config.get('depot_time_window', (480, 1200)),
            config.get('first_solution_strategy', 'PATH_CHEAPEST_ARC'),
            config.get('local_search_metaheuristic', 'GUIDED_LOCAL_SEARCH'),
            config.get('time_limit_seconds', 30)
        )
    
    # 1) Try with given config
    solution = try_solve()
    if 'error' not in solution:
        return solution, config, None
    
    # 2) Relax capacity (if in use)
    if config.get('use_capacity', False) and config.get('vehicle_capacity'):
        original_capacity = config['vehicle_capacity']
        for factor in [1.5, 2.0, 3.0]:
            config['vehicle_capacity'] = original_capacity * factor
            solution = try_solve()
            if 'error' not in solution:
                relaxation_info = {
                    'relaxed': True,
                    'capacity_factor': factor,
                    'new_capacity': config['vehicle_capacity'],
                    'max_route_hours': config.get('max_route_duration_hours'),
                }
                return solution, config, relaxation_info
    
        # Reset capacity for next step
        config['vehicle_capacity'] = original_capacity
    
    # 3) Relax time / hours
    original_max_hours = config.get('max_route_duration_hours', 9.0)
    original_close_hour = config.get('depot_close_hour', 18)
    depot_open = config.get('depot_open_hour', 8)
    
    for extra_hours in [2, 4, 6]:
        config['max_route_duration_hours'] = original_max_hours + extra_hours
        new_close_hour = min(23, original_close_hour + extra_hours)
        config['depot_close_hour'] = new_close_hour
        config['depot_time_window'] = (depot_open * 60, new_close_hour * 60)
        
        solution = try_solve()
        if 'error' not in solution:
            relaxation_info = {
                'relaxed': True,
                'capacity_factor': None,
                'new_capacity': config.get('vehicle_capacity'),
                'max_route_hours': config['max_route_duration_hours'],
                'depot_close_hour': new_close_hour,
            }
            return solution, config, relaxation_info
    
    # 4) Disable capacity/time windows as last resort
    config['use_capacity'] = False
    config['use_time_windows'] = False
    config['max_route_duration_hours'] = original_max_hours + 6  # Keep the extra hours
    config['depot_close_hour'] = min(23, original_close_hour + 6)
    config['depot_time_window'] = (depot_open * 60, config['depot_close_hour'] * 60)
    
    solution = try_solve()
    if 'error' not in solution:
        relaxation_info = {
            'relaxed': True,
            'capacity_disabled': True,
            'time_windows_disabled': True,
            'max_route_hours': config['max_route_duration_hours'],
            'depot_close_hour': config['depot_close_hour'],
        }
        return solution, config, relaxation_info
    
    # All attempts failed
    return solution, config, {'relaxed': False, 'all_failed': True}

