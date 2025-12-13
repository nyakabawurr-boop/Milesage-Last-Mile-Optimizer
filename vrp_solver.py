"""
VRP solver module for Milesage - Last-Mile Optimizer

Implements both a naive nearest-neighbor heuristic and OR-Tools-based VRP optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def run_naive_solution(
    df: pd.DataFrame,
    distance_matrix: np.ndarray,
    time_matrix: np.ndarray,
    n_vehicles: int,
    vehicle_capacity: Optional[float] = None,
    max_route_duration_hours: Optional[float] = None,
    depot_time_window: Tuple[int, int] = (480, 1200)  # 8:00-20:00 in minutes
) -> Dict:
    """
    Run a naive nearest-neighbor heuristic for VRP.
    
    Args:
        df: Normalized DataFrame with stops
        distance_matrix: NxN distance matrix in km
        time_matrix: NxN time matrix in minutes
        n_vehicles: Number of vehicles available
        vehicle_capacity: Vehicle capacity (None to disable capacity constraint)
        max_route_duration_hours: Maximum route duration in hours (None to disable)
        depot_time_window: (start_minutes, end_minutes) for depot operating hours
    
    Returns:
        Dictionary with solution details:
        - routes: List of routes, each route is a list of stop indices
        - total_distance: Total distance in km
        - total_time: Total time in minutes
        - route_details: List of dicts with per-route metrics
    """
    n = int(len(df))
    n_vehicles = int(n_vehicles)
    if n < 2 or n_vehicles < 1:
        return {
            'routes': [],
            'total_distance': 0,
            'total_time': 0,
            'route_details': [],
            'n_vehicles_used': 0,
            'error': 'Not enough stops or vehicles to build routes.'
        }
    depot_idx = df[df['is_depot']].index[0]
    customer_indices = df[~df['is_depot']].index.tolist()
    
    # Initialize routes
    routes = []
    unvisited = set(customer_indices)
    
    # Track whether we're using constraints
    use_capacity = vehicle_capacity is not None and df['demand'].sum() > 0
    use_duration = max_route_duration_hours is not None
    
    max_duration_minutes = max_route_duration_hours * 60 if use_duration else None
    
    for vehicle_id in range(n_vehicles):
        if not unvisited:
            break
        
        # Start route at depot
        route = [depot_idx]
        current_capacity = 0
        current_time = depot_time_window[0]  # Start at depot opening time
        
        while unvisited:
            current_idx = route[-1]
            best_next = None
            best_distance = float('inf')
            
            # Find nearest unvisited customer that satisfies constraints
            for next_idx in unvisited:
                # Check capacity constraint
                if use_capacity:
                    demand = df.loc[next_idx, 'demand']
                    if current_capacity + demand > vehicle_capacity:
                        continue
                
                # Check duration constraint
                if use_duration:
                    travel_time = time_matrix[current_idx, next_idx]
                    service_time = df.loc[next_idx, 'service_time']
                    arrival_time = current_time + travel_time
                    
                    # Check time window
                    earliest = df.loc[next_idx, 'earliest_time']
                    latest = df.loc[next_idx, 'latest_time']
                    
                    if earliest is not None:
                        if arrival_time < earliest:
                            arrival_time = earliest  # Wait until earliest time
                    
                    if latest is not None:
                        if arrival_time > latest:
                            continue  # Cannot meet time window
                    
                    departure_time = arrival_time + service_time
                    
                    # Check if we can return to depot in time
                    return_time = departure_time + time_matrix[next_idx, depot_idx]
                    if return_time > depot_time_window[1] or \
                       (return_time - depot_time_window[0]) > max_duration_minutes:
                        continue
                
                # Check distance
                dist = distance_matrix[current_idx, next_idx]
                if dist < best_distance:
                    best_distance = dist
                    best_next = next_idx
            
            if best_next is None:
                break  # No feasible next stop
            
            # Add best next stop to route
            route.append(best_next)
            unvisited.remove(best_next)
            
            # Update capacity and time
            if use_capacity:
                current_capacity += df.loc[best_next, 'demand']
            
            if use_duration:
                travel_time = time_matrix[current_idx, best_next]
                service_time = df.loc[best_next, 'service_time']
                
                arrival_time = current_time + travel_time
                earliest = df.loc[best_next, 'earliest_time']
                if earliest is not None and arrival_time < earliest:
                    arrival_time = earliest
                
                current_time = arrival_time + service_time
        
        # Return to depot
        route.append(depot_idx)
        routes.append(route)
    
    # Compute route metrics
    route_details = []
    total_distance = 0
    total_time = 0
    
    for vehicle_id, route in enumerate(routes):
        if len(route) <= 2:  # Only depot -> depot
            continue
        
        route_dist = 0
        route_time = 0
        route_demand = 0
        
        for i in range(len(route) - 1):
            route_dist += distance_matrix[route[i], route[i+1]]
            route_time += time_matrix[route[i], route[i+1]]
        
        # Add service times
        for stop_idx in route[1:-1]:  # Exclude depot at start/end
            route_time += df.loc[stop_idx, 'service_time']
            route_demand += df.loc[stop_idx, 'demand']
        
        route_details.append({
            'vehicle_id': vehicle_id,
            'stops': route,
            'distance': route_dist,
            'time': route_time,
            'demand': route_demand,
            'n_stops': len(route) - 2  # Exclude depot at start/end
        })
        
        total_distance += route_dist
        total_time += route_time
    
    return {
        'routes': routes,
        'total_distance': total_distance,
        'total_time': total_time,
        'route_details': route_details,
        'n_vehicles_used': len([r for r in routes if len(r) > 2])
    }


def run_ortools_solution(
    df: pd.DataFrame,
    distance_matrix: np.ndarray,
    time_matrix: np.ndarray,
    n_vehicles: int,
    vehicle_capacity: Optional[float] = None,
    max_route_duration_hours: Optional[float] = None,
    depot_time_window: Tuple[int, int] = (480, 1200),
    first_solution_strategy: str = 'PATH_CHEAPEST_ARC',
    local_search_metaheuristic: str = 'GUIDED_LOCAL_SEARCH',
    time_limit_seconds: int = 30
) -> Dict:
    """
    Run OR-Tools VRP solver.
    
    Args:
        df: Normalized DataFrame with stops
        distance_matrix: NxN distance matrix in km
        time_matrix: NxN time matrix in minutes
        n_vehicles: Number of vehicles available
        vehicle_capacity: Vehicle capacity (None to disable)
        max_route_duration_hours: Maximum route duration in hours (None to disable)
        depot_time_window: (start_minutes, end_minutes) for depot
        first_solution_strategy: OR-Tools first solution strategy
        local_search_metaheuristic: OR-Tools local search metaheuristic
        time_limit_seconds: Time limit for solver
    
    Returns:
        Dictionary with solution details (same format as naive solution)
    """
    n = int(len(df))
    n_vehicles = int(n_vehicles)
    if n < 2:
        return {
            'routes': [],
            'total_distance': 0,
            'total_time': 0,
            'route_details': [],
            'n_vehicles_used': 0,
            'error': 'Not enough stops to build a route.'
        }
    if n_vehicles < 1:
        return {
            'routes': [],
            'total_distance': 0,
            'total_time': 0,
            'route_details': [],
            'n_vehicles_used': 0,
            'error': 'Number of vehicles must be at least 1.'
        }

    depot_idx = int(df[df['is_depot']].index[0])
    
    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)
    
    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node, to_node] * 1000)  # Convert to meters (integer)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity dimension
    use_capacity = vehicle_capacity is not None and df['demand'].sum() > 0
    if use_capacity:
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(df.loc[from_node, 'demand'])
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimension(
            demand_callback_index,
            0,  # null capacity slack
            int(vehicle_capacity),  # vehicle maximum capacity
            True,  # start cumul to zero
            'Capacity'
        )
    
    # Add time dimension
    use_time_windows = df['earliest_time'].notna().any() or df['latest_time'].notna().any()
    use_duration = max_route_duration_hours is not None
    use_time = use_time_windows or use_duration
    
    if use_time:
        max_route_duration = int(max_route_duration_hours * 60) if max_route_duration_hours is not None else 999999
        
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel_time = int(time_matrix[from_node, to_node])
            service_time = int(df.loc[from_node, 'service_time'])
            return travel_time + service_time
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        routing.AddDimension(
            time_callback_index,
            max_route_duration,  # slack
            max_route_duration,  # maximum time per vehicle
            False,  # don't start cumul to zero
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add time window constraints for nodes
        for node_idx in range(n):
            index = manager.NodeToIndex(node_idx)
            earliest = df.loc[node_idx, 'earliest_time']
            latest = df.loc[node_idx, 'latest_time']
            
            if df.loc[node_idx, 'is_depot']:
                # Depot time window
                time_dimension.CumulVar(index).SetRange(depot_time_window[0], depot_time_window[1])
            else:
                # Customer time windows
                if earliest is not None and latest is not None:
                    time_dimension.CumulVar(index).SetRange(earliest, latest)
                elif earliest is not None:
                    time_dimension.CumulVar(index).SetMin(earliest)
                elif latest is not None:
                    time_dimension.CumulVar(index).SetMax(latest)
        
        # Add depot time window constraints for start/end of routes
        for vehicle_id in range(n_vehicles):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(depot_time_window[0], depot_time_window[1])
            index = routing.End(vehicle_id)
            time_dimension.CumulVar(index).SetRange(depot_time_window[0], depot_time_window[1])
    
    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # Map string to first solution strategy
    strategy_map = {
        'PATH_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        'PATH_MOST_CONSTRAINED_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        'SAVINGS': routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        'SWEEP': routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        'CHRISTOFIDES': routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
    }
    search_parameters.first_solution_strategy = strategy_map.get(
        first_solution_strategy,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    # Map string to local search metaheuristic
    metaheuristic_map = {
        'GUIDED_LOCAL_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        'TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        'SIMULATED_ANNEALING': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        'None': routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
    }
    if local_search_metaheuristic and local_search_metaheuristic != 'None':
        search_parameters.local_search_metaheuristic = metaheuristic_map.get(
            local_search_metaheuristic,
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_seconds
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return {
            'routes': [],
            'total_distance': 0,
            'total_time': 0,
            'route_details': [],
            'n_vehicles_used': 0,
            'error': 'No feasible solution found with current constraints'
        }
    
    # Extract solution
    routes = []
    route_details = []
    total_distance = 0
    total_time = 0
    
    for vehicle_id in range(n_vehicles):
        route = []
        index = routing.Start(vehicle_id)
        route_dist = 0
        route_time = 0
        route_demand = 0
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            route.append(node_idx)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            if not routing.IsEnd(index):
                route_dist += distance_matrix[node_idx, manager.IndexToNode(index)]
                route_time += time_matrix[node_idx, manager.IndexToNode(index)]
        
        # Add end depot
        node_idx = manager.IndexToNode(index)
        route.append(node_idx)
        
        # Add service times and demands
        for stop_idx in route[1:-1]:  # Exclude depot at start/end
            route_time += df.loc[stop_idx, 'service_time']
            route_demand += df.loc[stop_idx, 'demand']
        
        if len(route) > 2:  # Only include non-empty routes
            routes.append(route)
            route_details.append({
                'vehicle_id': vehicle_id,
                'stops': route,
                'distance': route_dist,
                'time': route_time,
                'demand': route_demand,
                'n_stops': len(route) - 2
            })
            total_distance += route_dist
            total_time += route_time
    
    return {
        'routes': routes,
        'total_distance': total_distance,
        'total_time': total_time,
        'route_details': route_details,
        'n_vehicles_used': len(routes)
    }

