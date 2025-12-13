"""
Command-line interface for Milesage VRP optimizer.

Usage:
    python cli.py --input data.csv --lat lat --lon lon --id stop_id --output routes.csv
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

# Import core modules
try:
    from data_utils import normalize_dataframe, build_distance_matrix, build_time_matrix
    from vrp_solver import run_ortools_solution
    from auto_config import auto_configure_parameters
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the project directory and dependencies are installed.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Milesage VRP Optimizer - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --input data.csv --lat latitude --lon longitude --id stop_id --output routes.csv
  python cli.py --input data.csv --lat lat --lon lon --id id --output routes.csv --vehicles 5
        """
    )
    
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path for routes')
    parser.add_argument('--lat', type=str, required=True, help='Latitude column name')
    parser.add_argument('--lon', type=str, required=True, help='Longitude column name')
    parser.add_argument('--id', type=str, required=True, help='Stop ID column name')
    parser.add_argument('--vehicles', type=int, default=5, help='Number of vehicles (default: 5)')
    parser.add_argument('--capacity', type=float, default=None, help='Vehicle capacity (optional)')
    parser.add_argument('--speed', type=float, default=40.0, help='Average speed in km/h (default: 40)')
    parser.add_argument('--max-hours', type=float, default=8.0, help='Max route duration in hours (default: 8)')
    
    args = parser.parse_args()
    
    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Loaded {len(df)} rows from {args.input}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Check required columns
    for col_name, col_arg in [('lat', args.lat), ('lon', args.lon), ('stop_id', args.id)]:
        if col_arg not in df.columns:
            print(f"Error: Column '{col_arg}' not found in input file.")
            print(f"Available columns: {', '.join(df.columns)}")
            sys.exit(1)
    
    # Build column mapping
    column_mapping = {
        'stop_id': args.id,
        'lat': args.lat,
        'lon': args.lon,
        'is_depot': None,
        'demand': None,
        'earliest_time': None,
        'latest_time': None,
        'service_time': None,
    }
    
    # Normalize dataframe
    try:
        normalized_df, error = normalize_dataframe(df, column_mapping, None)
        if error:
            print(f"Error normalizing data: {error}")
            sys.exit(1)
        print(f"✅ Normalized data: {len(normalized_df)} stops")
    except Exception as e:
        print(f"Error normalizing data: {e}")
        sys.exit(1)
    
    # Build distance and time matrices
    print("🔄 Building distance matrix...")
    try:
        distance_matrix = build_distance_matrix(normalized_df)
        time_matrix = build_time_matrix(distance_matrix, args.speed)
        print(f"✅ Built matrices: {len(normalized_df)}x{len(normalized_df)}")
    except Exception as e:
        print(f"Error building matrices: {e}")
        sys.exit(1)
    
    # Auto-configure parameters
    config = auto_configure_parameters(
        normalized_df,
        demand_column=None,
        earliest_time_column=None,
        latest_time_column=None,
    )
    
    # Override with CLI arguments
    config['n_vehicles'] = args.vehicles
    config['vehicle_capacity'] = args.capacity if args.capacity else config.get('vehicle_capacity')
    config['speed_kmh'] = args.speed
    config['max_route_duration_hours'] = args.max_hours
    config['depot_time_window'] = (480, 1200)  # 8:00-20:00
    config['first_solution_strategy'] = 'PATH_CHEAPEST_ARC'
    config['local_search_metaheuristic'] = 'GUIDED_LOCAL_SEARCH'
    config['time_limit_seconds'] = 30
    
    # Run optimization
    print("🔄 Running optimization...")
    try:
        solution = run_ortools_solution(
            normalized_df,
            distance_matrix,
            time_matrix,
            config['n_vehicles'],
            config['vehicle_capacity'] if config.get('use_capacity', False) else None,
            config['max_route_duration_hours'],
            config['depot_time_window'],
            config['first_solution_strategy'],
            config['local_search_metaheuristic'],
            config['time_limit_seconds']
        )
        
        if 'error' in solution:
            print(f"❌ Optimization failed: {solution['error']}")
            sys.exit(1)
        
        print(f"✅ Optimization complete!")
        print(f"   Total distance: {solution['total_distance']:.2f} km")
        print(f"   Total time: {solution['total_time'] / 60:.2f} hours")
        print(f"   Vehicles used: {solution['n_vehicles_used']}")
    except Exception as e:
        print(f"Error running optimization: {e}")
        sys.exit(1)
    
    # Export routes to CSV
    print("🔄 Exporting routes...")
    try:
        route_rows = []
        for route_detail in solution.get('route_details', []):
            vehicle_id = route_detail.get('vehicle_id', 'Unknown')
            stops = route_detail.get('stops', [])
            
            for seq_idx, stop_idx in enumerate(stops):
                if stop_idx < len(normalized_df):
                    stop_row = normalized_df.iloc[stop_idx]
                    route_rows.append({
                        'vehicle_id': vehicle_id,
                        'stop_sequence': seq_idx,
                        'stop_id': stop_row.get('stop_id', ''),
                        'latitude': stop_row.get('lat', 0),
                        'longitude': stop_row.get('lon', 0),
                        'is_depot': stop_row.get('is_depot', False),
                        'route_distance_km': route_detail.get('distance', 0),
                        'route_duration_minutes': route_detail.get('time', 0),
                    })
        
        routes_df = pd.DataFrame(route_rows)
        routes_df.to_csv(args.output, index=False)
        print(f"✅ Routes exported to {args.output}")
        print(f"   Total routes: {len(solution['route_details'])}")
    except Exception as e:
        print(f"Error exporting routes: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

