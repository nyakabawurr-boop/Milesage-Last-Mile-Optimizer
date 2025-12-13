"""
Scenario Manager - Save, load, and compare optimization scenarios.
"""

import streamlit as st
from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd


def save_scenario(name: str, notes: str = "") -> bool:
    """
    Save the current optimization scenario to session state.
    
    Args:
        name: Scenario name (must be unique)
        notes: Optional notes about the scenario
        
    Returns:
        True if saved successfully, False otherwise
    """
    # Validate that we have optimization results
    if st.session_state.get('optimized_solution') is None or st.session_state.get('naive_solution') is None:
        return False
    
    # Initialize scenarios dict if needed
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
    
    # Build scenario dict
    naive_sol = st.session_state.naive_solution
    opt_sol = st.session_state.optimized_solution
    config = st.session_state.config
    
    # Extract KPIs
    kpis = {
        'naive_distance': naive_sol.get('total_distance', 0),
        'naive_time': naive_sol.get('total_time', 0),
        'naive_vehicles': naive_sol.get('n_vehicles_used', 0),
        'opt_distance': opt_sol.get('total_distance', 0),
        'opt_time': opt_sol.get('total_time', 0),
        'opt_vehicles': opt_sol.get('n_vehicles_used', 0),
        'distance_improvement_pct': 0,
        'time_improvement_pct': 0,
    }
    
    # Calculate improvements
    if naive_sol.get('total_distance', 0) > 0:
        kpis['distance_improvement_pct'] = (
            (naive_sol['total_distance'] - opt_sol['total_distance']) / 
            naive_sol['total_distance'] * 100
        )
    
    if naive_sol.get('total_time', 0) > 0:
        kpis['time_improvement_pct'] = (
            (naive_sol['total_time'] - opt_sol['total_time']) / 
            naive_sol['total_time'] * 100
        )
    
    # Calculate total cost
    cost_per_km = config.get('cost_per_km', 1.5)
    fixed_cost = config.get('fixed_cost_per_vehicle', 50.0)
    cost_per_hour = config.get('cost_per_hour', 25.0)
    
    naive_cost = (
        naive_sol['total_distance'] * cost_per_km +
        naive_sol['n_vehicles_used'] * fixed_cost +
        (naive_sol['total_time'] / 60) * cost_per_hour
    )
    opt_cost = (
        opt_sol['total_distance'] * cost_per_km +
        opt_sol['n_vehicles_used'] * fixed_cost +
        (opt_sol['total_time'] / 60) * cost_per_hour
    )
    
    kpis['naive_cost'] = naive_cost
    kpis['opt_cost'] = opt_cost
    kpis['cost_improvement_pct'] = ((naive_cost - opt_cost) / naive_cost * 100) if naive_cost > 0 else 0
    
    # Extract routes
    routes = []
    for route_detail in opt_sol.get('route_details', []):
        routes.append({
            'vehicle_id': route_detail.get('vehicle_id'),
            'n_stops': route_detail.get('n_stops', 0),
            'distance': route_detail.get('distance', 0),
            'time': route_detail.get('time', 0),
            'demand': route_detail.get('demand', 0),
            'stops': route_detail.get('stops', [])
        })
    
    # Dataset label
    dataset_label = "Synthetic Dataset"
    if st.session_state.get('raw_df') is not None:
        dataset_label = f"Dataset ({len(st.session_state.raw_df)} rows)"
    
    scenario = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "dataset_label": dataset_label,
        "notes": notes,
        "filters": st.session_state.get('sampling_info', {}),
        "mapping": st.session_state.get('column_mapping', {}).copy(),
        "config_mode": st.session_state.get('config_mode', 'Manual'),
        "model_params": {
            'n_vehicles': config.get('n_vehicles', 5),
            'vehicle_capacity': config.get('vehicle_capacity'),
            'max_route_duration_hours': config.get('max_route_duration_hours', 8.0),
            'depot_open_hour': config.get('depot_open_hour', 8),
            'depot_close_hour': config.get('depot_close_hour', 20),
            'speed_kmh': config.get('speed_kmh', 40),
            'use_capacity': config.get('use_capacity', False),
            'use_time_windows': config.get('use_time_windows', False),
        },
        "solver_params": {
            'first_solution_strategy': config.get('first_solution_strategy', 'PATH_CHEAPEST_ARC'),
            'local_search_metaheuristic': config.get('local_search_metaheuristic', 'GUIDED_LOCAL_SEARCH'),
            'time_limit_seconds': config.get('time_limit_seconds', 30),
        },
        "cost_params": {
            'cost_per_km': cost_per_km,
            'fixed_cost_per_vehicle': fixed_cost,
            'cost_per_hour': cost_per_hour,
        },
        "kpis": kpis,
        "routes": routes,
    }
    
    # Save to session state
    st.session_state.scenarios[name] = scenario
    return True


def load_scenario(name: str) -> bool:
    """
    Load a saved scenario into the current session state.
    
    Args:
        name: Scenario name to load
        
    Returns:
        True if loaded successfully, False otherwise
    """
    if 'scenarios' not in st.session_state or name not in st.session_state.scenarios:
        return False
    
    scenario = st.session_state.scenarios[name]
    
    # Restore column mapping
    if 'mapping' in scenario:
        st.session_state.column_mapping = scenario['mapping'].copy()
    
    # Restore config mode
    if 'config_mode' in scenario:
        st.session_state.config_mode = scenario['config_mode']
    
    # Restore model and solver params into config
    if 'model_params' in scenario:
        for key, value in scenario['model_params'].items():
            st.session_state.config[key] = value
    
    if 'solver_params' in scenario:
        for key, value in scenario['solver_params'].items():
            st.session_state.config[key] = value
    
    if 'cost_params' in scenario:
        for key, value in scenario['cost_params'].items():
            st.session_state.config[key] = value
    
    return True


def get_scenario_names() -> List[str]:
    """Get list of all saved scenario names."""
    if 'scenarios' not in st.session_state:
        return []
    return list(st.session_state.scenarios.keys())


def compare_scenarios(scenario_names: List[str]) -> pd.DataFrame:
    """
    Create a comparison DataFrame for selected scenarios.
    
    Args:
        scenario_names: List of scenario names to compare
        
    Returns:
        DataFrame with comparison metrics
    """
    if 'scenarios' not in st.session_state:
        return pd.DataFrame()
    
    comparison_data = []
    
    for name in scenario_names:
        if name not in st.session_state.scenarios:
            continue
        
        scenario = st.session_state.scenarios[name]
        kpis = scenario.get('kpis', {})
        model_params = scenario.get('model_params', {})
        
        comparison_data.append({
            'Scenario': name,
            'Dataset': scenario.get('dataset_label', 'N/A'),
            'Mode': scenario.get('config_mode', 'N/A'),
            'Vehicles Used': kpis.get('opt_vehicles', 0),
            'Total Distance (km)': f"{kpis.get('opt_distance', 0):.2f}",
            'Total Time (hrs)': f"{kpis.get('opt_time', 0) / 60:.2f}",
            'Total Cost ($)': f"{kpis.get('opt_cost', 0):.2f}",
            'Distance Improvement (%)': f"{kpis.get('distance_improvement_pct', 0):.1f}",
            'Cost Improvement (%)': f"{kpis.get('cost_improvement_pct', 0):.1f}",
            'Max Vehicles': model_params.get('n_vehicles', 0),
        })
    
    return pd.DataFrame(comparison_data)


def render_scenario_manager_ui():
    """Render the Scenario Manager UI in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Scenario Manager")
    
    # Initialize scenarios if needed
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
    
    # Save scenario section
    with st.sidebar.expander("💾 Save Current Scenario", expanded=False):
        scenario_name = st.text_input(
            "Scenario Name",
            key="scenario_name_input",
            help="Enter a name for this scenario"
        )
        scenario_notes = st.text_area(
            "Notes (optional)",
            key="scenario_notes_input",
            help="Add any notes about this scenario",
            height=80
        )
        
        if st.button("💾 Save Scenario", use_container_width=True):
            if not scenario_name:
                st.error("Please enter a scenario name")
            elif st.session_state.get('optimized_solution') is None:
                st.error("Please run optimization first")
            else:
                if save_scenario(scenario_name, scenario_notes):
                    st.success(f"✅ Scenario '{scenario_name}' saved!")
                    st.rerun()
                else:
                    st.error("Failed to save scenario")
    
    # Load scenario section
    scenario_names = get_scenario_names()
    if scenario_names:
        with st.sidebar.expander("📂 Load Scenario", expanded=False):
            selected_scenario = st.selectbox(
                "Select scenario to load",
                options=[""] + scenario_names,
                key="load_scenario_select",
                help="Load a saved scenario's configuration"
            )
            
            if selected_scenario and st.button("📂 Load Scenario", use_container_width=True):
                if load_scenario(selected_scenario):
                    st.success(f"✅ Scenario '{selected_scenario}' loaded!")
                    st.info("💡 Go to Model Setup to review configuration, then re-run optimization if needed.")
                    st.rerun()
                else:
                    st.error("Failed to load scenario")
            
            # Delete scenario option
            if len(scenario_names) > 0:
                scenario_to_delete = st.selectbox(
                    "Delete scenario",
                    options=[""] + scenario_names,
                    key="delete_scenario_select"
                )
                if scenario_to_delete and st.button("🗑️ Delete", use_container_width=True):
                    del st.session_state.scenarios[scenario_to_delete]
                    st.success(f"✅ Scenario '{scenario_to_delete}' deleted!")
                    st.rerun()

