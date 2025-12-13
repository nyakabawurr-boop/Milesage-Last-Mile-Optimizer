"""
Export utilities for routes and reports.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Optional
from datetime import datetime
try:
    from fpdf import FPDF
except ImportError:
    from fpdf2 import FPDF


def export_routes_to_csv(solution: Dict, df: pd.DataFrame, scenario_name: str = "Current") -> pd.DataFrame:
    """
    Build a DataFrame with route details for CSV export.
    
    Args:
        solution: Solution dictionary
        df: DataFrame with stops
        scenario_name: Name of the scenario
        
    Returns:
        DataFrame with route details
    """
    route_rows = []
    
    for route_detail in solution.get('route_details', []):
        vehicle_id = route_detail.get('vehicle_id', 'Unknown')
        stops = route_detail.get('stops', [])
        
        for seq_idx, stop_idx in enumerate(stops):
            if stop_idx < len(df):
                stop_row = df.iloc[stop_idx]
                route_rows.append({
                    'scenario_name': scenario_name,
                    'vehicle_id': vehicle_id,
                    'stop_sequence': seq_idx,
                    'stop_id': stop_row.get('stop_id', ''),
                    'latitude': stop_row.get('lat', 0),
                    'longitude': stop_row.get('lon', 0),
                    'is_depot': stop_row.get('is_depot', False),
                    'demand': stop_row.get('demand', 0),
                    'route_distance_km': route_detail.get('distance', 0),
                    'route_duration_minutes': route_detail.get('time', 0),
                })
    
    return pd.DataFrame(route_rows)


def generate_pdf_report(
    scenario_name: str,
    kpis: Dict,
    route_details: list,
    config: Dict
) -> bytes:
    """
    Generate a PDF report with scenario details.
    
    Args:
        scenario_name: Name of the scenario
        kpis: Dictionary with KPIs
        route_details: List of route detail dictionaries
        config: Configuration dictionary
        
    Returns:
        PDF bytes
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Milesage Optimization Report: {scenario_name}", ln=1)
    pdf.ln(5)
    
    # Timestamp
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(5)
    
    # KPIs Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Key Performance Indicators", ln=1)
    pdf.set_font("Arial", "", 10)
    
    pdf.cell(0, 8, f"Total Distance: {kpis.get('opt_distance', 0):.2f} km", ln=1)
    pdf.cell(0, 8, f"Total Time: {kpis.get('opt_time', 0) / 60:.2f} hours", ln=1)
    pdf.cell(0, 8, f"Total Cost: ${kpis.get('opt_cost', 0):.2f}", ln=1)
    pdf.cell(0, 8, f"Vehicles Used: {kpis.get('opt_vehicles', 0)}", ln=1)
    
    if 'distance_improvement_pct' in kpis:
        pdf.cell(0, 8, f"Distance Improvement: {kpis['distance_improvement_pct']:.1f}%", ln=1)
    
    pdf.ln(5)
    
    # Route Details Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Route Details", ln=1)
    pdf.set_font("Arial", "", 9)
    
    for route_detail in route_details[:10]:  # Limit to first 10 routes to avoid overflow
        vehicle_id = route_detail.get('vehicle_id', 'Unknown')
        n_stops = route_detail.get('n_stops', 0)
        distance = route_detail.get('distance', 0)
        time = route_detail.get('time', 0)
        
        pdf.cell(0, 7, f"Vehicle {vehicle_id}: {n_stops} stops, {distance:.2f} km, {time/60:.2f} hrs", ln=1)
    
    if len(route_details) > 10:
        pdf.cell(0, 7, f"... and {len(route_details) - 10} more routes", ln=1)
    
    pdf.ln(5)
    
    # Configuration Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Configuration", ln=1)
    pdf.set_font("Arial", "", 10)
    
    model_params = config.get('model_params', {})
    pdf.cell(0, 8, f"Max Vehicles: {model_params.get('n_vehicles', 'N/A')}", ln=1)
    pdf.cell(0, 8, f"Vehicle Capacity: {model_params.get('vehicle_capacity', 'N/A')}", ln=1)
    pdf.cell(0, 8, f"Max Route Duration: {model_params.get('max_route_duration_hours', 'N/A')} hours", ln=1)
    
    return pdf.output(dest='S').encode('latin-1')


def render_export_ui(solution: Dict, df: pd.DataFrame, config: Dict, scenario_name: str = "Current"):
    """
    Render export buttons UI.
    
    Args:
        solution: Solution dictionary
        df: DataFrame with stops
        config: Configuration dictionary
        scenario_name: Name of the scenario
    """
    st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        if solution and 'route_details' in solution:
            routes_df = export_routes_to_csv(solution, df, scenario_name)
            
            csv = routes_df.to_csv(index=False)
            st.download_button(
                label="📊 Download All Routes as CSV",
                data=csv,
                file_name=f"milesage_routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Single vehicle export
            vehicle_ids = [rd.get('vehicle_id') for rd in solution.get('route_details', [])]
            if vehicle_ids:
                selected_vehicle = st.selectbox(
                    "Select vehicle for detailed export",
                    options=["All vehicles"] + vehicle_ids,
                    key="export_vehicle_select"
                )
                
                if selected_vehicle and selected_vehicle != "All vehicles":
                    vehicle_routes_df = routes_df[routes_df['vehicle_id'] == selected_vehicle]
                    csv_single = vehicle_routes_df.to_csv(index=False)
                    st.download_button(
                        label=f"📋 Download {selected_vehicle} Manifest",
                        data=csv_single,
                        file_name=f"milesage_vehicle_{selected_vehicle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    with col2:
        # PDF Export
        if solution and 'route_details' in solution:
            # Calculate KPIs for PDF
            kpis = {
                'opt_distance': solution.get('total_distance', 0),
                'opt_time': solution.get('total_time', 0),
                'opt_vehicles': solution.get('n_vehicles_used', 0),
                'opt_cost': 0,
            }
            
            # Calculate cost
            cost_per_km = config.get('cost_per_km', 1.5)
            fixed_cost = config.get('fixed_cost_per_vehicle', 50.0)
            cost_per_hour = config.get('cost_per_hour', 25.0)
            kpis['opt_cost'] = (
                kpis['opt_distance'] * cost_per_km +
                kpis['opt_vehicles'] * fixed_cost +
                (kpis['opt_time'] / 60) * cost_per_hour
            )
            
            # Get route details
            route_details = solution.get('route_details', [])
            
            # Build config dict for PDF
            pdf_config = {
                'model_params': {
                    'n_vehicles': config.get('n_vehicles', 5),
                    'vehicle_capacity': config.get('vehicle_capacity'),
                    'max_route_duration_hours': config.get('max_route_duration_hours', 8.0),
                }
            }
            
            try:
                pdf_bytes = generate_pdf_report(scenario_name, kpis, route_details, pdf_config)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"milesage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"PDF generation may not work: {e}")
                st.info("💡 Make sure fpdf2 is installed: pip install fpdf2")
    
    st.markdown('</div>', unsafe_allow_html=True)

