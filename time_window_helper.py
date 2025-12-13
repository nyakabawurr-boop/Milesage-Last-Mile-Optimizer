"""
Time Window Helper - Generate time windows from timestamps and SLA.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import streamlit as st


def generate_time_windows_from_timestamp(
    df: pd.DataFrame,
    timestamp_column: str,
    sla_hours: float,
    window_type: str = "forward"  # "forward" or "centered"
) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """
    Generate earliest_time and latest_time columns from a timestamp column.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of the timestamp column
        sla_hours: SLA window in hours (e.g., 2 or 4)
        window_type: "forward" (start at timestamp, end at timestamp + SLA) or 
                     "centered" (center window around timestamp ± SLA/2)
    
    Returns:
        Tuple of (DataFrame with new columns, earliest_time column name, latest_time column name)
    """
    df_result = df.copy()
    
    # Parse timestamp column
    try:
        timestamps = pd.to_datetime(df_result[timestamp_column])
    except Exception as e:
        raise ValueError(f"Could not parse timestamp column '{timestamp_column}': {e}")
    
    if window_type == "forward":
        # Start at timestamp, end at timestamp + SLA
        earliest_times = timestamps
        latest_times = timestamps + timedelta(hours=sla_hours)
    else:  # centered
        # Center window around timestamp
        half_sla = timedelta(hours=sla_hours / 2)
        earliest_times = timestamps - half_sla
        latest_times = timestamps + half_sla
    
    # Convert to HH:MM format (time of day)
    earliest_str = earliest_times.dt.strftime('%H:%M')
    latest_str = latest_times.dt.strftime('%H:%M')
    
    # Add new columns
    earliest_col_name = f"auto_earliest_time"
    latest_col_name = f"auto_latest_time"
    
    df_result[earliest_col_name] = earliest_str
    df_result[latest_col_name] = latest_str
    
    return df_result, earliest_col_name, latest_col_name


def compute_on_time_metrics(
    df: pd.DataFrame,
    solution: dict,
    earliest_time_col: Optional[str] = None,
    latest_time_col: Optional[str] = None
) -> dict:
    """
    Compute on-time delivery metrics if time windows are available.
    
    Args:
        df: DataFrame with stops
        solution: Solution dictionary with route details
        earliest_time_col: Column name for earliest time (optional)
        latest_time_col: Column name for latest time (optional)
    
    Returns:
        Dictionary with on-time metrics
    """
    if earliest_time_col not in df.columns or latest_time_col not in df.columns:
        return {
            'on_time_pct': None,
            'violations_count': None,
            'avg_early_minutes': None,
            'avg_late_minutes': None,
        }
    
    # This is a simplified version - in a full implementation, you'd need to
    # track actual arrival times through the routes
    # For now, return placeholder structure
    return {
        'on_time_pct': None,
        'violations_count': None,
        'avg_early_minutes': None,
        'avg_late_minutes': None,
        'note': 'Time window tracking requires route timing simulation (advanced feature)'
    }


def render_time_window_helper_ui(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Render UI for time window helper.
    
    Args:
        df: Current DataFrame
        
    Returns:
        Tuple of (updated DataFrame, earliest_time column name, latest_time column name)
    """
    if df is None or df.empty:
        return None, None, None
    
    st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
    st.markdown("### ⏰ Time Window Helper")
    st.markdown("Generate time windows from a timestamp column and delivery SLA.")
    
    # Find datetime-like columns
    datetime_columns = []
    for col in df.columns:
        try:
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if sample is not None:
                pd.to_datetime(sample)
                datetime_columns.append(col)
        except:
            pass
    
    if not datetime_columns:
        st.info("ℹ️ No datetime columns found in the dataset. Upload data with timestamp columns to use this feature.")
        st.markdown('</div>', unsafe_allow_html=True)
        return None, None, None
    
    col1, col2 = st.columns(2)
    
    with col1:
        timestamp_col = st.selectbox(
            "Base Timestamp Column",
            options=[None] + datetime_columns,
            help="Select the column containing delivery timestamps"
        )
    
    with col2:
        sla_hours = st.number_input(
            "Delivery SLA Window (hours)",
            min_value=0.5,
            max_value=24.0,
            value=2.0,
            step=0.5,
            help="Time window duration in hours"
        )
    
    window_type = st.radio(
        "Window Type",
        options=["forward", "centered"],
        format_func=lambda x: "Start at timestamp, end at timestamp + SLA" if x == "forward" else "Center window around timestamp (± SLA/2)",
        help="How to position the time window relative to the timestamp"
    )
    
    if st.button("🔄 Generate Time Windows", use_container_width=True):
        if timestamp_col:
            try:
                df_updated, earliest_col, latest_col = generate_time_windows_from_timestamp(
                    df, timestamp_col, sla_hours, window_type
                )
                
                st.success(f"✅ Time windows generated! New columns: `{earliest_col}`, `{latest_col}`")
                st.info(f"💡 You can now map these columns in the Column Mapping section above.")
                
                # Auto-update column mapping if not already set
                if 'earliest_time' not in st.session_state.column_mapping or not st.session_state.column_mapping.get('earliest_time'):
                    st.session_state.column_mapping['earliest_time'] = earliest_col
                if 'latest_time' not in st.session_state.column_mapping or not st.session_state.column_mapping.get('latest_time'):
                    st.session_state.column_mapping['latest_time'] = latest_col
                
                st.markdown('</div>', unsafe_allow_html=True)
                return df_updated, earliest_col, latest_col
            except Exception as e:
                st.error(f"❌ Error generating time windows: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
                return None, None, None
    
    st.markdown('</div>', unsafe_allow_html=True)
    return None, None, None

