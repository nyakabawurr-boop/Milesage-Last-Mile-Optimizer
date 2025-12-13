"""
Milesage – Last-Mile Optimizer

A Streamlit web application for optimizing last-mile delivery routes using Vehicle Routing Problem (VRP) optimization.

The app solves the Vehicle Routing Problem for last-mile delivery:
- Input: A set of delivery orders (stops) with geographic positions and optional constraints
- Outputs: Naive routing plan (simple heuristic baseline) and Optimized routing plan (using OR-Tools)
- KPIs: Total distance, total travel time, estimated total cost, % improvement from naive to optimized

How to run:
    streamlit run app.py

Dependencies:
    See requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from streamlit_folium import st_folium

# Import custom modules
try:
    from data_utils import (
        generate_synthetic_data,
        normalize_dataframe,
        build_distance_matrix,
        build_time_matrix
    )
    from vrp_solver import run_naive_solution, run_ortools_solution
    from visualization import (
        create_route_map_folium,
        create_summary_dataframe,
        generate_business_summary
    )
    from auto_config import auto_configure_parameters, run_vrp_with_auto_relaxation
except ImportError as e:
    st.error(f"Error importing modules: {e}. Please ensure all required files are present.")
    st.stop()


def inject_custom_css():
    """Inject custom CSS for modern dark-themed data analyst dashboard styling."""
    st.markdown(
        """
        <style>
        /* Main container - Dark teal background */
        .main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem 1rem;
            background-color: #1A3A3F;
        }

        /* Page background */
        .stApp {
            background-color: #1A3A3F;
        }

        /* Typography - White text for dark theme */
        h1, h2, h3, h4 {
            font-weight: 700;
            color: #FFFFFF;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }

        h1 {
            font-size: 2.25rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 0.5rem;
        }

        h2 {
            font-size: 1.75rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        h3 {
            font-size: 1.35rem;
            font-weight: 600;
            color: #FFFFFF;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }
        
        h4 {
            color: #FFFFFF;
        }

        /* Card styling - Dark cards with lighter dark background */
        .milesage-card {
            background-color: #243D42;
            border-radius: 14px;
            padding: 1.75rem 2rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
            margin-bottom: 1.5rem;
            border: 1px solid #2F4F54;
        }

        /* Subtitle styling */
        .milesage-subtitle {
            color: #FFFFFF;
            font-size: 1rem;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }

        /* Section title */
        .milesage-section-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #FFFFFF;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #2F4F54;
        }

        /* Improved spacing for paragraphs and lists */
        p {
            color: #FFFFFF;
            line-height: 1.7;
            margin-bottom: 1rem;
        }

        ul, ol {
            color: #FFFFFF;
            line-height: 1.8;
            margin-left: 1.5rem;
            margin-bottom: 1rem;
        }

        li {
            margin-bottom: 0.5rem;
            color: #FFFFFF;
        }
        
        /* Global text color - All text white by default */
        body, .stApp, div, span, label, p, li, ul, ol, td, th {
            color: #FFFFFF !important;
        }
        
        /* Override Streamlit default text colors */
        .stApp > div > div {
            color: #FFFFFF !important;
        }
        
        /* Ensure all Streamlit text elements are white */
        [class*="st"] {
            color: #FFFFFF;
        }
        
        /* Markdown content */
        .markdown-text-container, .markdown-viewer {
            color: #FFFFFF !important;
        }
        
        /* All paragraph and text nodes */
        * {
            color: inherit;
        }
        
        .milesage-card, .milesage-card * {
            color: #FFFFFF !important;
        }

        /* Streamlit component styling improvements */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s;
            background-color: #FFC107;
            color: #1A3A3F;
            border: none;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(255, 193, 7, 0.4);
            background-color: #FFD54F;
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #FFFFFF;
        }

        [data-testid="stMetricLabel"] {
            color: #FFFFFF;
            font-weight: 500;
        }

        /* Dataframe styling - Dark theme */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            background-color: #243D42;
        }
        
        .dataframe table {
            background-color: #243D42;
            color: #E5E7EB;
        }

        /* Tabs styling - Dark theme */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 2px solid #2F4F54;
            background-color: #1A3A3F;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            color: #FFFFFF;
            background-color: #243D42;
        }

        .stTabs [aria-selected="true"] {
            background-color: #2F4F54;
            color: #FFC107;
        }

        /* Info boxes - Dark theme */
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid;
            background-color: #243D42;
        }

        /* Sidebar improvements - Dark theme */
        [data-testid="stSidebar"] {
            background-color: #243D42;
            border-right: 1px solid #2F4F54;
        }

        /* Selectbox and input styling - Dark theme */
        .stSelectbox label, .stNumberInput label, .stSlider label, .stTextInput label {
            color: #FFFFFF;
            font-weight: 500;
            font-size: 0.95rem;
        }
        
        /* Input fields dark theme */
        .stSelectbox > div > div, .stNumberInput > div > div {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Number input fields - dark background */
        .stNumberInput > div > div > input,
        .stNumberInput > div > div,
        .stNumberInput input[type="number"] {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
            border: 1px solid #2F4F54 !important;
        }
        
        /* Number input buttons (increment/decrement) */
        .stNumberInput button,
        .stNumberInput > div > div > button {
            background-color: #2F4F54 !important;
            color: #FFFFFF !important;
            border: 1px solid #2F4F54 !important;
        }
        
        .stNumberInput button:hover {
            background-color: #3A5F64 !important;
        }
        
        .stTextInput > div > div > input, .stNumberInput > div > div > input {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
            border: 1px solid #2F4F54 !important;
        }
        
        /* Selectbox dropdown styling - Dark background with white text */
        .stSelectbox select,
        .stSelectbox > div > div > div,
        .stSelectbox > div > div > div > div {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Dropdown menu options container - Dark theme */
        .stSelectbox [class*="menu"],
        .stSelectbox [class*="popover"],
        .stSelectbox [class*="dropdown"],
        .stSelectbox ul,
        .stSelectbox li {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* BaseWeb select component styling */
        div[data-baseweb="select"] {
            background-color: #243D42 !important;
        }
        
        div[data-baseweb="select"] > div {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Dropdown menu items */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Selected option styling */
        .stSelectbox [aria-selected="true"],
        .stSelectbox [role="option"][aria-selected="true"] {
            background-color: #2F4F54 !important;
            color: #FFFFFF !important;
        }
        
        /* Hover state for dropdown options */
        .stSelectbox [role="option"]:hover,
        .stSelectbox li:hover {
            background-color: #2F4F54 !important;
            color: #FFFFFF !important;
        }
        
        /* Dropdown arrow/chevron icon */
        .stSelectbox svg {
            fill: #FFFFFF !important;
        }

        /* Success/Info message styling - Dark theme */
        .stSuccess {
            background-color: #1F3A1F;
            border-color: #4ADE80;
            color: #D1FAE5;
        }

        .stInfo {
            background-color: #1E3A5F;
            border-color: #60A5FA;
            color: #DBEAFE;
        }

        .stWarning {
            background-color: #3A3A1F;
            border-color: #FCD34D;
            color: #FEF3C7;
        }

        .stError {
            background-color: #3A1F1F;
            border-color: #F87171;
            color: #FEE2E2;
        }

        /* File uploader styling - Dark theme */
        .stFileUploader label {
            color: #FFFFFF;
            font-weight: 500;
        }
        
        .stFileUploader {
            color: #FFFFFF;
        }
        
        /* File uploader drop zone text - make visible */
        .stFileUploader > div > div {
            color: #FFFFFF !important;
        }
        
        .stFileUploader > div > div > div {
            color: #FFFFFF !important;
        }
        
        /* Upload area text */
        .uploadedFile {
            color: #FFFFFF !important;
        }
        
        /* File uploader container background to match cards */
        .stFileUploader > div:first-child {
            background-color: #243D42 !important;
            border-radius: 10px !important;
            padding: 0.6rem !important;
            border: 1px solid #2F4F54 !important;
        }

        /* File uploader drop zone */
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background-color: #243D42 !important;
            border: 2px dashed #2F4F54 !important;
            border-radius: 10px !important;
            color: #FFFFFF !important;
        }
        
        .stFileUploader [data-testid="stFileUploaderDropzone"] * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }
        
        .stFileUploader [data-testid="stFileUploaderDropzone"] p,
        .stFileUploader [data-testid="stFileUploaderDropzone"] span {
            color: #FFFFFF !important;
            opacity: 1 !important;
        }
        
        /* File input text and placeholder */
        .stFileUploader input {
            color: #FFFFFF !important;
            background-color: #243D42 !important;
        }

        /* Ensure any uploaded file name text stays visible */
        .uploadedFile, .uploadedFile * {
            color: #FFFFFF !important;
        }

        /* Browse files button - yellow accent */
        .stFileUploader button,
        .stFileUploader > div > button,
        .stFileUploader > div > div > button {
            background-color: #FFC107 !important;
            color: #1A3A3F !important;
            border: none !important;
            font-weight: 500 !important;
        }
        
        .stFileUploader button:hover,
        .stFileUploader > div > button:hover,
        .stFileUploader > div > div > button:hover {
            background-color: #FFD54F !important;
            color: #1A3A3F !important;
        }
        
        /* Remove any white background from buttons */
        .stFileUploader button[style*="background"] {
            background-color: #FFC107 !important;
        }
        
        /* File uploader instructions text */
        .stFileUploader > div > div > div > p,
        .stFileUploader > div > div > div > span,
        .stFileUploader > div > div > div > div {
            color: #FFFFFF !important;
        }
        
        /* More specific selectors for file uploader text */
        .stFileUploader *:not(button) {
            color: #FFFFFF !important;
        }
        
        /* Override any nested text colors in upload area */
        .stFileUploader p,
        .stFileUploader span,
        .stFileUploader div:not(button),
        .stFileUploader label {
            color: #FFFFFF !important;
            opacity: 1 !important;
            background-color: transparent !important;
        }
        
        /* File uploader dropzone specific styling - remove all white backgrounds */
        div[data-testid="stFileUploaderDropzone"] *:not(button) {
            color: #FFFFFF !important;
            opacity: 1 !important;
            background-color: transparent !important;
            background: transparent !important;
        }
        
        /* Ensure "Drag and drop file here" text is visible */
        div[data-testid="stFileUploaderDropzone"] p,
        div[data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stFileUploaderDropzone"] div:not(button) {
            color: #FFFFFF !important;
            opacity: 1 !important;
            background-color: transparent !important;
            background: transparent !important;
        }
        
        /* Force remove white backgrounds from all uploader container elements */
        .stFileUploader > div > div:not(button) {
            background: transparent !important;
            background-color: transparent !important;
        }
        
        /* Remove white backgrounds from nested divs and containers */
        .stFileUploader div div div:not(button),
        .stFileUploader div div div div:not(button) {
            background-color: transparent !important;
            background: transparent !important;
        }
        
        /* Icon containers and SVGs - make transparent */
        .stFileUploader svg,
        .stFileUploader img,
        .stFileUploader [role="img"],
        .stFileUploader [class*="icon"] {
            background-color: transparent !important;
            background: transparent !important;
        }
        
        /* Remove any inline style backgrounds */
        .stFileUploader [style*="background-color: white"],
        .stFileUploader [style*="background-color:#fff"],
        .stFileUploader [style*="background-color:#FFF"],
        .stFileUploader [style*="background: white"],
        .stFileUploader [style*="background:#fff"],
        .stFileUploader [style*="background:#FFF"] {
            background-color: transparent !important;
            background: transparent !important;
        }

        /* Additional aggressive rules to remove white backgrounds from file uploader */
        .stFileUploader [style*="background"]:not(button) {
            background-color: transparent !important;
            background: transparent !important;
        }
        
        /* Target BaseWeb uploader components specifically */
        .stFileUploader [class*="baseweb-file-uploader"],
        .stFileUploader [class*="uploader"],
        .stFileUploader [class*="dropzone"] {
            background-color: transparent !important;
            background: transparent !important;
        }
        
        /* Remove white backgrounds from any child elements */
        .stFileUploader > div > div > div:not(button),
        .stFileUploader > div > div > div > div:not(button),
        .stFileUploader > div > div > div > div > div:not(button) {
            background-color: transparent !important;
            background: transparent !important;
        }
        
        /* Icon/placeholder square - make it transparent or dark */
        .stFileUploader [class*="icon"],
        .stFileUploader [class*="Icon"],
        .stFileUploader [class*="placeholder"],
        .stFileUploader svg,
        .stFileUploader img,
        .stFileUploader [role="img"],
        .stFileUploader [class*="icon"] {
            background-color: transparent !important;
            background: transparent !important;
        }

        /* Spacing improvements */
        .element-container {
            margin-bottom: 1rem;
        }

        /* Code block styling - Dark theme */
        pre {
            border-radius: 8px;
            background-color: #1A3A3F;
            border: 1px solid #2F4F54;
            color: #E5E7EB;
        }

        /* Text elements - Dark theme - Updated for dropdowns */
        div[data-baseweb="select"] > div {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #243D42;
            color: #FFFFFF;
        }
        
        .streamlit-expanderContent {
            color: #FFFFFF;
        }
        
        /* Slider styling - Dark theme */
        .stSlider > div > div {
            color: #FFFFFF;
            background-color: #243D42 !important;
        }
        
        .stSlider label {
            color: #FFFFFF;
        }
        
        /* Slider container - dark background */
        .stSlider > div {
            background-color: #243D42 !important;
        }
        
        /* Slider track background - dark */
        .stSlider [class*="track"],
        .stSlider [class*="Track"],
        .stSlider div[role="slider"] {
            background-color: #2F4F54 !important;
        }
        
        /* Slider active track - yellow accent */
        .stSlider [class*="active"],
        .stSlider [class*="Active"],
        .stSlider [class*="filled"],
        .stSlider [class*="Filled"] {
            background-color: #FFC107 !important;
        }
        
        /* Slider thumb - yellow accent */
        .stSlider [class*="thumb"],
        .stSlider [class*="Thumb"],
        .stSlider [class*="handle"],
        .stSlider [class*="Handle"] {
            background-color: #FFC107 !important;
            border-color: #FFC107 !important;
        }
        
        /* Slider value display */
        .stSlider [class*="value"],
        .stSlider [class*="Value"] {
            color: #FFFFFF !important;
            background-color: #243D42 !important;
        }
        
        /* BaseWeb Slider component styling */
        [data-baseweb="slider"],
        [data-baseweb="slider"] > div {
            background-color: #243D42 !important;
        }
        
        [data-baseweb="slider"] [class*="track"] {
            background-color: #2F4F54 !important;
        }
        
        [data-baseweb="slider"] [class*="thumb"] {
            background-color: #FFC107 !important;
        }
        
        /* Checkbox styling - Dark theme */
        .stCheckbox label {
            color: #FFFFFF;
        }
        
        .stCheckbox > div > div {
            background-color: #243D42 !important;
        }
        
        /* Checkbox input */
        .stCheckbox input[type="checkbox"] {
            background-color: #243D42 !important;
            border: 1px solid #2F4F54 !important;
        }
        
        .stCheckbox input[type="checkbox"]:checked {
            background-color: #FFC107 !important;
            border-color: #FFC107 !important;
        }
        
        /* Comprehensive dropdown menu styling - Dark theme for all dropdowns */
        /* BaseWeb select component styling */
        [data-baseweb="select"] {
            background-color: #243D42 !important;
        }
        
        [data-baseweb="select"] > div {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Dropdown menu container (popover) */
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div {
            background-color: #243D42 !important;
            border: 1px solid #2F4F54 !important;
            color: #FFFFFF !important;
        }
        
        /* Dropdown menu list */
        [data-baseweb="popover"] ul,
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="menu"],
        [data-baseweb="menu"] ul {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Dropdown menu items */
        [role="option"],
        [data-baseweb="popover"] li,
        [data-baseweb="menu"] li,
        [role="listbox"] li {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Dropdown menu item hover state */
        [role="option"]:hover,
        [data-baseweb="popover"] li:hover,
        [data-baseweb="menu"] li:hover,
        [role="listbox"] li:hover {
            background-color: #2F4F54 !important;
            color: #FFFFFF !important;
        }
        
        /* Selected dropdown option */
        [role="option"][aria-selected="true"],
        [data-baseweb="popover"] li[aria-selected="true"] {
            background-color: #2F4F54 !important;
            color: #FFFFFF !important;
        }
        
        /* Dropdown arrow/chevron icon - white */
        [data-baseweb="select"] svg {
            fill: #FFFFFF !important;
            color: #FFFFFF !important;
        }
        
        /* Additional BaseWeb component styling for dropdowns */
        [class*="baseweb-select"],
        [class*="baseweb-menu"],
        [class*="baseweb-popover"] {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Tooltip/Hint styling - Dark theme for all tooltips */
        /* BaseWeb Tooltip component */
        [data-baseweb="tooltip"],
        [data-baseweb="popover"][role="tooltip"],
        [class*="baseweb-tooltip"],
        [class*="tooltip"],
        [class*="Tooltip"] {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
            border: 1px solid #2F4F54 !important;
        }
        
        /* Tooltip content */
        [data-baseweb="tooltip"] > div,
        [role="tooltip"],
        [role="tooltip"] > div,
        [class*="tooltip"] > div,
        [class*="Tooltip"] > div {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Tooltip text */
        [data-baseweb="tooltip"] p,
        [data-baseweb="tooltip"] span,
        [data-baseweb="tooltip"] div,
        [role="tooltip"] p,
        [role="tooltip"] span,
        [role="tooltip"] div,
        [class*="tooltip"] p,
        [class*="tooltip"] span,
        [class*="tooltip"] div {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* Streamlit help tooltips */
        .stTooltip,
        .stTooltip > div,
        .stTooltip p,
        .stTooltip span {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
            border: 1px solid #2F4F54 !important;
        }
        
        /* Help icon tooltips */
        [data-testid="stTooltip"],
        [data-testid="stTooltip"] > div,
        [data-testid="stTooltip"] p,
        [data-testid="stTooltip"] span {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* BaseWeb Popover when used as tooltip */
        [data-baseweb="popover"][role="tooltip"],
        [data-baseweb="popover"][aria-describedby] {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
            border: 1px solid #2F4F54 !important;
        }
        
        /* Override any white backgrounds in tooltip containers */
        [class*="tooltip"][style*="background"],
        [class*="Tooltip"][style*="background"],
        [role="tooltip"][style*="background"] {
            background-color: #243D42 !important;
            background: #243D42 !important;
        }
        
        /* Tooltip arrow/pointer */
        [data-baseweb="tooltip"]::before,
        [data-baseweb="tooltip"]::after,
        [role="tooltip"]::before,
        [role="tooltip"]::after {
            border-color: #243D42 transparent transparent transparent !important;
        }
        
        /* Additional tooltip/hint selectors for Streamlit */
        div[data-baseweb="popover"][id*="tooltip"],
        div[data-baseweb="popover"][id*="Tooltip"],
        div[class*="StyledPopover"][role="tooltip"],
        div[class*="StyledTooltip"] {
            background-color: #243D42 !important;
            color: #FFFFFF !important;
            border: 1px solid #2F4F54 !important;
        }
        
        /* Override white backgrounds in any popover used as tooltip */
        [data-baseweb="popover"]:has([role="tooltip"]),
        [data-baseweb="popover"]:has([class*="tooltip"]) {
            background-color: #243D42 !important;
        }
        
        /* Force dark theme for all tooltip-related elements */
        *[style*="background"][class*="tooltip"],
        *[style*="background"][class*="Tooltip"],
        *[style*="background"][role="tooltip"] {
            background-color: #243D42 !important;
            background: #243D42 !important;
            color: #FFFFFF !important;
        }
        
        /* All markdown text */
        .stMarkdown {
            color: #FFFFFF;
        }
        
        .stMarkdown p, .stMarkdown li, .stMarkdown ul, .stMarkdown ol {
            color: #FFFFFF;
        }
        
        /* Strong and bold text */
        strong, b {
            color: #FFFFFF;
        }
        
        /* DataFrame container */
        .stDataFrame {
            background-color: #243D42;
        }
        
        /* Table styling */
        table {
            background-color: #243D42;
            color: #FFFFFF;
        }
        
        thead th {
            background-color: #2F4F54;
            color: #FFFFFF;
        }
        
        tbody tr {
            background-color: #243D42;
            color: #FFFFFF;
        }
        
        tbody td {
            color: #FFFFFF;
        }
        
        tbody tr:nth-child(even) {
            background-color: #1F3539;
        }
        
        tbody tr:nth-child(even) td {
            color: #FFFFFF;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        </style>
        """,
        unsafe_allow_html=True,
    )


# Initialize session state
if 'normalized_df' not in st.session_state:
    st.session_state.normalized_df = None
if 'distance_matrix' not in st.session_state:
    st.session_state.distance_matrix = None
if 'time_matrix' not in st.session_state:
    st.session_state.time_matrix = None
if 'naive_solution' not in st.session_state:
    st.session_state.naive_solution = None
if 'optimized_solution' not in st.session_state:
    st.session_state.optimized_solution = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'config' not in st.session_state:
    st.session_state.config = {
        'cost_per_km': 1.5,
        'fixed_cost_per_vehicle': 50.0,
        'cost_per_hour': 25.0
    }


def load_data() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Handle data loading from upload or synthetic generation."""
    st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a data file",
            type=['csv', 'xlsx', 'xls', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'],
            help="Upload a CSV, Excel, or other data file with location data (latitude, longitude columns required)"
        )
    
    with col2:
        if st.button("📦 Generate Sample Synthetic Dataset", use_container_width=True):
            with st.spinner("Generating synthetic dataset..."):
                synthetic_df = generate_synthetic_data(n_customers=40, seed=42)
                st.session_state.raw_df = synthetic_df
                st.success(f"Generated {len(synthetic_df)} stops (1 depot + {len(synthetic_df)-1} customers)")
                st.rerun()
    
    if uploaded_file is not None:
        try:
            # Get file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Read file based on extension
            if file_extension in ['csv']:
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls', 'xlsm', 'xlsb']:
                # Read Excel file - try first sheet if multiple sheets exist
                df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else None)
            elif file_extension in ['ods', 'odf']:
                # Read OpenDocument Spreadsheet
                df = pd.read_excel(uploaded_file, engine='odf')
            else:
                # Try CSV as fallback
                df = pd.read_csv(uploaded_file)
            
            st.session_state.raw_df = df
            return df, None
        except Exception as e:
            return None, f"Error reading file: {str(e)}. Please ensure the file is a valid CSV or Excel file."
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.raw_df is not None:
        return st.session_state.raw_df, None
    
    return None, None


def data_filtering_ui(df: pd.DataFrame) -> pd.DataFrame:
    """UI for optional pre-filtering of datasets (non-blocking)."""
    n_rows = len(df)
    
    # Show filtering option for large datasets, but don't block
    if n_rows > 1000:
        st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Optional Data Pre-Filtering")
        st.info(
            f"📊 **Dataset size**: {n_rows:,} rows. Large datasets are supported. "
            f"The optimization will automatically sample a subset based on your settings in the Model Setup tab. "
            f"You can optionally pre-filter here if desired."
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_method = st.radio(
                "Filter Method (Optional)",
                options=["Random Sample", "First N Rows", "Last N Rows"],
                help="Optional: Choose how to pre-filter your dataset"
            )
        
        with col2:
            sample_size = st.number_input(
                f"Number of stops to keep (Optional)",
                min_value=100,
                max_value=min(10000, n_rows),
                value=min(1000, n_rows),
                step=100,
                help="Optional: Pre-filter dataset to this size"
            )
        
        with col3:
            if st.button("🔄 Apply Pre-Filter", type="primary", use_container_width=True):
                if filter_method == "Random Sample":
                    sampled_df = df.sample(n=min(sample_size, n_rows), random_state=42).reset_index(drop=True)
                elif filter_method == "First N Rows":
                    sampled_df = df.head(sample_size).copy()
                else:  # Last N Rows
                    sampled_df = df.tail(sample_size).copy()
                
                st.session_state.raw_df = sampled_df
                st.success(f"✅ Dataset pre-filtered to {len(sampled_df)} stops!")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return df


def column_mapping_ui(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """UI for mapping user columns to internal schema."""
    st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Column Mapping")
    
    if df is None or df.empty:
        st.warning("Please load data first.")
        return None, None
    
    st.dataframe(df.head(10), use_container_width=True)
    
    columns = df.columns.tolist()
    column_mapping = {}
    
    st.markdown("**Required Mappings:**")
    col1, col2 = st.columns(2)
    
    with col1:
        stop_id_col = st.selectbox(
            "Stop ID / Order ID",
            options=[None] + columns,
            index=0 if 'stop_id' not in st.session_state.column_mapping else 
                   columns.index(st.session_state.column_mapping.get('stop_id', '')) + 1 if 
                   st.session_state.column_mapping.get('stop_id') in columns else 0,
            help="Column containing unique stop/order identifiers"
        )
        column_mapping['stop_id'] = stop_id_col
    
    with col2:
        lat_col = st.selectbox(
            "Latitude",
            options=[None] + columns,
            index=0 if 'lat' not in st.session_state.column_mapping else 
                   columns.index(st.session_state.column_mapping.get('lat', '')) + 1 if 
                   st.session_state.column_mapping.get('lat') in columns else 0,
            help="Column containing latitude values"
        )
        column_mapping['lat'] = lat_col
    
    lon_col = st.selectbox(
        "Longitude",
        options=[None] + columns,
        index=0 if 'lon' not in st.session_state.column_mapping else 
               columns.index(st.session_state.column_mapping.get('lon', '')) + 1 if 
               st.session_state.column_mapping.get('lon') in columns else 0,
        help="Column containing longitude values"
    )
    column_mapping['lon'] = lon_col
    
    st.markdown("**Optional Mappings:**")
    col3, col4 = st.columns(2)
    
    with col3:
        is_depot_col = st.selectbox(
            "Is Depot? (boolean column)",
            options=[None] + columns,
            index=0 if 'is_depot' not in st.session_state.column_mapping else 
                   columns.index(st.session_state.column_mapping.get('is_depot', '')) + 1 if 
                   st.session_state.column_mapping.get('is_depot') in columns else 0,
            help="Column indicating if a stop is a depot (True/False)"
        )
        column_mapping['is_depot'] = is_depot_col
        
        demand_col = st.selectbox(
            "Demand",
            options=[None] + columns,
            index=0 if 'demand' not in st.session_state.column_mapping else 
                   columns.index(st.session_state.column_mapping.get('demand', '')) + 1 if 
                   st.session_state.column_mapping.get('demand') in columns else 0,
            help="Column containing demand/package quantities"
        )
        column_mapping['demand'] = demand_col
        
        earliest_time_col = st.selectbox(
            "Earliest Time (HH:MM)",
            options=[None] + columns,
            index=0 if 'earliest_time' not in st.session_state.column_mapping else 
                   columns.index(st.session_state.column_mapping.get('earliest_time', '')) + 1 if 
                   st.session_state.column_mapping.get('earliest_time') in columns else 0,
            help="Column containing earliest delivery time (HH:MM format)"
        )
        column_mapping['earliest_time'] = earliest_time_col
    
    with col4:
        latest_time_col = st.selectbox(
            "Latest Time (HH:MM)",
            options=[None] + columns,
            index=0 if 'latest_time' not in st.session_state.column_mapping else 
                   columns.index(st.session_state.column_mapping.get('latest_time', '')) + 1 if 
                   st.session_state.column_mapping.get('latest_time') in columns else 0,
            help="Column containing latest delivery time (HH:MM format)"
        )
        column_mapping['latest_time'] = latest_time_col
        
        service_time_col = st.selectbox(
            "Service Time (minutes)",
            options=[None] + columns,
            index=0 if 'service_time' not in st.session_state.column_mapping else 
                   columns.index(st.session_state.column_mapping.get('service_time', '')) + 1 if 
                   st.session_state.column_mapping.get('service_time') in columns else 0,
            help="Column containing service duration per stop in minutes"
        )
        column_mapping['service_time'] = service_time_col
    
    # Depot selection if no is_depot column
    depot_stop_id = None
    if not is_depot_col:
        if stop_id_col:
            unique_stop_ids = df[stop_id_col].unique().tolist()
            depot_stop_id = st.selectbox(
                "Select Depot Stop ID",
                options=[None] + unique_stop_ids,
                help="Manually select which stop ID represents the depot"
            )
    
    st.session_state.column_mapping = column_mapping
    
    # Normalize data
    if stop_id_col and lat_col and lon_col:
        n_rows = len(df)
        
        # Inform about large datasets but don't block
        if n_rows > 1000:
            st.info(
                f"📊 **Dataset size**: {n_rows:,} stops. Large datasets are fully supported. "
                f"The optimizer will automatically sample a subset based on your 'Maximum stops' setting in the Model Setup tab."
            )
        
        if st.button("✅ Validate & Normalize Data", type="primary", use_container_width=True):
            with st.spinner("Normalizing data..."):
                normalized_df, error = normalize_dataframe(df, column_mapping, depot_stop_id)
                
                if error:
                    st.error(f"❌ {error}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return None, error
                else:
                    n_stops = len(normalized_df)
                    
                    st.session_state.normalized_df = normalized_df
                    st.success(f"✅ Data normalized successfully! {len(normalized_df):,} stops loaded.")
                    st.session_state.distance_matrix = None
                    st.session_state.time_matrix = None
                    st.session_state.naive_solution = None
                    st.session_state.optimized_solution = None
                    st.session_state.sampling_info = None  # Reset sampling info
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.normalized_df is not None:
        st.markdown("**✅ Normalized Data Preview:**")
        st.dataframe(st.session_state.normalized_df.head(10), use_container_width=True)
        return st.session_state.normalized_df, None
    
    return None, None


def build_matrices(df: pd.DataFrame, speed_kmh: float, max_stops: int = 500):
    """
    Build distance and time matrices for the provided DataFrame.
    
    Args:
        df: DataFrame with stops (should already be sampled if needed)
        speed_kmh: Average vehicle speed
        max_stops: Maximum stops (for info display only, df should already be subset)
    """
    if df is None:
        return
    
    if st.session_state.distance_matrix is None:
        n = len(df)
        
        if n > 100:
            st.info(
                f"📊 Building distance matrix for {n:,} stops. This may take a moment..."
            )
        
        with st.spinner(f"Building distance matrix for {n:,} stops (this may take a moment)..."):
            try:
                st.session_state.distance_matrix = build_distance_matrix(df)
                st.session_state.time_matrix = build_time_matrix(st.session_state.distance_matrix, speed_kmh)
                st.success(f"✅ Distance and time matrices built successfully for {n:,} stops!")
            except MemoryError:
                st.error(
                    f"❌ **Memory error**: Unable to allocate memory for distance matrix with {n:,} stops. "
                    f"Please reduce the 'Maximum stops' setting in Model Setup."
                )
                st.session_state.distance_matrix = None
                st.session_state.time_matrix = None


def show_model_setup() -> Dict:
    """UI for model setup and configuration."""
    st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Model Configuration")
    
    if st.session_state.normalized_df is None:
        st.warning("Please load and normalize data first.")
        return {}
    
    df = st.session_state.normalized_df
    
    config = {}
    
    # Configuration Mode selector
    config_mode = st.radio(
        "Configuration Mode",
        options=["Automatic (recommended)", "Manual"],
        index=0,
        help="Automatic mode chooses fleet and constraint parameters for you. "
             "Manual mode lets you set all parameters yourself."
    )
    st.session_state.config_mode = config_mode
    
    # Auto-configure if in Automatic mode
    auto_config = None
    if config_mode == "Automatic (recommended)":
        # Determine which columns exist
        has_demand = df['demand'].sum() > 0 if 'demand' in df.columns else False
        has_earliest = df['earliest_time'].notna().any() if 'earliest_time' in df.columns else False
        has_latest = df['latest_time'].notna().any() if 'latest_time' in df.columns else False
        
        auto_config = auto_configure_parameters(
            df,
            demand_column='demand' if has_demand else None,
            earliest_time_column='earliest_time' if has_earliest else None,
            latest_time_column='latest_time' if has_latest else None,
        )
        
        st.info(
            "🤖 **Automatic mode**: Analyzes your dataset and suggests feasible configuration parameters. "
            "You can still adjust the values below before running optimization."
        )
    else:
        st.info("⚙️ **Manual mode**: Set all parameters yourself. Adjust values as needed.")
    
    st.markdown("**Fleet Parameters:**")
    col1, col2 = st.columns(2)
    
    with col1:
        # Number of vehicles
        if auto_config:
            default_n_vehicles = auto_config['n_vehicles']
        else:
            default_n_vehicles = 5
        
        config['n_vehicles'] = st.number_input(
            "Number of Vehicles",
            min_value=1,
            max_value=50,
            value=default_n_vehicles,
            help="Maximum number of vehicles available"
        )
        
        # Vehicle capacity
        has_demand = df['demand'].sum() > 0 if 'demand' in df.columns else False
        if has_demand:
            if auto_config:
                default_capacity = auto_config['vehicle_capacity']
            else:
                default_capacity = float(df['demand'].sum() / config['n_vehicles'] * 1.5)
            
            config['vehicle_capacity'] = st.number_input(
                "Vehicle Capacity",
                min_value=1.0,
                value=default_capacity,
                step=1.0,
                help="Maximum capacity per vehicle (units/packages)"
            )
        else:
            config['vehicle_capacity'] = None
    
    with col2:
        # Max route duration
        if auto_config:
            default_max_hours = auto_config['max_route_duration_hours']
        else:
            default_max_hours = 8.0
        
        config['max_route_duration_hours'] = st.slider(
            "Maximum Route Duration (hours)",
            min_value=1.0,
            max_value=12.0,
            value=default_max_hours,
            step=0.5,
            help="Maximum duration for each vehicle route"
        )
        
        # Depot hours
        if auto_config:
            default_open = auto_config['depot_open_hour']
            default_close = auto_config['depot_close_hour']
        else:
            default_open = 8
            default_close = 20
        
        depot_start_hour = st.number_input(
            "Depot Opening Hour",
            min_value=0,
            max_value=23,
            value=default_open,
            step=1,
            help="Depot opening hour (0-23)"
        )
        depot_end_hour = st.number_input(
            "Depot Closing Hour",
            min_value=0,
            max_value=23,
            value=default_close,
            step=1,
            help="Depot closing hour (0-23)"
        )
        config['depot_time_window'] = (depot_start_hour * 60, depot_end_hour * 60)
        config['depot_open_hour'] = depot_start_hour
        config['depot_close_hour'] = depot_end_hour
    
    # Average speed
    if auto_config:
        default_speed = auto_config['speed_kmh']
    else:
        default_speed = 40
    
    config['speed_kmh'] = st.slider(
        "Average Vehicle Speed (km/h)",
        min_value=10,
        max_value=80,
        value=int(default_speed),
        step=5,
        help="Average vehicle speed for travel time calculation"
    )
    
    # Rebuild matrices if speed changed
    if st.session_state.distance_matrix is not None:
        st.session_state.time_matrix = build_time_matrix(
            st.session_state.distance_matrix, 
            config['speed_kmh']
        )
    
    st.markdown("**Cost Parameters:**")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        config['fixed_cost_per_vehicle'] = st.number_input(
            "Fixed Cost per Vehicle ($)",
            min_value=0.0,
            value=50.0,
            step=5.0
        )
    
    with col4:
        config['cost_per_km'] = st.number_input(
            "Variable Cost per km ($)",
            min_value=0.0,
            value=1.5,
            step=0.1
        )
    
    with col5:
        config['cost_per_hour'] = st.number_input(
            "Cost per Hour ($)",
            min_value=0.0,
            value=25.0,
            step=1.0
        )
    
    st.markdown("**Constraints:**")
    col6, col7 = st.columns(2)
    
    with col6:
        # Use capacity checkbox
        if auto_config:
            default_use_capacity = auto_config['use_capacity']
        else:
            default_use_capacity = has_demand
        
        config['use_capacity'] = st.checkbox(
            "Use Capacity Constraints",
            value=default_use_capacity,
            disabled=not has_demand,
            help="Enable capacity constraints (only available if demand column is mapped)"
        )
    
    with col7:
        # Use time windows checkbox
        has_time_windows_data = df['earliest_time'].notna().any() or df['latest_time'].notna().any() if 'earliest_time' in df.columns and 'latest_time' in df.columns else False
        
        if auto_config:
            default_use_time_windows = auto_config['use_time_windows']
        else:
            default_use_time_windows = has_time_windows_data
        
        config['use_time_windows'] = st.checkbox(
            "Use Time Window Constraints",
            value=default_use_time_windows,
            disabled=not has_time_windows_data,
            help="Enable time window constraints (only available if time columns are mapped)"
        )
    
    st.markdown("**Solver Settings:**")
    col8, col9, col10 = st.columns(3)
    
    with col8:
        config['first_solution_strategy'] = st.selectbox(
            "First Solution Strategy",
            options=['PATH_CHEAPEST_ARC', 'PATH_MOST_CONSTRAINED_ARC', 'SAVINGS', 'SWEEP', 'CHRISTOFIDES'],
            index=0,
            help="Initial solution strategy for OR-Tools"
        )
    
    with col9:
        config['local_search_metaheuristic'] = st.selectbox(
            "Local Search Metaheuristic",
            options=['GUIDED_LOCAL_SEARCH', 'TABU_SEARCH', 'SIMULATED_ANNEALING', 'None'],
            index=0,
            help="Metaheuristic for local search improvement"
        )
    
    with col10:
        config['time_limit_seconds'] = st.number_input(
            "Solver Time Limit (seconds)",
            min_value=1,
            max_value=300,
            value=30,
            step=5
        )
    
    # Sample dataset if needed before building matrices
    max_stops = config.get('max_stops_for_optimization', 500)
    sampling_strategy = config.get('sampling_strategy', 'Random Sample')
    n_total = len(df)
    
    # Create subset for optimization if needed
    # IMPORTANT: Always include the depot in the subset
    if n_total > max_stops:
        # Separate depot and customers
        depot_df = df[df['is_depot']].copy()
        customer_df = df[~df['is_depot']].copy()
        n_depots = len(depot_df)
        
        # Calculate how many customers we can include (max_stops - n_depots)
        n_customers_needed = max(1, max_stops - n_depots)  # At least 1 customer
        
        if len(customer_df) > n_customers_needed:
            if sampling_strategy == "Random Sample":
                customer_subset = customer_df.sample(n=n_customers_needed, random_state=42).reset_index(drop=True)
            else:  # First N Rows
                customer_subset = customer_df.head(n_customers_needed).copy().reset_index(drop=True)
        else:
            customer_subset = customer_df.copy()
        
        # Combine depot(s) and sampled customers
        df_subset = pd.concat([depot_df, customer_subset], ignore_index=True).reset_index(drop=True)
        
        # Store sampling info for display in Results
        st.session_state.sampling_info = {
            'total_stops': n_total,
            'sampled_stops': len(df_subset),
            'strategy': sampling_strategy
        }
    else:
        df_subset = df.copy()
        st.session_state.sampling_info = None
    
    # Store the working DataFrame (subset if sampled, full if not)
    st.session_state.working_df = df_subset
    
    # Clear matrices if dataset changed (different size or sampling)
    if (st.session_state.distance_matrix is not None and 
        st.session_state.get('working_df_size') != len(df_subset)):
        st.session_state.distance_matrix = None
        st.session_state.time_matrix = None
        st.session_state.naive_solution = None
        st.session_state.optimized_solution = None
    
    st.session_state.working_df_size = len(df_subset)
    
    # Build matrices for the working DataFrame (subset if sampled)
    build_matrices(df_subset, config['speed_kmh'], max_stops=max_stops)
    
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        if st.session_state.distance_matrix is None:
            st.error("Please wait for distance matrix to be built.")
            return config
        
        # Use the working DataFrame (subset if sampled)
        working_df = st.session_state.working_df
        
        with st.spinner("Running naive solution..."):
            naive_sol = run_naive_solution(
                working_df,
                st.session_state.distance_matrix,
                st.session_state.time_matrix,
                config['n_vehicles'],
                config['vehicle_capacity'] if config.get('use_capacity', False) else None,
                config['max_route_duration_hours'],
                config['depot_time_window']
            )
            st.session_state.naive_solution = naive_sol
        
        with st.spinner("Running OR-Tools optimization (this may take a moment)..."):
            try:
                # Use automatic relaxation if in Automatic mode
                config_mode = st.session_state.get('config_mode', 'Manual')
                if config_mode == "Automatic (recommended)":
                    opt_sol, relaxed_config, relaxation_info = run_vrp_with_auto_relaxation(
                        run_ortools_solution,
                        working_df,
                        st.session_state.distance_matrix,
                        st.session_state.time_matrix,
                        config
                    )
                    
                    # Update config with relaxed values
                    if relaxation_info and relaxation_info.get('relaxed'):
                        config.update(relaxed_config)
                    
                    if 'error' in opt_sol:
                        st.error(f"❌ Optimization failed: {opt_sol['error']}")
                        st.info("💡 Try switching to Manual mode and adjusting parameters, or reduce the number of stops.")
                    else:
                        # Show relaxation info if any constraints were relaxed
                        if relaxation_info and relaxation_info.get('relaxed'):
                            relax_msg = "🤖 **Automatic configuration**: Relaxed some constraints to find a feasible solution:"
                            relax_details = []
                            
                            if relaxation_info.get('capacity_factor'):
                                relax_details.append(f"Capacity increased by {relaxation_info['capacity_factor']:.1f}x (new: {relaxation_info['new_capacity']:.1f})")
                            if relaxation_info.get('capacity_disabled'):
                                relax_details.append("Capacity constraints disabled")
                            if relaxation_info.get('time_windows_disabled'):
                                relax_details.append("Time window constraints disabled")
                            if relaxation_info.get('max_route_hours'):
                                relax_details.append(f"Max route duration: {relaxation_info['max_route_hours']:.1f} hours")
                            if relaxation_info.get('depot_close_hour'):
                                relax_details.append(f"Depot closing hour: {relaxation_info['depot_close_hour']}:00")
                            
                            st.warning(f"{relax_msg}\n- " + "\n- ".join(relax_details))
                        
                        st.session_state.optimized_solution = opt_sol
                        st.success("✅ Optimization completed successfully!")
                        st.rerun()
                else:
                    # Manual mode: no automatic relaxation
                    opt_sol = run_ortools_solution(
                        working_df,
                        st.session_state.distance_matrix,
                        st.session_state.time_matrix,
                        config['n_vehicles'],
                        config['vehicle_capacity'] if config.get('use_capacity', False) else None,
                        config['max_route_duration_hours'],
                        config['depot_time_window'],
                        config['first_solution_strategy'],
                        config['local_search_metaheuristic'],
                        config['time_limit_seconds']
                    )
                    
                    if 'error' in opt_sol:
                        st.error(f"❌ Optimization failed: {opt_sol['error']}")
                        st.info("💡 Try relaxing capacity, time windows, or maximum route duration constraints.")
                    else:
                        st.session_state.optimized_solution = opt_sol
                        st.success("✅ Optimization completed successfully!")
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Error running optimization: {str(e)}")
                st.info("💡 Make sure ortools is installed: pip install ortools")
    
    # Store config in session state
    st.session_state.config = config
    st.markdown('</div>', unsafe_allow_html=True)
    return config


def show_results(config: Dict):
    """Display results and visualizations."""
    st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Results & Visualization")
    st.markdown('</div>', unsafe_allow_html=True)
    
    naive_sol = st.session_state.naive_solution
    opt_sol = st.session_state.optimized_solution
    
    if naive_sol is None or opt_sol is None:
        st.info("👈 Please run optimization first in the Model Setup tab.")
        return
    
    if 'error' in opt_sol:
        st.error("Optimization failed. Please check constraints.")
        return
    
    # Sampling info (if dataset was sampled)
    sampling_info = st.session_state.get('sampling_info')
    if sampling_info:
        st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
        st.info(
            f"📊 **Dataset Sampling**: Optimization ran on {sampling_info['sampled_stops']:,} stops "
            f"out of {sampling_info['total_stops']:,} total stops "
            f"using the '{sampling_info['strategy']}' strategy. "
            f"Adjust the 'Maximum stops' setting in Model Setup to change the sample size."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # KPI Comparison
    st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
    st.markdown("### 📊 KPI Comparison")
    summary_df = create_summary_dataframe(naive_sol, opt_sol)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Business Summary
    if 'cost_per_km' in config:
        st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
        business_summary = generate_business_summary(
            naive_sol,
            opt_sol,
            config['cost_per_km'],
            config['fixed_cost_per_vehicle'],
            config.get('cost_per_hour', 0)
        )
        st.markdown(business_summary)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Solution tabs
    tab1, tab2 = st.tabs(["🗺️ Route Map", "📋 Route Details"])
    
    with tab1:
        st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
        st.markdown("#### Route Map")
        
        # Route selection toggle
        if st.session_state.naive_solution is not None and st.session_state.optimized_solution is not None:
            route_type = st.radio(
                "Select route visualization:",
                options=["Optimized Routes", "Naive Routes"],
                index=0,
                horizontal=True
            )
            
            if route_type == "Optimized Routes":
                selected_solution = opt_sol
                solution_name = "Optimized Solution"
            else:
                selected_solution = naive_sol
                solution_name = "Naive Solution"
        else:
            selected_solution = opt_sol
            solution_name = "Optimized Solution"
        
        # Fullscreen map toggle
        full_screen = st.checkbox(
            "🗺️ Full screen map",
            value=False,
            help="Expand the map to use more of the screen height for better route visualization."
        )
        
        # Adjust padding in fullscreen mode
        if full_screen:
            st.markdown(
                """
                <style>
                .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0.5rem;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        
        # Set map height based on fullscreen toggle
        map_height = 900 if full_screen else 650
        
        # Use working_df if available (subset), otherwise normalized_df (full)
        df_for_map = st.session_state.get('working_df', st.session_state.normalized_df)
        if df_for_map is not None:
            try:
                route_map = create_route_map_folium(
                    df_for_map,
                    selected_solution,
                    solution_name
                )
                # Render Folium map with responsive width and variable height
                st_folium(
                    route_map,
                    use_container_width=True,
                    height=map_height,
                    returned_objects=[]
                )
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
                st.info("💡 Make sure folium and streamlit-folium are installed: pip install folium streamlit-folium")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
        st.markdown("#### Route Details")
        
        solution_tabs = st.tabs(["Optimized Solution", "Naive Solution"])
        
        for solution_name, solution in [("Optimized", opt_sol), ("Naive", naive_sol)]:
            with solution_tabs[0] if solution_name == "Optimized" else solution_tabs[1]:
                route_details = []
                for route_detail in solution['route_details']:
                    route_details.append({
                        'Vehicle ID': route_detail['vehicle_id'],
                        'Number of Stops': route_detail['n_stops'],
                        'Distance (km)': f"{route_detail['distance']:.2f}",
                        'Duration (hours)': f"{route_detail['time'] / 60:.2f}",
                        'Demand': f"{route_detail['demand']:.0f}" if 'demand' in route_detail else "N/A"
                    })
                
                if route_details:
                    route_df = pd.DataFrame(route_details)
                    st.dataframe(route_df, use_container_width=True, hide_index=True)
                    
                    # Show detailed stop sequences
                    with st.expander(f"View Detailed Stop Sequences for {solution_name} Solution"):
                        # Use working_df if available (subset), otherwise normalized_df (full)
                        df_for_details = st.session_state.get('working_df', st.session_state.normalized_df)
                        for route_detail in solution['route_details']:
                            st.markdown(f"**Vehicle {route_detail['vehicle_id']}:**")
                            stop_ids = [df_for_details.loc[idx, 'stop_id'] 
                                       for idx in route_detail['stops']]
                            st.write(" → ".join(stop_ids))
                            st.write("---")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Page configuration (must be first)
    st.set_page_config(
        page_title="Milesage – Last-Mile Optimizer",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Title with improved styling
    st.markdown(
        """
        <div style="text-align: left; margin-bottom: 2rem;">
            <h1 style="margin-bottom: 0.25rem;">🚚 Milesage – Last-Mile Optimizer</h1>
            <p class="milesage-subtitle">Route planning & cost minimizer for last-mile delivery</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Overview",
        "📊 Data & Mapping",
        "⚙️ Model Setup",
        "📈 Results & Visualization"
    ])
    
    with tab1:
        st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
        st.markdown("### Overview")
        st.markdown("""
        **Milesage** is a powerful tool for optimizing last-mile delivery routes using advanced 
        Vehicle Routing Problem (VRP) optimization techniques.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
            st.markdown("#### What It Does")
            st.markdown("""
            Milesage helps transportation and logistics companies reduce costs and improve efficiency by:
            - **Optimizing delivery routes** to minimize total distance and travel time
            - **Comparing routing strategies** (naive vs optimized) to quantify improvements
            - **Supporting real-world constraints** like vehicle capacity, time windows, and route duration limits
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
            st.markdown("#### Key Features")
            st.markdown("""
            - ✅ **Flexible data input**: Upload any CSV file with location data, or use synthetic sample data
            - ✅ **Intelligent column mapping**: Map your dataset columns to the required schema
            - ✅ **Advanced optimization**: Uses Google OR-Tools for state-of-the-art VRP solving
            - ✅ **Interactive visualization**: View routes on an interactive map
            - ✅ **Business metrics**: See cost savings and efficiency improvements
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
        st.markdown("#### How It Works")
        st.markdown("""
        1. **Load Data**: Upload your CSV/Excel file or generate sample data
        2. **Map Columns**: Tell Milesage which columns contain location and constraint data
        3. **Configure Model**: Set fleet parameters, costs, and constraints
        4. **Run Optimization**: Compare naive (manual-style) routing vs optimized routing
        5. **View Results**: Analyze improvements and visualize routes on a map
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Demo video card
        st.markdown('<div class="milesage-card">', unsafe_allow_html=True)
        st.markdown("#### Demo: Using Your Own Dataset")
        st.markdown("Paste a video URL (YouTube/MP4) showing how to use Milesage with your data. A default demo is provided.")
        default_video_url = "https://www.youtube.com/watch?v=R2nr1uZ8ffc"  # Streamlit intro (replace with your demo)
        video_url = st.text_input("Demo video URL", value=default_video_url, key="demo_video_url")
        if video_url:
            st.video(video_url)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("💡 **Getting Started**: Navigate to the **Data & Mapping** tab to begin!")
    
    with tab2:
        df, error = load_data()
        if error:
            st.error(f"❌ {error}")
        else:
            # Show data filtering UI if dataset is large (before column mapping)
            current_df = df if df is not None else st.session_state.raw_df
            # Show optional pre-filtering UI for large datasets (non-blocking)
            if current_df is not None:
                current_df = data_filtering_ui(current_df)
            
            # Then show column mapping
            normalized_df, norm_error = column_mapping_ui(current_df)
            if norm_error:
                st.error(f"❌ {norm_error}")
    
    with tab3:
        config = show_model_setup()
    
    with tab4:
        show_results(st.session_state.config)


if __name__ == "__main__":
    main()

