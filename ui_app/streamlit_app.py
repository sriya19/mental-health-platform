# streamlit_app.py - Complete Updated Version with Data Indexing
"""
Mindcube Data Map US - Mental Health Metadata Catalog Insight Generator
Complete Working Streamlit Application with Data Indexing for RAG
"""
from __future__ import annotations

import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import time
from typing import Optional, Dict, Any, List
import json
from openai import OpenAI

# Load environment variables from .env file
from dotenv import load_dotenv
from pathlib import Path

# Load .env from parent directory (project root)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Optional visualization libraries
try:
    import pydeck as pdk
    HAVE_PYDECK = True
except ImportError:
    HAVE_PYDECK = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

# ===========================
# Configuration
# ===========================
st.set_page_config(
    page_title="Mindcube Data Map US - Mental Health Insights",
    page_icon="🧠",
    layout="wide",
)

# Backend configuration - Use environment variables or defaults
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_ORG = os.getenv("DEFAULT_ORG", "CDC")

# Personas
PERSONAS = [
    "Public health researcher",
    "Policy maker",
    "Clinician",
    "Epidemiologist",
    "Data analyst",
]

# ===========================
# Session State Management
# ===========================
if "results" not in st.session_state:
    st.session_state.results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = {}
if "ingested_datasets" not in st.session_state:
    st.session_state.ingested_datasets = set()
if "indexed_datasets" not in st.session_state:
    st.session_state.indexed_datasets = set()
if "datasets_loaded" not in st.session_state:
    st.session_state.datasets_loaded = False
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if 'show_viz' not in st.session_state:
    st.session_state.show_viz = False
if 'viz_data' not in st.session_state:
    st.session_state.viz_data = None
if 'viz_name' not in st.session_state:
    st.session_state.viz_name = ""
if 'viz_org' not in st.session_state:
    st.session_state.viz_org = None
if 'viz_uid' not in st.session_state:
    st.session_state.viz_uid = None
if 'viz_full_data' not in st.session_state:
    st.session_state.viz_full_data = None
# ===========================
# Helper Functions
# ===========================

def call_backend(method: str, endpoint: str, **kwargs):
    """Make API call to backend with error handling"""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        response = requests.request(method, url, timeout=90, **kwargs)
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text
            return None, f"Error {response.status_code}: {error_detail}"
        return response.json(), None
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Please check if the server is running."
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=900)
def preview_dataset(org: str, uid: str, rows: int = 200) -> Optional[pd.DataFrame]:
    """Preview dataset from backend"""
    # Try the preview endpoint
    data, error = call_backend(
        "GET", 
        f"/datasets/preview?org={org}&uid={uid}&rows={rows}"
    )
    
    if not error and data:
        sample_data = data.get("sample", [])
        if sample_data:
            return pd.DataFrame(sample_data)
    
    # Fallback to quick preview
    data, error = call_backend(
        "GET",
        f"/datasets/quick_preview?org={org}&uid={uid}&rows={rows}"
    )
    
    if not error and data:
        records = data.get("records", [])
        if records:
            return pd.DataFrame(records)
    
    return None

def load_existing_datasets():
    """
    Load ingested datasets from backend database.
    Note: We don't load indexed status - that's tracked only within the current session.
    """
    try:
        # Fetch all datasets from backend
        data, error = call_backend("GET", "/datasets?org=All")

        if error or not data:
            print(f"[UI] Failed to load datasets: {error}")
            return

        datasets = data.get("items", [])

        # Extract UIDs of ingested datasets only
        ingested_uids = set()

        for dataset in datasets:
            uid = dataset.get("uid")
            if uid and uid != "None":
                # All datasets in the database are considered ingested
                ingested_uids.add(uid)

        # Update session state - only ingested, not indexed
        st.session_state.ingested_datasets = ingested_uids
        # Keep indexed_datasets as empty set (tracked only in current session)
        st.session_state.datasets_loaded = True

        print(f"[UI] Loaded {len(ingested_uids)} ingested datasets")

    except Exception as e:
        print(f"[UI] Error loading datasets: {e}")

def analyze_dataset_with_ai(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Use AI to analyze the dataset and suggest meaningful visualizations
    """
    try:
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OpenAI API key not configured"}

        client = OpenAI(api_key=api_key)

        # Prepare dataset summary for AI
        summary = {
            "name": dataset_name,
            "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "columns": []
        }

        # Add column information
        for col in df.columns[:30]:  # Limit to first 30 columns
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "sample_values": []
            }

            # Add sample values (avoid unhashable types)
            try:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    # Get up to 5 sample values
                    samples = non_null.head(5).tolist()
                    # Convert to string if necessary
                    col_info["sample_values"] = [str(s)[:100] for s in samples]
                    col_info["unique_count"] = int(df[col].nunique()) if df[col].nunique() < 10000 else "10000+"
            except:
                col_info["sample_values"] = ["Complex type"]

            summary["columns"].append(col_info)

        # Create prompt for AI
        prompt = f"""Analyze this dataset and suggest the 3-5 most meaningful visualizations.

Dataset: {dataset_name}
Shape: {summary['shape']}

Columns:
{json.dumps(summary['columns'], indent=2)}

Please provide:
1. A brief analysis of what this dataset represents
2. 3-5 specific visualization recommendations with:
   - Chart type (histogram, bar, line, scatter, heatmap, box, pie)
   - Which column(s) to use
   - For heatmaps: specify x_column (e.g., year), y_column (e.g., state/category), and optionally value_column (numeric metric to display)
   - Why this visualization is meaningful
   - What insights it reveals
3. Key patterns or insights you notice in the data

Chart type guidance:
- histogram: Distribution of single numeric variable
- bar: Compare categories or groups
- line: Show trends over time
- scatter: Show relationship between two numeric variables
- heatmap: Show patterns across two categorical/ordinal dimensions with a numeric value (great for geographic + temporal data)
- box: Show statistical distribution and outliers
- pie: Show proportions (use sparingly)

Format your response as JSON:
{{
  "analysis": "Brief description of the dataset",
  "visualizations": [
    {{
      "chart_type": "bar",
      "x_column": "column_name",
      "y_column": "column_name" (optional for some chart types),
      "value_column": "numeric_column" (optional, for heatmaps),
      "title": "Chart title",
      "description": "What this shows and why it matters",
      "insight": "Key insight from this visualization"
    }}
  ],
  "key_insights": ["insight 1", "insight 2", "insight 3"]
}}"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst expert specializing in health and public policy data. Provide practical, actionable visualization recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        # Parse response
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        return {"error": str(e)}

def create_ai_chart(df: pd.DataFrame, viz_config: Dict[str, Any]):
    """
    Create a chart based on AI recommendations
    """
    try:
        chart_type = viz_config.get("chart_type", "").lower()
        x_col = viz_config.get("x_column")
        y_col = viz_config.get("y_column")
        title = viz_config.get("title", "Chart")

        # Validate columns exist
        if x_col and x_col not in df.columns:
            st.warning(f"Column '{x_col}' not found in dataset")
            return

        if y_col and y_col not in df.columns:
            st.warning(f"Column '{y_col}' not found in dataset")
            return

        # Create the chart
        if chart_type == "histogram":
            if HAVE_PLOTLY and x_col:
                fig = px.histogram(df.dropna(subset=[x_col]), x=x_col, title=title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df[x_col].value_counts().head(20))

        elif chart_type == "bar":
            if x_col and y_col and HAVE_PLOTLY:
                # Group and aggregate
                grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(20)
                fig = px.bar(x=grouped.index, y=grouped.values, title=title,
                           labels={'x': x_col, 'y': f'Average {y_col}'})
                st.plotly_chart(fig, use_container_width=True)
            elif x_col:
                # Simple value counts
                counts = df[x_col].value_counts().head(20)
                if HAVE_PLOTLY:
                    fig = px.bar(x=counts.index, y=counts.values, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(counts)

        elif chart_type == "line":
            if x_col and y_col:
                grouped = df.groupby(x_col)[y_col].mean().sort_index()
                if HAVE_PLOTLY:
                    fig = px.line(x=grouped.index, y=grouped.values, title=title,
                                labels={'x': x_col, 'y': y_col})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(grouped)

        elif chart_type == "scatter":
            if x_col and y_col and HAVE_PLOTLY:
                fig = px.scatter(df.dropna(subset=[x_col, y_col]), x=x_col, y=y_col, title=title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.scatter_chart(df[[x_col, y_col]].dropna())

        elif chart_type == "box":
            if x_col and HAVE_PLOTLY:
                fig = px.box(df.dropna(subset=[x_col]), y=x_col, title=title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(df[x_col].describe())

        elif chart_type == "pie":
            if x_col:
                counts = df[x_col].value_counts().head(10)
                if HAVE_PLOTLY:
                    fig = px.pie(values=counts.values, names=counts.index, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(counts)

        elif chart_type == "heatmap":
            if x_col and y_col and HAVE_PLOTLY:
                # For heatmaps, we need to aggregate data into a matrix
                # Use the first numeric column as the value, or use count
                try:
                    # Find a numeric column to use for values
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

                    # Try to find the value column from viz_config
                    value_col = None
                    if 'value_column' in viz_config:
                        value_col = viz_config['value_column']
                    elif numeric_cols:
                        # Use the first numeric column that's not x or y
                        for col in numeric_cols:
                            if col not in [x_col, y_col]:
                                value_col = col
                                break
                        if not value_col:
                            value_col = numeric_cols[0]

                    if value_col:
                        # Create pivot table for heatmap
                        # Limit to top 20 categories for each axis for readability
                        top_x = df[x_col].value_counts().head(20).index
                        top_y = df[y_col].value_counts().head(20).index

                        filtered_df = df[df[x_col].isin(top_x) & df[y_col].isin(top_y)]

                        pivot_data = filtered_df.pivot_table(
                            values=value_col,
                            index=y_col,
                            columns=x_col,
                            aggfunc='mean'
                        )

                        fig = px.imshow(
                            pivot_data,
                            title=title,
                            labels=dict(x=x_col, y=y_col, color=value_col),
                            aspect="auto",
                            color_continuous_scale='RdYlBu_r'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fall back to count-based heatmap
                        crosstab = pd.crosstab(df[y_col], df[x_col])
                        # Limit to top 20x20
                        if len(crosstab) > 20:
                            crosstab = crosstab.head(20)
                        if len(crosstab.columns) > 20:
                            crosstab = crosstab[crosstab.columns[:20]]

                        fig = px.imshow(
                            crosstab,
                            title=title,
                            labels=dict(x=x_col, y=y_col, color="Count"),
                            aspect="auto",
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Could not create heatmap: {e}")
                    # Show a simpler alternative
                    st.info("Showing crosstab instead:")
                    crosstab = pd.crosstab(df[y_col], df[x_col])
                    st.dataframe(crosstab.head(20))
            else:
                if not HAVE_PLOTLY:
                    st.warning("Heatmaps require plotly. Install with: pip install plotly")
                else:
                    st.warning("Heatmap requires both x_column and y_column")
        else:
            st.info(f"Chart type '{chart_type}' not yet supported")

    except Exception as e:
        st.error(f"Error creating chart: {e}")

def show_visualization_page():
    """Display the visualization page"""

    # Back button
    if st.button("⬅️ Back to Results"):
        st.session_state.show_viz = False
        st.session_state.viz_full_data = None  # Clear full data when going back
        st.rerun()

    # Display visualizations in full width
    st.header(f"📊 Visualizations for {st.session_state.viz_name}")

    # Determine which dataset to use
    using_full_data = st.session_state.viz_full_data is not None
    df = st.session_state.viz_full_data if using_full_data else st.session_state.viz_data

    # Show data status and load button
    col1, col2 = st.columns([3, 1])
    with col1:
        if using_full_data:
            st.success(f"✅ **Showing FULL dataset:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        else:
            st.warning(f"⚠️ **Showing PREVIEW only:** {df.shape[0]:,} rows × {df.shape[1]} columns (statistics are based on preview)")
    with col2:
        if not using_full_data:
            if st.button("📂 Load Full Dataset", type="primary", use_container_width=True):
                with st.spinner("Loading complete dataset... This may take a moment for large datasets."):
                    # Fetch full dataset (no row limit)
                    full_df = preview_dataset(
                        st.session_state.viz_org,
                        st.session_state.viz_uid,
                        100000  # High limit for full data
                    )
                    if full_df is not None:
                        st.session_state.viz_full_data = full_df
                        st.success(f"✅ Loaded {len(full_df):,} rows")
                        st.rerun()
                    else:
                        st.error("Failed to load full dataset")
        else:
            st.caption("Using full dataset")
    
    # Process the dataframe
    # Try to convert potential numeric columns
    numeric_patterns = ['rate', 'count', 'number', 'total', 'percent', 'value', 'amount', 'score', 'phase', 'year']
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if column might be numeric
        if any(pattern in col_lower for pattern in numeric_patterns) or df[col].dtype == 'object':
            try:
                # Try to convert to numeric
                test_convert = pd.to_numeric(df[col], errors='coerce')
                # If more than 30% converts successfully, keep it as numeric
                if test_convert.notna().sum() > len(df) * 0.3:
                    df[col] = test_convert
            except:
                pass
    
    # Detect column types
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 AI Insights", "📈 Charts", "📊 Statistics", "📋 Data Table", "💾 Export"])
    
    with tab1:
        st.subheader("🤖 AI-Powered Data Analysis")
        st.info("Let AI analyze your dataset and recommend the most meaningful visualizations")

        # Check if OpenAI API key is configured
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            # Debug info
            with st.expander("🔍 Debug Info"):
                st.write(f"Looking for .env at: {env_path}")
                st.write(f".env exists: {env_path.exists()}")
                st.write(f"Current working directory: {os.getcwd()}")
        else:
            # Show that API key is configured (don't expose any part of it)
            st.success("✅ OpenAI API Key configured successfully")

            # Add button to trigger AI analysis
            if st.button("🔍 Analyze Dataset with AI", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your dataset... This may take 30-60 seconds"):
                    analysis = analyze_dataset_with_ai(df, st.session_state.viz_name)

                    if "error" in analysis:
                        st.error(f"Error: {analysis['error']}")
                    else:
                        # Store analysis in session state
                        st.session_state['ai_analysis'] = analysis
                        st.success("✅ Analysis complete!")

            # Display analysis if available
            if 'ai_analysis' in st.session_state and st.session_state['ai_analysis']:
                analysis = st.session_state['ai_analysis']

                # Dataset Analysis
                st.markdown("### 📋 Dataset Overview")
                st.write(analysis.get('analysis', 'No analysis available'))

                st.divider()

                # Key Insights
                if 'key_insights' in analysis and analysis['key_insights']:
                    st.markdown("### 💡 Key Insights")
                    for i, insight in enumerate(analysis['key_insights'], 1):
                        st.markdown(f"{i}. {insight}")

                    st.divider()

                # AI-Recommended Visualizations
                if 'visualizations' in analysis and analysis['visualizations']:
                    st.markdown("### 📊 AI-Recommended Visualizations")
                    st.caption(f"AI suggested {len(analysis['visualizations'])} visualizations based on your data")

                    for i, viz in enumerate(analysis['visualizations'], 1):
                        with st.expander(f"📈 {i}. {viz.get('title', 'Visualization')}", expanded=i<=2):
                            # Description
                            st.markdown(f"**Why this matters:** {viz.get('description', 'N/A')}")

                            # Key insight
                            if 'insight' in viz:
                                st.info(f"💡 **Insight:** {viz['insight']}")

                            # Chart details
                            columns_info = f"x: {viz.get('x_column', 'N/A')}"
                            if viz.get('y_column'):
                                columns_info += f", y: {viz.get('y_column')}"
                            if viz.get('value_column'):
                                columns_info += f", value: {viz.get('value_column')}"
                            st.caption(f"Chart type: {viz.get('chart_type', 'N/A')} | Columns: {columns_info}")

                            # Generate the chart
                            st.markdown("---")
                            try:
                                create_ai_chart(df, viz)
                            except Exception as e:
                                st.error(f"Could not create chart: {e}")
                else:
                    st.warning("No visualizations recommended")
            else:
                st.info("👆 Click the button above to get AI-powered insights and visualization recommendations")

    with tab2:
        if numeric_cols:
            st.success(f"Found {len(numeric_cols)} numeric columns")

            # Show visualizations for first 3 numeric columns
            for i, col in enumerate(numeric_cols[:3]):
                st.subheader(f"Distribution of {col}")

                # Clean data
                clean_data = df[col].dropna()

                if len(clean_data) > 0:
                    # Create two columns
                    c1, c2 = st.columns(2)

                    with c1:
                        # Histogram
                        st.write("**Histogram**")
                        if HAVE_PLOTLY:
                            fig = px.histogram(df.dropna(subset=[col]), x=col, nbins=30)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Create bins for histogram
                            counts, bins = np.histogram(clean_data, bins=20)
                            hist_df = pd.DataFrame({'count': counts}, index=bins[:-1].round(2))
                            st.bar_chart(hist_df)

                    with c2:
                        # Basic stats
                        st.write("**Statistics**")
                        stats_df = pd.DataFrame({
                            'Statistic': ['Count', 'Mean', 'Median', 'Min', 'Max', 'Std Dev'],
                            'Value': [
                                f"{len(clean_data)}",
                                f"{clean_data.mean():.2f}",
                                f"{clean_data.median():.2f}",
                                f"{clean_data.min():.2f}",
                                f"{clean_data.max():.2f}",
                                f"{clean_data.std():.2f}"
                            ]
                        })
                        st.dataframe(stats_df, hide_index=True)

            # Grouped analysis if text columns exist
            if text_cols and numeric_cols:
                st.divider()
                st.subheader("Grouped Analysis")

                # Find text columns with reasonable number of unique values
                # Skip columns with unhashable types (like dicts from GeoJSON)
                suitable_text_cols = []
                for col in text_cols:
                    try:
                        if df[col].nunique() < 20:
                            suitable_text_cols.append(col)
                    except TypeError:
                        # Skip columns with unhashable types (e.g., dicts, lists)
                        pass

                if suitable_text_cols:
                    for txt_col in suitable_text_cols[:2]:  # First 2 suitable text columns
                        for num_col in numeric_cols[:2]:  # First 2 numeric columns
                            st.write(f"**Average {num_col} by {txt_col}**")

                            try:
                                grouped = df.groupby(txt_col)[num_col].mean().sort_values(ascending=False).head(10)

                                if HAVE_PLOTLY:
                                    fig = px.bar(
                                        x=grouped.index,
                                        y=grouped.values,
                                        labels={'x': txt_col, 'y': f'Average {num_col}'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.bar_chart(grouped)
                            except Exception as e:
                                st.error(f"Could not create grouped analysis: {e}")
        else:
            st.warning("No numeric columns found for visualization")

    with tab3:
        st.subheader("Dataset Statistics")
        
        # Overall stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", f"{len(df.columns)}")
        with col3:
            st.metric("Missing Values", f"{df.isna().sum().sum():,}")
        
        # Column-wise stats
        st.subheader("Column Information")
        col_info = []
        for col in df.columns:
            # Try to get unique count, but handle unhashable types (dicts, lists)
            try:
                unique_count = f"{df[col].nunique():,}"
            except TypeError:
                unique_count = "N/A (complex type)"

            col_info.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null': f"{df[col].notna().sum():,}",
                'Unique': unique_count,
                'Missing %': f"{(df[col].isna().sum() / len(df) * 100):.1f}%"
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, hide_index=True, use_container_width=True)

    with tab4:
        st.subheader("Data Table")
        
        # Add filters
        with st.expander("🔍 Filters"):
            filter_cols = st.multiselect("Select columns to display", 
                                        options=list(df.columns),
                                        default=list(df.columns)[:10])
            
            n_rows = st.slider("Number of rows to display", 
                              min_value=10, 
                              max_value=min(1000, len(df)), 
                              value=min(100, len(df)))
        
        if filter_cols:
            display_df = df[filter_cols].head(n_rows)
        else:
            display_df = df.head(n_rows)
        
        st.dataframe(display_df, use_container_width=True)

    with tab5:
        st.subheader("Export Data")

        # Use full data if available, otherwise use preview
        export_df = df  # df is already set to full data if available

        # Show export status
        if using_full_data:
            st.success(f"✅ Exporting FULL dataset: {len(export_df):,} rows")
        else:
            st.warning(f"⚠️ Exporting PREVIEW only: {len(export_df):,} rows")
            st.info("💡 Tip: Click 'Load Full Dataset' button at the top to export the complete dataset")

        st.divider()

        # Export format selection
        export_format = st.selectbox(
            "Select export format",
            ["CSV", "Excel", "JSON", "Parquet"]
        )

        if export_format == "CSV":
            csv = export_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download CSV ({len(export_df):,} rows)",
                data=csv,
                file_name=f"{st.session_state.viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        elif export_format == "Excel":
            # Note: This requires openpyxl or xlsxwriter
            try:
                import io
                buffer = io.BytesIO()
                export_df.to_excel(buffer, index=False, engine='openpyxl')
                buffer.seek(0)

                st.download_button(
                    label=f"📥 Download Excel ({len(export_df):,} rows)",
                    data=buffer,
                    file_name=f"{st.session_state.viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.error("Excel export requires 'openpyxl' package. Install with: pip install openpyxl")

        elif export_format == "JSON":
            json_str = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label=f"📥 Download JSON ({len(export_df):,} rows)",
                data=json_str,
                file_name=f"{st.session_state.viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        elif export_format == "Parquet":
            try:
                import io
                buffer = io.BytesIO()
                export_df.to_parquet(buffer, index=False)
                buffer.seek(0)

                st.download_button(
                    label=f"📥 Download Parquet ({len(export_df):,} rows)",
                    data=buffer,
                    file_name=f"{st.session_state.viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                    mime="application/octet-stream"
                )
            except ImportError:
                st.error("Parquet export requires 'pyarrow' or 'fastparquet' package")

# ===========================
# Main Application
# ===========================
def main():
    # Header
    st.markdown("# 🧠 Mindcube Data Map US")
    st.markdown("## Mental Health Metadata Catalog Insight Generator")

    # Load existing datasets from backend on first run
    if not st.session_state.datasets_loaded:
        with st.spinner("Loading dataset status..."):
            load_existing_datasets()

    # Check if we're in visualization mode
    if st.session_state.show_viz:
        show_visualization_page()
        return
    
    # Sidebar for organization selection
    with st.sidebar:
        st.header("Configuration")
        org = st.selectbox(
            "Select Organization",
            ["All", "CDC", "SAMHSA", "BRFSS", "NHIS", "YRBSS", "NSSP", "LOCAL"],
            index=0,
            help="Select 'All' to search across all datasets"
        )
        
        st.divider()
        
        # Health check
        if st.button("🔍 Check Backend Status"):
            health_data, error = call_backend("GET", "/health")
            if not error:
                st.success("✅ Backend is running")
            else:
                st.error(f"❌ Backend error: {error}")
        
        # RAG Status
        st.divider()
        st.subheader("📊 RAG Index Status")
        if st.button("🔄 Refresh Status"):
            rag_status, error = call_backend("GET", f"/rag_status?org={org}")
            if rag_status:
                st.metric("Indexed Datasets", rag_status.get("indexed_datasets", 0))
                st.metric("Total Chunks", rag_status.get("total_chunks", 0))
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search & Results", 
        "🤖 Ask Questions About the Data", 
        "📊 Analytics",
        "🗂️ Manage Indexed Data"
    ])
    
    # ===========================
    # Tab 1: Search & Results
    # ===========================
    with tab1:
        st.header("Find Relevant Datasets")
        
        # User story input
        with st.form("search_form"):
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                persona = st.selectbox(
                    "I am a...",
                    PERSONAS,
                    help="Select your role to get tailored results"
                )

            with col2:
                search_type = st.selectbox(
                    "Search Type",
                    ["Semantic", "Keyword"],
                    help="Semantic search uses AI to understand context"
                )

            with col3:
                num_results = st.selectbox(
                    "Results",
                    [10, 20, 30, 50],
                    index=0,
                    help="Number of results to return"
                )

            user_story = st.text_area(
                "Describe what you're looking for",
                placeholder="Example: As a public health researcher, I need data on adolescent depression prevalence by gender to identify disparities and inform targeted interventions.",
                height=100
            )

            submitted = st.form_submit_button("🔍 Find Datasets", type="primary", use_container_width=True)
        
        if submitted and user_story:
            with st.spinner("Searching for relevant datasets..."):
                # Store query
                st.session_state.last_query = {
                    "persona": persona,
                    "story": user_story,
                    "org": org,
                    "timestamp": datetime.now()
                }
                
                # Add to search history
                st.session_state.search_history.append(st.session_state.last_query)
                
                # Perform search based on type
                if search_type == "Semantic":
                    # Try semantic search first
                    data, error = call_backend(
                        "POST",
                        "/semantic/search",
                        json={"story": user_story, "org": org, "k": num_results, "persona": persona}
                    )
                else:
                    # Use keyword search
                    data, error = call_backend(
                        "GET",
                        f"/catalog/search?org={org}&q={user_story[:100]}"
                    )
                
                if not error:
                    results = data.get("results", [])
                    st.session_state.results = results
                    
                    if results:
                        st.success(f"Found {len(results)} relevant datasets")
                    else:
                        st.warning("No datasets found. Try different search terms.")
                else:
                    st.error(f"Search failed: {error}")
        
        # Display results
        if st.session_state.results:
            st.divider()
            st.subheader(f"📊 Search Results ({len(st.session_state.results)} datasets)")
            
            for i, result in enumerate(st.session_state.results):
                with st.expander(f"**{i+1}. {result.get('name', 'Unnamed Dataset')}**", expanded=i<3):
                    # Dataset info
                    uid = result.get('uid', '')
                    desc = result.get('description', 'No description available')
                    
                    st.write(f"**UID:** `{uid}`")
                    st.write(f"**Description:** {desc[:500]}..." if len(desc) > 500 else f"**Description:** {desc}")
                    
                    # Action buttons
                    col1, col2, col3, col4, col5 = st.columns(5)

                    # Get the actual org and source from the result
                    result_org = result.get('org', org)
                    result_source = result.get('source', 'socrata')

                    with col1:
                        # Check if already ingested or if it's a local dataset
                        is_ingested = uid in st.session_state.ingested_datasets
                        is_local = result_source == 'local'

                        if is_local:
                            st.success("💾 Local")
                        else:
                            # Auto-index checkbox
                            auto_index = st.checkbox(
                                "Auto-index",
                                key=f"auto_{i}_{uid}",
                                help="Automatically create searchable chunks after ingesting"
                            )

                            if st.button(f"💾 Ingest", key=f"ingest_{i}_{uid}"):
                                with st.spinner(f"Ingesting {result['name'][:30]}..."):
                                    ingest_data, error = call_backend(
                                        "POST",
                                        "/ingest",
                                        json={
                                            "org": result_org,  # Use the result's org, not the sidebar selection
                                            "pick_uid": uid,
                                            "auto_index": auto_index
                                        }
                                    )
                                    if error:
                                        st.error(f"Failed to ingest: {error}")
                                    elif ingest_data.get("ingested"):
                                        st.success(f"✅ Ingested {ingest_data['rows']} rows")
                                        st.session_state.ingested_datasets.add(uid)

                                        if ingest_data.get("indexed"):
                                            st.success(f"✅ Indexed {ingest_data.get('chunks_created', 0)} chunks")
                                            st.session_state.indexed_datasets.add(uid)
                                    else:
                                        st.warning(f"Not ingested: {ingest_data.get('reason', 'Unknown')}")
                    
                    with col2:
                        # Index for RAG button
                        is_indexed = uid in st.session_state.indexed_datasets
                        if is_indexed:
                            # Already indexed - show status only
                            st.success("✅ Indexed")
                        elif is_ingested or is_local:
                            # Not indexed yet - show Index button
                            if st.button(f"📚 Index", key=f"index_{i}_{uid}"):
                                with st.spinner(f"Indexing {result['name'][:30]}... (1-2 min)"):
                                    index_data, error = call_backend(
                                        "POST",
                                        "/index_dataset",
                                        json={
                                            "org": result_org,  # Use the result's org, not the sidebar selection
                                            "uid": uid,
                                            "limit_rows": 20000
                                        }
                                    )
                                    if error:
                                        st.error(f"Failed: {error}")
                                    elif index_data.get("success"):
                                        st.success(f"✅ Indexed {index_data.get('chunks_created', 0)} chunks")
                                        st.session_state.indexed_datasets.add(uid)
                                    else:
                                        st.error(f"Failed: {index_data.get('error', 'Unknown')}")
                        else:
                            st.info("💾 Ingest first")
                    
                    with col3:
                        if st.button(f"👁️ Preview", key=f"preview_{i}_{uid}"):
                            with st.spinner("Loading preview..."):
                                df = preview_dataset(result_org, uid, 200)
                                if df is not None:
                                    st.write(f"**Preview** ({len(df)} rows):")
                                    st.dataframe(df.head(50), use_container_width=True)
                                else:
                                    st.error("Could not load preview")

                    with col4:
                        if st.button(f"📊 Visualize", key=f"viz_{i}_{uid}"):
                            with st.spinner("Loading data for visualization..."):
                                df = preview_dataset(result_org, uid, 1000)
                                if df is not None and len(df) > 0:
                                    st.session_state.viz_data = df
                                    st.session_state.viz_name = result.get('name', 'Dataset')
                                    st.session_state.viz_org = result_org
                                    st.session_state.viz_uid = uid
                                    st.session_state.viz_full_data = None  # Clear any previous full data
                                    st.session_state.show_viz = True
                                    st.rerun()
                                else:
                                    st.error("Could not load data for visualization")

                    with col5:
                        # Show source link only for online datasets, not local ones
                        if not is_local:
                            link = f"https://data.{result_org.lower()}.gov/d/{uid}"
                            st.link_button("🔗 Source", link)
                        else:
                            st.caption("Local file")
        else:
            st.info("No results yet. Enter a user story above and click 'Find Datasets' to search.")
    
    # ===========================
    # Tab 2: Ask Questions (RAG)
    # ===========================
    with tab2:
        st.header("Ask Questions About the Data")
        
        # Check RAG status
        rag_status, error = call_backend("GET", f"/rag_status?org={org}")
        
        if rag_status and rag_status.get("indexed_datasets", 0) > 0:
            st.success(f"✅ {rag_status['indexed_datasets']} datasets indexed with {rag_status['total_chunks']} searchable chunks")
            
            # Show indexed datasets
            with st.expander("View Indexed Datasets"):
                for dataset in rag_status.get("recent_datasets", []):
                    st.write(f"• **{dataset.get('dataset_name', dataset['dataset_uid'])}**: {dataset['chunk_count']} chunks")
        else:
            st.warning("⚠️ No datasets indexed yet. Search and index datasets in the 'Search & Results' tab first!")
            st.info("To get answers from actual data:\n1. Search for datasets\n2. Click 'Ingest Dataset'\n3. Click 'Index' to make it searchable")
        
        question = st.text_input(
            "Your Question",
            placeholder="What are the key trends in mental health outcomes?",
            help="Ask any question about the datasets you've indexed"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            answer_persona = st.selectbox(
                "Answer Style",
                options=PERSONAS,
                help="Choose how you want the answer formatted"
            )
        with col2:
            context_k = st.slider(
                "Context Chunks",
                min_value=3,
                max_value=10,
                value=5,
                help="How much context to use for the answer"
            )
        
        if st.button("🤖 Generate Answer", type="primary", use_container_width=True):
            if not question:
                st.warning("Please enter a question")
            else:
                with st.spinner("Thinking..."):
                    payload = {
                        "question": question,
                        "org": org,
                        "k": context_k,
                        "persona": answer_persona,
                        "use_actual_data": True  # Use actual data when available
                    }
                    
                    data, error = call_backend("POST", "/answer", json=payload)
                    
                    if error:
                        st.error(f"Failed to generate answer: {error}")
                    else:
                        st.markdown("### 💬 Answer")
                        
                        # Show data source mode
                        if data.get("mode"):
                            mode = data["mode"]
                            if mode == "actual_data":
                                st.info("📊 This answer is based on **actual data** from indexed datasets")
                            elif mode == "metadata_semantic" or mode == "semantic":
                                st.warning("📝 This answer is based on **dataset descriptions** (metadata only)")
                            elif mode == "keyword_fallback":
                                st.warning("🔍 This answer is based on **keyword search** in dataset descriptions")
                        
                        answer_text = data.get("answer", "No answer generated")
                        st.markdown(answer_text)
                        
                        if data.get("sources"):
                            st.markdown("### 📚 Sources")
                            for source in data["sources"]:
                                source_name = source.get('name', 'Unknown')
                                source_link = source.get('link', '#')
                                st.markdown(f"- [{source_name}]({source_link})")
    
    # ===========================
    # Tab 3: Analytics
    # ===========================
    with tab3:
        st.header("System Analytics")
        
        if st.session_state.search_history:
            # Convert to dataframe
            search_df = pd.DataFrame(st.session_state.search_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Searches by persona
                if HAVE_PLOTLY:
                    fig = px.pie(
                        search_df,
                        names='persona',
                        title='Searches by Persona'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    persona_counts = search_df['persona'].value_counts()
                    st.bar_chart(persona_counts)
                    st.caption("Searches by Persona")
            
            with col2:
                # Searches over time
                search_df['timestamp'] = pd.to_datetime(search_df['timestamp'])
                search_df['date'] = search_df['timestamp'].dt.date
                daily_searches = search_df.groupby('date').size()
                
                st.line_chart(daily_searches)
                st.caption("Searches Over Time")
            
            # Recent searches
            st.subheader("Recent Searches")
            recent = search_df.tail(5)[['timestamp', 'persona', 'story', 'org']]
            st.dataframe(recent, hide_index=True, use_container_width=True)
        else:
            st.info("No search history yet. Start searching to see analytics.")
        
        # Dataset statistics
        st.divider()
        st.subheader("Dataset Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Datasets Ingested", len(st.session_state.ingested_datasets))
        
        with col2:
            st.metric("Datasets Indexed", len(st.session_state.indexed_datasets))
        
        with col3:
            st.metric("Total Searches", len(st.session_state.search_history))
    
    # ===========================
    # Tab 4: Manage Indexed Data
    # ===========================
    with tab4:
        st.header("Manage Indexed Data")
        
        # Get list of all datasets
        datasets_data, error = call_backend("GET", f"/datasets?org={org}")
        
        if datasets_data and datasets_data.get("items"):
            indexed_count = sum(1 for d in datasets_data["items"] if d.get("indexed_for_rag"))
            total_count = datasets_data["count"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Datasets", total_count)
            with col2:
                st.metric("Indexed for RAG", indexed_count)
            with col3:
                st.metric("Not Indexed", total_count - indexed_count)
            
            st.divider()
            
            # Filter options
            show_only = st.radio(
                "Show:",
                ["All Datasets", "Indexed Only", "Not Indexed"],
                horizontal=True
            )
            
            # Display datasets with their status
            for item in datasets_data["items"]:
                is_indexed = item.get("indexed_for_rag", False)
                chunk_count = item.get("chunk_count", 0)
                
                # Apply filter
                if show_only == "Indexed Only" and not is_indexed:
                    continue
                elif show_only == "Not Indexed" and is_indexed:
                    continue
                
                with st.container():
                    col1, col2, col3 = st.columns([4, 1, 1])
                    
                    with col1:
                        st.write(f"**{item['name'][:100]}**")
                        st.caption(f"UID: {item['uid']} | Org: {item['org']}")
                    
                    with col2:
                        if is_indexed:
                            st.success(f"✅ {chunk_count} chunks")
                        else:
                            st.info("Not indexed")
                    
                    with col3:
                        if not is_indexed and item.get("uid"):
                            if st.button("Index Now", key=f"index_manage_{item['uid']}"):
                                with st.spinner("Indexing... This may take 1-2 minutes"):
                                    index_result, error = call_backend(
                                        "POST",
                                        "/index_dataset",
                                        json={
                                            "org": item["org"],
                                            "uid": item["uid"],
                                            "limit_rows": 20000
                                        }
                                    )
                                    if index_result and index_result.get("success"):
                                        st.success(f"✅ Indexed {index_result['chunks_created']} chunks!")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {error or 'Unknown error'}")
                        elif is_indexed:
                            if st.button("Re-index", key=f"reindex_manage_{item['uid']}"):
                                with st.spinner("Re-indexing..."):
                                    index_result, error = call_backend(
                                        "POST",
                                        "/index_dataset",
                                        json={
                                            "org": item["org"],
                                            "uid": item["uid"],
                                            "limit_rows": 20000
                                        }
                                    )
                                    if index_result and index_result.get("success"):
                                        st.success(f"✅ Re-indexed {index_result['chunks_created']} chunks!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to re-index")
                    
                    st.divider()
        else:
            st.info("No datasets found. Ingest some datasets first using the 'Search & Results' tab.")

if __name__ == "__main__":
    main()