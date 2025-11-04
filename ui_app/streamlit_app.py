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
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if 'show_viz' not in st.session_state:
    st.session_state.show_viz = False
if 'viz_data' not in st.session_state:
    st.session_state.viz_data = None
if 'viz_name' not in st.session_state:
    st.session_state.viz_name = ""
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

def show_visualization_page():
    """Display the visualization page"""
    
    # Back button
    if st.button("⬅️ Back to Results"):
        st.session_state.show_viz = False
        st.rerun()
    
    # Display visualizations in full width
    st.header(f"📊 Visualizations for {st.session_state.viz_name}")
    
    df = st.session_state.viz_data
    st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "📊 Statistics", "📋 Data Table", "💾 Export"])
    
    with tab1:
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
                suitable_text_cols = [col for col in text_cols if df[col].nunique() < 20]
                
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
    
    with tab2:
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
            col_info.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null': f"{df[col].notna().sum():,}",
                'Unique': f"{df[col].nunique():,}",
                'Missing %': f"{(df[col].isna().sum() / len(df) * 100):.1f}%"
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, hide_index=True, use_container_width=True)
    
    with tab3:
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
    
    with tab4:
        st.subheader("Export Data")
        
        # Export format selection
        export_format = st.selectbox(
            "Select export format",
            ["CSV", "Excel", "JSON", "Parquet"]
        )
        
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{st.session_state.viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        elif export_format == "Excel":
            # Note: This requires openpyxl or xlsxwriter
            try:
                import io
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False, engine='openpyxl')
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer,
                    file_name=f"{st.session_state.viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.error("Excel export requires 'openpyxl' package. Install with: pip install openpyxl")
        
        elif export_format == "JSON":
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{st.session_state.viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        elif export_format == "Parquet":
            try:
                import io
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False)
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Parquet",
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
    
    # Check if we're in visualization mode
    if st.session_state.show_viz:
        show_visualization_page()
        return
    
    # Sidebar for organization selection
    with st.sidebar:
        st.header("Configuration")
        org = st.selectbox(
            "Select Organization",
            ["CDC", "SAMHSA"],
            index=0 if DEFAULT_ORG == "CDC" else 1
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
            col1, col2 = st.columns([3, 1])
            
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
                        json={"story": user_story, "org": org, "k": 10, "persona": persona}
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
                    
                    with col1:
                        # Check if already ingested
                        is_ingested = uid in st.session_state.ingested_datasets
                        if is_ingested:
                            st.info("✅ Ingested")
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
                                            "org": org,
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
                            st.info("📚 Indexed")
                        elif is_ingested:
                            if st.button(f"📚 Index", key=f"index_{i}_{uid}"):
                                with st.spinner(f"Indexing {result['name'][:30]}... (1-2 min)"):
                                    index_data, error = call_backend(
                                        "POST",
                                        "/index_dataset",
                                        json={
                                            "org": org,
                                            "uid": uid,
                                            "limit_rows": 5000
                                        }
                                    )
                                    if error:
                                        st.error(f"Failed: {error}")
                                    elif index_data.get("success"):
                                        st.success(f"✅ Indexed {index_data.get('chunks_created', 0)} chunks")
                                        st.session_state.indexed_datasets.add(uid)
                                    else:
                                        st.error(f"Failed: {index_data.get('error', 'Unknown')}")
                    
                    with col3:
                        if st.button(f"👁️ Preview", key=f"preview_{i}_{uid}"):
                            with st.spinner("Loading preview..."):
                                df = preview_dataset(org, uid, 200)
                                if df is not None:
                                    st.write(f"**Preview** ({len(df)} rows):")
                                    st.dataframe(df.head(50), use_container_width=True)
                                else:
                                    st.error("Could not load preview")
                    
                    with col4:
                        if st.button(f"📊 Visualize", key=f"viz_{i}_{uid}"):
                            with st.spinner("Loading data for visualization..."):
                                df = preview_dataset(org, uid, 1000)
                                if df is not None and len(df) > 0:
                                    st.session_state.viz_data = df
                                    st.session_state.viz_name = result.get('name', 'Dataset')
                                    st.session_state.show_viz = True
                                    st.rerun()
                                else:
                                    st.error("Could not load data for visualization")
                    
                    with col5:
                        link = f"https://data.{org.lower()}.gov/d/{uid}"
                        st.link_button("🔗 Source", link)
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
                                            "limit_rows": 5000
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
                                            "limit_rows": 5000
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