"""
🎯 AutoML Tabular - Interactive Demo
Pure UX showcase for recruiters and stakeholders
Zero changes to core logic - presentation layer only
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from automl_tabular.orchestrator import AutoMLOrchestrator


# Page config
st.set_page_config(
    page_title="AutoML Tabular Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2E86AB 0%, #A23B72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown('<h1 class="main-header">🤖 AutoML Tabular</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Production-grade AutoML with explainability • 1.4x faster • Zero configuration</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Quick Demo")
    st.markdown("""
    **What this does:**
    - Detects problem type automatically
    - Searches best model families
    - Generates explainable HTML report
    - Optimized for speed (1.4x faster)
    
    **Perfect for:**
    - Binary/Multiclass Classification
    - Regression tasks
    - Any tabular dataset
    """)
    
    st.markdown("---")
    st.markdown("### ⚡ Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Speedup", "1.37x", "37% faster")
    with col2:
        st.metric("Quality", "100%", "Maintained")
    
    st.markdown("---")
    st.markdown("### ⚠️ Demo Note")
    st.info("Demo uses 5-20 trials for responsiveness. Full production runs via CLI support higher trial budgets for maximum quality.")
    
    st.markdown("---")
    st.markdown("**By:** Mounusha Ram Metti  \n**Version:** v1.1.0-performance-optimized")


# Main content
tab1, tab2 = st.tabs(["🚀 Run Demo", "📊 About"])

with tab1:
    # Dataset selection
    st.markdown("### 1️⃣ Choose Dataset")
    
    dataset_option = st.radio(
        "Select a sample dataset or upload your own:",
        ["Titanic (Binary Classification)", "Upload Custom CSV"],
        horizontal=True
    )
    
    df = None
    target_col = None
    
    if dataset_option == "Titanic (Binary Classification)":
        # Look for Titanic dataset
        titanic_paths = [
            "examples/titanic.csv",
            "data/titanic.csv",
            "titanic.csv"
        ]
        titanic_path = None
        for path in titanic_paths:
            if os.path.exists(path):
                titanic_path = path
                break
        
        if titanic_path:
            df = pd.read_csv(titanic_path)
            target_col = "Survived"
            st.success(f"✅ Loaded Titanic dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            st.error("❌ Titanic dataset not found. Please check examples/titanic.csv exists.")
    
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            target_col = st.selectbox("Select target column:", df.columns.tolist())
    
    # Show data preview
    if df is not None:
        st.markdown("### 2️⃣ Data Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Target", target_col or "Not selected")
        
        with st.expander("📋 View Data Sample"):
            st.dataframe(df.head(10), use_container_width=True)
    
    # Run AutoML
    if df is not None and target_col:
        st.markdown("### 3️⃣ Run AutoML")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            run_button = st.button("🚀 Run AutoML Pipeline", type="primary", use_container_width=True)
        with col2:
            n_trials = st.selectbox("Trials per model:", [5, 10, 20], index=0)
        
        if run_button:
            # Save temp CSV
            temp_csv = f"temp_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(temp_csv, index=False)
            
            # Create output directory
            output_dir = f"output/demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize with custom config
                status_text.markdown("🔧 **Initializing AutoML...**")
                progress_bar.progress(10)
                
                # Custom config for trials
                custom_config = {
                    'search': {
                        'n_trials_per_model': n_trials
                    }
                }
                
                orchestrator = AutoMLOrchestrator(config=custom_config)
                
                # Run the complete pipeline
                status_text.markdown("🚀 **Running AutoML pipeline...**")
                progress_bar.progress(20)
                
                start_time = time.time()
                
                # Note: run() handles all steps internally
                results = orchestrator.run(
                    data_path=temp_csv,
                    target_column=target_col,
                    output_dir=output_dir,
                    metric='auto'
                )
                
                search_time = time.time() - start_time
                progress_bar.progress(100)
                
                status_text.markdown("✅ **AutoML Complete!**")
                
                # Results
                st.markdown("---")
                st.markdown("## 🏆 Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    best_model = results.get('recommended_model_name', 'Unknown')
                    st.metric("Best Model", best_model)
                with col2:
                    best_score = results.get('recommended_metric_value', 0.0)
                    metric_name = results.get('primary_metric_name', 'score').upper()
                    st.metric(metric_name, f"{best_score:.4f}")
                with col3:
                    st.metric("Search Time", f"{search_time:.1f}s")
                with col4:
                    total_trials = results.get('total_models', 0)
                    st.metric("Models Tried", total_trials)
                
                # Leaderboard
                st.markdown("### 📊 Model Leaderboard")
                if orchestrator.leaderboard and orchestrator.leaderboard.df is not None and len(orchestrator.leaderboard.df) > 0:
                    # Get leaderboard as DataFrame
                    display_board = orchestrator.leaderboard.df.head(10).copy()
                    display_board['score'] = display_board['score'].apply(lambda x: f"{x:.4f}")
                    display_board['train_time_sec'] = display_board['train_time_sec'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(display_board, use_container_width=True)
                
                # Report
                st.markdown("### 📄 Generated Report")
                report_path = results.get('report_path')
                if report_path and os.path.exists(report_path):
                    # Provide download
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_html = f.read()
                    
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=report_html,
                        file_name="automl_report.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    # Embed report
                    with st.expander("👁️ Preview Report (Click to expand)"):
                        st.components.v1.html(report_html, height=800, scrolling=True)
                
                # Model download
                model_path = results.get('model_path')
                if model_path and os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model_bytes = f.read()
                    
                    st.download_button(
                        label="📦 Download Trained Model",
                        data=model_bytes,
                        file_name="automl_model.joblib",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
            finally:
                # Always cleanup temp file
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)

with tab2:
    st.markdown("## 📊 About AutoML Tabular")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Key Features
        
        - **🎯 Auto Task Detection**
          - Binary/Multiclass Classification
          - Regression
          - Automatic metric selection
        
        - **🔍 Smart Model Search**
          - Optuna hyperparameter optimization
          - MedianPruner for early stopping
          - Dynamic CV folds
        
        - **⚡ Performance Optimized**
          - 1.37x faster than baseline
          - Sparse matrix support
          - Float32 precision
        
        - **📊 Explainable AI**
          - Feature importance
          - Model selection rationale
          - Data quality warnings
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Technical Stack
        
        - **AutoML**: Optuna + scikit-learn
        - **Models**: Logistic, RF, XGBoost, LightGBM
        - **Validation**: Stratified CV + holdout
        - **Reports**: HTML with visualizations
        
        ### 📈 Performance
        
        - **Speedup**: 1.37x (Titanic benchmark)
        - **Quality**: 100% preserved
        - **Memory**: Optimized with sparse matrices
        - **Parallelism**: All CPU cores utilized
        
        ### 📚 Documentation
        
        - [README.md](README.md) - Project overview
        - [PERFORMANCE.md](PERFORMANCE.md) - Optimization guide
        - [SMART_PREPROCESSING.md](SMART_PREPROCESSING.md) - Preprocessing strategies
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Perfect For
    
    - **ML Engineers**: Production-ready AutoML with performance optimization
    - **Data Scientists**: Fast prototyping with maintained quality
    - **Researchers**: Reproducible baselines with detailed reporting
    - **Recruiters**: See ML Engineering best practices in action
    
    ### 🔒 Project Status
    
    **Version**: v1.1.0-performance-optimized  
    **Status**: ✅ Production-ready, thoroughly tested, FROZEN  
    **Author**: Mounusha Ram Metti  
    **Date**: January 2026
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🤖 <strong>AutoML Tabular</strong> v1.1.0 • Production-grade AutoML with explainability</p>
    <p>Built with ❤️ by Mounusha Ram Metti • January 2026</p>
</div>
""", unsafe_allow_html=True)
