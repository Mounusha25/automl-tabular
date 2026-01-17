"""
Streamlit UI for AutoML Tabular
Interactive demo for recruiters and users
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import tempfile
import shutil
from io import StringIO
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from automl_tabular.orchestrator import AutoMLOrchestrator
from automl_tabular.config import load_default_config

# Page config
st.set_page_config(
    page_title="AutoML Tabular Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🤖 AutoML Tabular</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explainable AutoML for Tabular Data • By Mounusha Ram Metti</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Configuration")
    
    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your tabular dataset (CSV format)"
    )
    
    # Example datasets
    st.markdown("### Or Try Examples")
    example_choice = st.selectbox(
        "Example datasets:",
        ["None", "Titanic (Binary)", "Wine Quality (Multiclass)", "California Housing (Regression)"]
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # These will be populated after data is loaded
    target_column = None
    metric_choice = None
    time_budget = None

# Main content
if uploaded_file is not None or example_choice != "None":
    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        data_name = uploaded_file.name.replace('.csv', '')
    else:
        # Load example dataset
        example_paths = {
            "Titanic (Binary)": "examples/titanic.csv",
            "Wine Quality (Multiclass)": "examples/wine_quality.csv",
            "California Housing (Regression)": "examples/california_housing.csv"
        }
        example_path = example_paths[example_choice]
        df = pd.read_csv(example_path)
        data_name = example_choice.split(' ')[0].lower()
    
    # Data preview
    st.success(f"✅ Loaded dataset: **{data_name}** ({len(df)} rows, {len(df.columns)} columns)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Dataset Info")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns)}")
        st.write(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            st.warning(f"⚠️ {missing} missing values detected")
        else:
            st.info("✓ No missing values")
    
    # Configuration in sidebar
    with st.sidebar:
        target_column = st.selectbox(
            "Target column:",
            options=df.columns.tolist(),
            index=len(df.columns) - 1,  # Default to last column
            help="The column you want to predict"
        )
        
        # Auto-detect problem type
        if target_column:
            unique_values = df[target_column].nunique()
            if df[target_column].dtype in ['object', 'category'] or unique_values <= 20:
                problem_type = "classification"
                if unique_values == 2:
                    default_metric = "roc_auc"
                    metric_options = ["roc_auc", "accuracy", "f1", "precision", "recall"]
                else:
                    default_metric = "accuracy"
                    metric_options = ["accuracy", "roc_auc", "f1"]
            else:
                problem_type = "regression"
                default_metric = "rmse"
                metric_options = ["rmse", "mae", "r2"]
            
            st.info(f"🎯 Detected: **{problem_type.title()}** ({unique_values} unique values)")
        
        metric_choice = st.selectbox(
            "Metric:",
            options=metric_options,
            index=0,
            help="Primary metric for model selection"
        )
        
        time_budget = st.slider(
            "Time budget (seconds):",
            min_value=30,
            max_value=300,
            value=60,
            step=30,
            help="⚠️ Demo mode: shorter time = faster results, fewer trials"
        )
        
        st.markdown("---")
        run_automl = st.button("🚀 Run AutoML", type="primary", use_container_width=True)
    
    # Run AutoML
    if run_automl and target_column:
        st.markdown("---")
        st.markdown("### 🔄 Running AutoML Pipeline...")
        
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Save uploaded data to temp file
            temp_csv = tmp_path / "data.csv"
            df.to_csv(temp_csv, index=False)
            
            # Configure AutoML (demo mode with reduced trials)
            config = load_default_config()
            config['search']['time_limit_seconds'] = time_budget
            config['search']['max_trials_per_model'] = 5  # Reduced for demo
            config['search']['cv_folds'] = 3
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("⚙️ Initializing AutoML...")
                progress_bar.progress(10)
                
                # Run AutoML
                orchestrator = AutoMLOrchestrator(config=config)
                
                status_text.text("🔍 Analyzing data...")
                progress_bar.progress(20)
                
                results = orchestrator.run(
                    data_path=str(temp_csv),
                    target_column=target_column,
                    output_dir=str(tmp_path / "output"),
                    metric=metric_choice
                )
                
                progress_bar.progress(100)
                status_text.text("✅ AutoML complete!")
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Results")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Problem Type",
                        results['summary']['problem_type']
                    )
                
                with col2:
                    st.metric(
                        "Best Model",
                        results['summary']['recommended_model']
                    )
                
                with col3:
                    metric_name = results['summary']['primary_metric'].upper()
                    metric_value = results['summary']['validation_metric']
                    st.metric(
                        metric_name,
                        f"{metric_value:.4f}"
                    )
                
                with col4:
                    st.metric(
                        "Models Tried",
                        len(results['leaderboard'])
                    )
                
                # Model Selection Explanation
                st.markdown("#### 🧠 Model Selection")
                selection_info = results.get('selection_info', {})
                reason = selection_info.get('reason_text', 'Selected based on validation metric')
                
                st.info(f"**{results['summary']['recommended_model']}** was chosen because it {reason}")
                
                if selection_info.get('num_contenders', 0) > 1:
                    st.caption(f"ℹ️ {selection_info['num_contenders']} models were within {selection_info.get('tolerance', 0)*100:.1f}% of the best metric")
                
                # Leaderboard
                st.markdown("#### 📊 Model Leaderboard")
                
                leaderboard_df = pd.DataFrame(results['leaderboard'])
                
                # Format display
                display_cols = ['model_name', 'metric_value', 'cv_mean', 'training_time']
                if all(col in leaderboard_df.columns for col in display_cols):
                    display_df = leaderboard_df[display_cols].copy()
                    display_df.columns = ['Model', 'Validation Score', 'CV Score', 'Training Time (s)']
                    display_df = display_df.round(4)
                    display_df = display_df.sort_values('Validation Score', ascending=False)
                    
                    # Highlight best model
                    st.dataframe(
                        display_df.head(10),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.dataframe(leaderboard_df.head(10), use_container_width=True)
                
                # Feature Importance
                if 'feature_importance' in results and results['feature_importance']:
                    st.markdown("#### 🎯 Feature Importance")
                    
                    # Convert to DataFrame
                    fi_data = []
                    for feat, importance in results['feature_importance'].items():
                        fi_data.append({
                            'Feature': feat,
                            'Importance': importance
                        })
                    
                    fi_df = pd.DataFrame(fi_data).sort_values('Importance', ascending=False).head(10)
                    
                    # Bar chart
                    st.bar_chart(fi_df.set_index('Feature')['Importance'])
                
                # Model Family Summary
                if 'families' in results and results['families']:
                    st.markdown("#### 🏆 Model Family Performance")
                    
                    families_data = []
                    for family, stats in results['families'].items():
                        families_data.append({
                            'Family': family.replace('_', ' ').title(),
                            'Best Score': f"{stats['best_score']:.4f}",
                            'Avg Score': f"{stats['mean_score']:.4f}",
                            'Trials': stats['count']
                        })
                    
                    families_df = pd.DataFrame(families_data)
                    st.dataframe(families_df, use_container_width=True, hide_index=True)
                
                # Download model
                st.markdown("---")
                st.markdown("#### 💾 Download Trained Model")
                
                model_path = results.get('model_path')
                if model_path and Path(model_path).exists():
                    with open(model_path, 'rb') as f:
                        model_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download Model (.joblib)",
                        data=model_bytes,
                        file_name=f"{data_name}_model.joblib",
                        mime="application/octet-stream",
                        help="Download the trained model for use in your own code"
                    )
                    
                    # Usage instructions
                    with st.expander("📖 How to use the downloaded model"):
                        st.code("""
import joblib
import pandas as pd

# Load model
artifacts = joblib.load('model.joblib')
pipeline = artifacts['pipeline']
label_encoder = artifacts.get('label_encoder')

# Make predictions
new_data = pd.read_csv('your_data.csv')
predictions = pipeline.predict(new_data)

# For classification, convert back to original labels
if label_encoder:
    predictions = label_encoder.inverse_transform(predictions)

# Get probabilities (classification only)
if hasattr(pipeline, 'predict_proba'):
    probabilities = pipeline.predict_proba(new_data)
    class_labels = artifacts['class_labels']
    print(class_labels)  # Column names for probabilities
""", language='python')
                
            except Exception as e:
                st.error(f"❌ Error running AutoML: {str(e)}")
                st.exception(e)

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to AutoML Tabular!
    
    This is an **explainable AutoML engine** that automatically:
    - ✅ Detects problem type (binary, multiclass, regression)
    - ✅ Preprocesses your data (handles missing values, encodes categories)
    - ✅ Searches for the best model using Optuna
    - ✅ Explains model selection with tolerance-based comparison
    - ✅ Provides feature importance and performance metrics
    
    ### 🚀 Get Started
    
    1. **Upload your CSV** in the sidebar (or try an example dataset)
    2. **Select your target column** (what you want to predict)
    3. **Choose a metric** (e.g., ROC AUC for classification, RMSE for regression)
    4. **Click "Run AutoML"** and watch the magic happen!
    
    ### 📚 Features
    
    - **Smart Model Selection**: Not just the highest metric—considers simplicity when models are close
    - **Generic Label Encoding**: Works for any classification task (binary or multiclass)
    - **Professional Reports**: Full HTML reports with explanations
    - **Production-Ready**: Models saved with all metadata for deployment
    
    ---
    
    ### 🔗 Links
    
    - **GitHub**: [github.com/Mounusha25/automl-tabular](https://github.com/Mounusha25/automl-tabular)
    - **Sample Reports**: [View Examples](https://github.com/Mounusha25/automl-tabular/tree/main/examples/sample_reports)
    
    ---
    
    _Built by Mounusha Ram Metti • Powered by scikit-learn, XGBoost, LightGBM, and Optuna_
    """)
    
    # Show example datasets info
    st.markdown("### 📊 Example Datasets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🚢 Titanic**
        - Type: Binary Classification
        - Rows: 891
        - Target: Survived
        """)
    
    with col2:
        st.markdown("""
        **🍷 Wine Quality**
        - Type: Multiclass (6 classes)
        - Rows: 1,599
        - Target: quality
        """)
    
    with col3:
        st.markdown("""
        **🏠 California Housing**
        - Type: Regression
        - Rows: 20,640
        - Target: MedHouseVal
        """)
