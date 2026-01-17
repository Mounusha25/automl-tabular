"""
Quick demo of AutoML Tabular.

This script demonstrates the basic usage of the AutoML system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from automl_tabular import run_automl_job


def demo_classification():
    """Demo classification on loan approval dataset."""
    print("\n" + "="*80)
    print("DEMO 1: Classification (Loan Approval Prediction)")
    print("="*80)
    
    results = run_automl_job(
        data_path="examples/example_classification.csv",
        target_column="loan_approved",
        output_dir="demo_output/classification",
        metric="roc_auc"
    )
    
    print("\n✅ Classification demo completed!")
    print(f"Report: {results['report_path']}")
    print(f"Model: {results['model_path']}")


def demo_regression():
    """Demo regression on house price dataset."""
    print("\n" + "="*80)
    print("DEMO 2: Regression (House Price Prediction)")
    print("="*80)
    
    results = run_automl_job(
        data_path="examples/example_regression.csv",
        target_column="price",
        output_dir="demo_output/regression",
        metric="rmse"
    )
    
    print("\n✅ Regression demo completed!")
    print(f"Report: {results['report_path']}")
    print(f"Model: {results['model_path']}")


if __name__ == "__main__":
    print("\n🤖 AutoML Tabular - Quick Demo")
    print("This demo will run AutoML on two example datasets.\n")
    
    try:
        # Classification
        demo_classification()
        
        # Regression
        demo_regression()
        
        print("\n" + "="*80)
        print("✅ ALL DEMOS COMPLETED!")
        print("="*80)
        print("\nCheck the demo_output/ directory for:")
        print("  - Trained models (.joblib files)")
        print("  - HTML reports (open in browser)")
        print("  - Visualization plots (in reports/plots/)")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
