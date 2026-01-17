#!/usr/bin/env python3
"""
Simple runner for AutoML - works without installation.
Just sets up the Python path and runs AutoML.
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import and run
from automl_tabular import run_automl_job

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AutoML on a dataset")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--metric", default="auto", help="Primary metric")
    parser.add_argument("--algorithms", help="Comma-separated list of algorithms (e.g., logistic_regression,random_forest)")
    
    args = parser.parse_args()
    
    print("\n🤖 Running AutoML...\n")
    
    # Build config if algorithms specified
    config = None
    if args.algorithms:
        algo_list = [a.strip() for a in args.algorithms.split(',')]
        config = {'models': {'algorithms': algo_list}}
        print(f"Using algorithms: {', '.join(algo_list)}\n")
    
    results = run_automl_job(
        data_path=args.data,
        target_column=args.target,
        output_dir=args.output,
        metric=args.metric,
        config=config
    )
    
    print("\n✅ Done!")
    print(f"Report: {results['report_path']}")
    print(f"Model: {results['model_path']}\n")
