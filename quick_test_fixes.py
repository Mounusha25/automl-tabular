"""
Quick test to verify all three fixes are working:
1. Methodology preprocessing line
2. Tree-based recommendation
3. Dropped features shown in preprocessing decisions
"""

import sys
sys.path.insert(0, 'src')

from automl_tabular.orchestrator import run_automl_job

print("Generating quick test report with all fixes...")

result = run_automl_job(
    data='examples/titanic.csv',
    target_column='Survived',
    output_dir='output/final_fixed',
    config={
        'preprocessing': {'enable_smart_strategies': True},
        'training': {'cv_folds': 2},  # Faster
        'search': {'optuna_trials': 20}  # Faster
    }
)

print(f"\n✅ Report: {result['report_path']}")
print("\nVerify:")
print("1. Methodology shows 'Profile-based imputation...'")
print("2. Recommendations say tree models are 'slightly lower' (not higher)")  
print("3. PassengerId/Name/Ticket appear in preprocessing decisions as dropped")
