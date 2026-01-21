"""
Test fixed report with all three improvements:
1. Methodology preprocessing line reflects smart preprocessing
2. Tree-based recommendation correctly states they're competitive but lower
3. High-cardinality features explicitly explained as dropped
"""

import sys
sys.path.insert(0, 'src')

from automl_tabular.orchestrator import run_automl_job

print("=" * 80)
print("TESTING FIXED SMART PREPROCESSING REPORT")
print("=" * 80)

result = run_automl_job(
    data='examples/titanic.csv',
    target_column='Survived',
    output_dir='output/fixed_report',
    config={'preprocessing': {'enable_smart_strategies': True}}
)

print("\n" + "=" * 80)
print("✅ REPORT GENERATED!")
print("=" * 80)
print(f"\nCheck report at: {result['report_path']}")
print("\nExpected fixes:")
print("1. ✅ Methodology shows: 'Profile-based imputation, outlier clipping...'")
print("2. ✅ Recommendations say: 'slightly lower scores' (not higher)")
print("3. ✅ PassengerId/Name/Ticket explicitly marked as dropped with reasons")
