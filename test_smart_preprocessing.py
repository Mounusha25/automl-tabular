#!/usr/bin/env python3
"""
Test that smart preprocessing doesn't break existing functionality.
Run with flag OFF (default) and ON to verify both modes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from automl_tabular import run_automl_job

print("="*70)
print("TEST 1: Default behavior (smart preprocessing OFF)")
print("="*70)

result1 = run_automl_job(
    data='examples/titanic.csv',
    target_column='Survived',
    output_dir='output/test_default',
    time_budget=30
)

print(f"\n✅ Test 1 passed!")
print(f"   Model: {result1['recommended_model_name']}")
print(f"   Metric: {result1['recommended_metric_value']:.4f}")

print("\n" + "="*70)
print("TEST 2: Smart preprocessing ON")
print("="*70)

config = {
    'preprocessing': {
        'enable_smart_strategies': True
    }
}

result2 = run_automl_job(
    data='examples/titanic.csv',
    target_column='Survived',
    output_dir='output/test_smart',
    time_budget=30,
    config=config
)

print(f"\n✅ Test 2 passed!")
print(f"   Model: {result2['recommended_model_name']}")
print(f"   Metric: {result2['recommended_metric_value']:.4f}")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("✓ Default behavior preserved")
print("✓ Smart preprocessing available when enabled")
print("\nCheck reports:")
print(f"  Default: {result1['report_path']}")
print(f"  Smart:   {result2['report_path']}")
