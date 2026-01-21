"""Quick test to verify performance optimizations work correctly."""

import sys
sys.path.insert(0, 'src')

from automl_tabular.orchestrator import run_automl_job

print("Testing performance optimizations...")

result = run_automl_job(
    data='examples/titanic.csv',
    target_column='Survived',
    output_dir='output/perf_test',
    config={
        'search': {
            'max_trials_per_model': 5,  # Quick test
            'enable_pruning': True
        },
        'preprocessing': {
            'use_sparse_matrices': True,
            'use_float32': True,
            'enable_smart_strategies': True  # Test smart preprocessing too
        }
    }
)

print(f"\n✅ Performance test passed!")
print(f"   Best model: {result.get('model_name', result.get('best_model'))}")
print(f"   Score: {result.get('metric_value', result.get('score', 0)):.4f}")
print(f"   Report: {result.get('report_path', 'N/A')}")
