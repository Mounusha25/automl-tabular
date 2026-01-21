"""
Performance Benchmark Script
=============================

Tests the impact of performance optimizations on runtime.

Layers tested:
1. Cached profiling
2. Optuna pruning + dynamic CV folds
3. Sparse matrices + float32
"""

import sys
import time
sys.path.insert(0, 'src')

from automl_tabular.orchestrator import run_automl_job

def benchmark_config(name, config, data_path='examples/titanic.csv'):
    """Run AutoML with given config and measure time."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*80}")
    
    start = time.time()
    
    result = run_automl_job(
        data=data_path,
        target_column='Survived',
        output_dir=f'output/benchmark_{name.lower().replace(" ", "_")}',
        config=config
    )
    
    elapsed = time.time() - start
    
    model_name = result.get('model_name', result.get('best_model', 'Unknown'))
    score = result.get('metric_value', result.get('score', 0))
    
    print(f"\n{'='*80}")
    print(f"✅ {name} completed in {elapsed:.1f}s")
    print(f"   Best model: {model_name}")
    print(f"   Score: {score:.4f}")
    print(f"{'='*80}\n")
    
    return elapsed, score


if __name__ == "__main__":
    results = {}
    
    # Baseline: All optimizations OFF
    print("\n" + "="*80)
    print("BASELINE: All Optimizations OFF")
    print("="*80)
    baseline_config = {
        'search': {
            'max_trials_per_model': 10,  # Reduce for faster benchmark
            'enable_pruning': False
        },
        'experiment': {
            'n_splits': 3  # Fixed CV folds
        },
        'preprocessing': {
            'use_sparse_matrices': False,
            'use_float32': False,
            'enable_smart_strategies': False
        }
    }
    results['baseline'] = benchmark_config("Baseline (no optimizations)", baseline_config)
    
    # Layer 1+2: Pruning + Dynamic CV
    print("\n" + "="*80)
    print("LAYER 1+2: Pruning + Dynamic CV Folds")
    print("="*80)
    layer12_config = {
        'search': {
            'max_trials_per_model': 10,
            'enable_pruning': True,
            'auto_adjust_cv_folds': True
        },
        'experiment': {
            'n_splits': 3
        },
        'preprocessing': {
            'use_sparse_matrices': False,
            'use_float32': False,
            'enable_smart_strategies': False
        }
    }
    results['layer12'] = benchmark_config("Pruning + Dynamic CV", layer12_config)
    
    # Layer 3: Add Data Optimizations
    print("\n" + "="*80)
    print("LAYER 3: Add Sparse Matrices + Float32")
    print("="*80)
    layer3_config = {
        'search': {
            'max_trials_per_model': 10,
            'enable_pruning': True,
            'auto_adjust_cv_folds': True
        },
        'experiment': {
            'n_splits': 3
        },
        'preprocessing': {
            'use_sparse_matrices': True,
            'use_float32': True,
            'enable_smart_strategies': False
        }
    }
    results['layer3'] = benchmark_config("All optimizations", layer3_config)
    
    # Full: Add Smart Preprocessing
    print("\n" + "="*80)
    print("FULL: All Optimizations + Smart Preprocessing")
    print("="*80)
    full_config = {
        'search': {
            'max_trials_per_model': 10,
            'enable_pruning': True,
            'auto_adjust_cv_folds': True
        },
        'experiment': {
            'n_splits': 3
        },
        'preprocessing': {
            'use_sparse_matrices': True,
            'use_float32': True,
            'enable_smart_strategies': True
        }
    }
    results['full'] = benchmark_config("All + Smart Preprocessing", full_config)
    
    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    baseline_time = results['baseline'][0]
    
    for name, (elapsed, score) in results.items():
        speedup = baseline_time / elapsed
        print(f"\n{name:30s}")
        print(f"  Time:    {elapsed:6.1f}s  (speedup: {speedup:.2f}x)")
        print(f"  Score:   {score:.4f}")
    
    print("\n" + "="*80)
    print(f"🚀 Total speedup: {baseline_time / results['full'][0]:.2f}x faster")
    print("="*80)
