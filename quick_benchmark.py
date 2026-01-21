"""
Quick Performance Benchmark
============================

Compares baseline vs optimized performance on Titanic dataset.
"""

import sys
import time
sys.path.insert(0, 'src')

from automl_tabular.orchestrator import run_automl_job

def quick_benchmark(name, config):
    """Run AutoML with given config and measure time."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    start = time.time()
    
    result = run_automl_job(
        data='examples/titanic.csv',
        target_column='Survived',
        output_dir=f'output/bench_{name.lower().replace(" ", "_").replace("+", "").strip()}',
        config=config
    )
    
    elapsed = time.time() - start
    model_name = result.get('model_name', result.get('best_model', 'Unknown'))
    score = result.get('metric_value', result.get('score', 0))
    
    print(f"\n✅ Time: {elapsed:.1f}s | Model: {model_name} | Score: {score:.4f}\n")
    
    return elapsed, score

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)
    print("\nDataset: Titanic (891 rows)")
    print("Trials: 5 per model family (quick test)\n")
    
    results = {}
    
    # Baseline: Optimizations OFF
    print("\n" + "="*70)
    print("BASELINE (No Optimizations)")
    print("="*70)
    baseline_cfg = {
        'search': {'max_trials_per_model': 5, 'enable_pruning': False},
        'preprocessing': {'use_sparse_matrices': False, 'use_float32': False}
    }
    results['baseline'] = quick_benchmark("Baseline", baseline_cfg)
    
    # Optimized: All ON
    print("\n" + "="*70)
    print("OPTIMIZED (Pruning + Sparse + Float32)")
    print("="*70)
    optimized_cfg = {
        'search': {'max_trials_per_model': 5, 'enable_pruning': True},
        'preprocessing': {'use_sparse_matrices': True, 'use_float32': True}
    }
    results['optimized'] = quick_benchmark("Optimized", optimized_cfg)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    baseline_time, baseline_score = results['baseline']
    opt_time, opt_score = results['optimized']
    speedup = baseline_time / opt_time
    
    print(f"\nBaseline:   {baseline_time:6.1f}s  |  Score: {baseline_score:.4f}")
    print(f"Optimized:  {opt_time:6.1f}s  |  Score: {opt_score:.4f}")
    print(f"\n🚀 Speedup: {speedup:.2f}x faster")
    print(f"✅ Score preserved: {abs(baseline_score - opt_score) < 0.01}")
    print("\n" + "="*70)
