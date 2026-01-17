"""
Orchestrator - High-level AutoML pipeline coordinator.

This module ties together all components to run the complete AutoML workflow.
"""

import os
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from automl_tabular.config import load_default_config, merge_configs
from automl_tabular.data import DataLoader, DataValidator
from automl_tabular.preprocessing import (
    TabularPreprocessor,
    remove_constant_columns,
    remove_high_missing_columns
)
from automl_tabular.models import ModelFactory, ModelSearcher, get_default_param_space
from automl_tabular.models.algorithms import MODEL_SIMPLICITY
from automl_tabular.evaluation import (
    get_metric_function,
    compute_metrics,
    get_primary_metric,
    split_train_validation,
    Leaderboard
)
from automl_tabular.explainability import (
    get_feature_importance,
    normalize_importance,
    plot_feature_importance,
    plot_target_distribution,
    generate_model_summary,
    generate_feature_importance_summary,
    generate_data_summary,
    generate_recommendations
)
from automl_tabular.reporting import ReportBuilder


class AutoMLOrchestrator:
    """Orchestrates the complete AutoML pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize orchestrator.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        # Load and merge configs
        self.config = load_default_config()
        if config:
            self.config = merge_configs(self.config, config)
        
        # Initialize components
        self.loader = DataLoader()
        self.validator = DataValidator()
        self.preprocessor = None
        self.searcher = None
        self.leaderboard = None
        self.label_encoder = None  # For classification tasks
        
        # Results
        self.results = {}
    
    def _prepare_target(self, y, problem_type):
        """
        Prepare target variable for modeling.
        
        For classification: label-encode to 0..n-1 for compatibility with all algorithms.
        For regression: leave as-is.
        
        Args:
            y: Target series
            problem_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (encoded_target, label_encoder or None)
        """
        if problem_type == "classification":
            # Always encode classification labels to ensure 0..n-1 range
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            return y_encoded, le
        else:
            # Regression - use as-is
            return y.values, None
    
    def _select_recommended_model(self, leaderboard, tolerance: float, strategy: str, metric_name: str):
        """
        Select recommended model using tolerance-based selection with simplicity tie-breaking.
        
        Args:
            leaderboard: Leaderboard object with results
            tolerance: Tolerance threshold for near-ties
            strategy: 'most_accurate' or 'simple_if_close'
            metric_name: Name of the primary metric for dynamic text generation
            
        Returns:
            Tuple of (recommended_model, top_contenders, selection_info)
        """
        best = leaderboard.get_best_model()
        top_contenders = leaderboard.get_top_contenders(tolerance=tolerance)
        
        # If only one contender or strategy is most_accurate, return best
        if len(top_contenders) <= 1 or strategy == 'most_accurate':
            reason_text = f"because it achieved the highest validation {metric_name}."
            return best, top_contenders, {
                'is_tie': False,
                'margin': 0.0,
                'tolerance': tolerance,
                'tolerance_percent': tolerance * 100.0,
                'reason': 'clear_winner',
                'reason_text': reason_text,
                'num_contenders': len(top_contenders),
                'best_by_metric': best.model_name,
                'best_metric_value': best.score,
                'recommended_model_name': best.model_name,
                'recommended_metric_value': best.score,
                'metric_name': metric_name
            }
        
        # Multiple contenders - check if we should prefer simpler model
        if strategy == 'simple_if_close':
            def get_simplicity(model):
                return MODEL_SIMPLICITY.get(model.model_name, 999)
            
            # Choose simplest among contenders
            recommended = min(top_contenders, key=get_simplicity)
            
            # Calculate margin from best
            margin = abs(best.score - recommended.score)
            
            # Generate conditional reason text
            if recommended.model_name == best.model_name:
                reason_text = f"because it achieved the highest validation {metric_name}."
            else:
                reason_text = (
                    f"because its performance is within the tolerance of the best model "
                    f"while being simpler and more interpretable."
                )
            
            return recommended, top_contenders, {
                'is_tie': True,
                'margin': margin,
                'tolerance': tolerance,
                'tolerance_percent': tolerance * 100.0,
                'reason': 'simplicity_preferred',
                'reason_text': reason_text,
                'num_contenders': len(top_contenders),
                'best_by_metric': best.model_name,
                'best_metric_value': best.score,
                'recommended_model_name': recommended.model_name,
                'recommended_metric_value': recommended.score,
                'metric_name': metric_name
            }
        
        reason_text = f"because it achieved the highest validation {metric_name}."
        return best, top_contenders, {
            'is_tie': False,
            'margin': 0.0,
            'tolerance': tolerance,
            'tolerance_percent': tolerance * 100.0,
            'reason': 'default',
            'reason_text': reason_text,
            'num_contenders': len(top_contenders),
            'best_by_metric': best.model_name,
            'best_metric_value': best.score,
            'recommended_model_name': best.model_name,
            'recommended_metric_value': best.score,
            'metric_name': metric_name
        }
    
    def run(
        self,
        data_path: str,
        target_column: str,
        output_dir: str = "output",
        metric: str = "auto",
        report_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete AutoML pipeline.
        
        Args:
            data_path: Path to CSV data file
            target_column: Name of target column
            output_dir: Directory for outputs
            metric: Primary metric ('auto' or specific metric name)
            report_name: Optional custom report name
            
        Returns:
            Dictionary with results and paths
        """
        print("="*80)
        print("🤖 AutoML Pipeline Started")
        print("="*80)
        
        # Create output directories
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        plots_dir = reports_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Load and validate data
        print("\n📁 Step 1: Loading and validating data...")
        df = self.loader.load_csv(data_path)
        print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate
        is_valid, errors, warnings_list = self.validator.validate(df, target_column)
        
        if not is_valid:
            print("\n❌ Validation failed:")
            for error in errors:
                print(f"   ERROR: {error}")
            raise ValueError("Data validation failed")
        
        if warnings_list:
            print("\n⚠️  Warnings:")
            for warning in warnings_list:
                print(f"   {warning}")
        
        # Get data summary
        data_info = self.loader.get_data_summary(df)
        
        # Step 2: Infer problem type
        print("\n🎯 Step 2: Analyzing problem type...")
        schema = self.loader.infer_column_types(df)
        problem_type = self.loader.infer_problem_type(df[target_column])
        
        # Display problem type with more detail for classification
        if problem_type == 'classification':
            n_classes = df[target_column].nunique()
            if n_classes == 2:
                problem_type_display = "BINARY CLASSIFICATION"
            else:
                problem_type_display = f"MULTICLASS CLASSIFICATION ({n_classes} classes)"
        else:
            problem_type_display = problem_type.upper()
        
        print(f"   Problem type: {problem_type_display}")
        
        # Step 3: Preprocessing
        print("\n🔧 Step 3: Preprocessing data...")
        
        # Remove problematic columns
        df_clean = remove_constant_columns(df, target_column)
        df_clean = remove_high_missing_columns(df_clean, target_column, threshold=0.95)
        
        print(f"   Features after cleaning: {len(df_clean.columns) - 1}")
        
        # Separate features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Prepare target (label encode for classification)
        y_prepared, self.label_encoder = self._prepare_target(y, problem_type)
        
        # Update schema for cleaned data
        schema = self.loader.infer_column_types(X)
        numeric_features = schema['numeric']
        categorical_features = schema['categorical']
        
        print(f"   Numeric features: {len(numeric_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        
        # Build preprocessing pipeline
        self.preprocessor = TabularPreprocessor(self.config.get('preprocessing', {}))
        pipeline = self.preprocessor.build_pipeline(numeric_features, categorical_features)
        
        # Split data
        random_state = self.config['experiment']['random_seed']
        test_size = self.config['experiment']['test_size']
        
        X_train, X_val, y_train, y_val = split_train_validation(
            X, y_prepared, test_size=test_size,
            problem_type=problem_type,
            random_state=random_state
        )
        
        print(f"   Train set: {len(X_train)} samples")
        print(f"   Validation set: {len(X_val)} samples")
        
        # Fit preprocessing
        X_train_processed = self.preprocessor.fit_transform(X_train, y_train)
        X_val_processed = self.preprocessor.transform(X_val)
        
        print(f"   Processed features shape: {X_train_processed.shape}")
        
        # Step 4: Model search
        print("\n🔍 Step 4: Searching for best models...")
        
        # Get metric
        if metric == "auto":
            metric_name = get_primary_metric(problem_type, self.config)
        else:
            metric_name = metric
        
        metric_func = get_metric_function(metric_name, problem_type)
        
        # Get model list
        if problem_type == 'classification':
            model_configs = self.config['models']['classifiers']
        else:
            model_configs = self.config['models']['regressors']
        
        model_names = [m['name'] for m in model_configs if m.get('enabled', True)]
        print(f"   Models to try: {', '.join(model_names)}")
        
        # Search
        self.searcher = ModelSearcher(
            problem_type=problem_type,
            metric_func=metric_func,
            max_trials_per_model=self.config['search']['max_trials_per_model'],
            time_limit_seconds=self.config['search'].get('time_limit_seconds'),
            cv_folds=self.config['experiment']['n_splits'],
            random_state=random_state
        )
        
        results = self.searcher.search(
            X_train_processed, y_train,
            model_names=model_names,
            use_optuna=True
        )
        
        if not results:
            raise RuntimeError("No models were successfully trained")
        
        print(f"\n   ✅ Completed! Tried {len(results)} model configurations")
        
        # Step 5: Build leaderboard
        print("\n🏆 Step 5: Building leaderboard...")
        self.leaderboard = Leaderboard(results)
        self.leaderboard.display(top_k=10)
        
        # Step 5.5: Smart model selection with tolerance-based tie-breaking
        tolerance = self.config.get('selection', {}).get('tolerance', 0.005)
        strategy = self.config.get('selection', {}).get('strategy', 'simple_if_close')
        
        recommended_model, top_contenders, selection_info = self._select_recommended_model(
            self.leaderboard, tolerance, strategy, metric_name
        )
        
        # Display selection reasoning
        if selection_info['is_tie']:
            print(f"\n💡 Model Selection:")
            print(f"   Found {selection_info['num_contenders']} models within {tolerance:.1%} performance")
            print(f"   Best by metric: {self.leaderboard.get_best_model().model_name} ({self.leaderboard.get_best_model().score:.4f})")
            print(f"   Recommended: {recommended_model.model_name} ({recommended_model.score:.4f})")
            print(f"   Reason: Simpler model with comparable performance (margin: {selection_info['margin']:.4f})")
        else:
            print(f"\n   Selected model: {recommended_model.model_name}")
        
        best_result = recommended_model
        
        # Step 6: Evaluate recommended model on validation set
        print(f"\n📊 Step 6: Evaluating recommended model ({best_result.model_name})...")
        
        # Retrain best model on full training set
        if problem_type == 'classification':
            best_model = ModelFactory.create_classifier(
                best_result.model_name,
                **best_result.params
            )
        else:
            best_model = ModelFactory.create_regressor(
                best_result.model_name,
                **best_result.params
            )
        
        best_model.fit(X_train_processed, y_train)
        
        # Predict on validation set
        y_val_pred = best_model.predict(X_val_processed)
        
        # Inverse transform predictions if label encoder was used
        if self.label_encoder is not None:
            y_val_pred_original = self.label_encoder.inverse_transform(y_val_pred)
            # Use original labels for display/confusion matrix
            y_val_original = self.label_encoder.inverse_transform(y_val)
        else:
            y_val_pred_original = y_val_pred
            y_val_original = y_val
        
        # Get probabilities for classification
        y_val_proba = None
        class_labels = None
        if problem_type == 'classification' and hasattr(best_model, 'predict_proba'):
            y_val_proba = best_model.predict_proba(X_val_processed)
            # Track class labels in encoded order
            if self.label_encoder is not None:
                class_labels = self.label_encoder.classes_
            else:
                class_labels = np.arange(y_val_proba.shape[1]) if y_val_proba is not None else None
        
        # Compute metrics on holdout for reference
        holdout_metrics = compute_metrics(y_val, y_val_pred, problem_type, y_val_proba)
        
        print("\n   Holdout Metrics (for reference):")
        for metric_key, value in holdout_metrics.items():
            print(f"   - {metric_key}: {value:.4f}")
        
        # Use the CV score from leaderboard as the primary metric (consistent with leaderboard)
        primary_metric_value = best_result.score
        
        # Step 7: Explainability
        print("\n🔍 Step 7: Generating explanations...")
        
        # Feature importance
        feature_names = self.preprocessor.get_feature_names()
        feature_importance_df = get_feature_importance(
            best_model, feature_names,
            X_val_processed, y_val,
            method='auto'
        )
        
        feature_importance_df = normalize_importance(feature_importance_df)
        
        # Aggregate by original column for better interpretability
        from automl_tabular.explainability.feature_importance import aggregate_importance_by_column
        feature_importance_agg = aggregate_importance_by_column(
            feature_importance_df,
            exclude_high_cardinality=True,
            high_cardinality_threshold=50
        )
        
        # Use aggregated if available, otherwise use raw
        display_importance = feature_importance_agg if not feature_importance_agg.empty else feature_importance_df
        
        print(f"   Top 5 features:")
        for i, row in display_importance.head(5).iterrows():
            feature_name = row['feature']
            if 'num_features' in row and row['num_features'] > 1:
                print(f"   {i+1}. {feature_name} ({int(row['num_features'])} encoded features): {row['importance']:.4f}")
            else:
                print(f"   {i+1}. {feature_name}: {row['importance']:.4f}")
        
        # Generate plots
        print("\n📈 Step 8: Creating visualizations...")
        
        plots = {}
        
        # Feature importance plot
        if not feature_importance_df.empty:
            plot_path = plots_dir / f"{run_id}_feature_importance.png"
            # Use aggregated importance for the plot if available
            plot_data = display_importance if not display_importance.empty else feature_importance_df
            plots['feature_importance'] = plot_feature_importance(
                plot_data,
                top_k=15,
                save_path=str(plot_path),
                dpi=self.config['reporting']['plot_dpi']
            )
            print(f"   ✅ Feature importance plot saved")
        
        # Target distribution plot
        plot_path = plots_dir / f"{run_id}_target_distribution.png"
        plots['target_distribution'] = plot_target_distribution(
            y.values,
            problem_type,
            save_path=str(plot_path),
            dpi=self.config['reporting']['plot_dpi']
        )
        print(f"   ✅ Target distribution plot saved")
        
        # Model comparison plot
        from automl_tabular.explainability.plots import plot_model_comparison, plot_missing_data, plot_confusion_matrix
        
        plot_path = plots_dir / f"{run_id}_model_comparison.png"
        plots['model_comparison'] = plot_model_comparison(
            self.leaderboard.df,
            metric_name='score',
            save_path=str(plot_path),
            dpi=self.config['reporting']['plot_dpi']
        )
        print(f"   ✅ Model comparison plot saved")
        
        # Missing data plot
        plot_path = plots_dir / f"{run_id}_missing_data.png"
        missing_plot = plot_missing_data(
            df,
            save_path=str(plot_path),
            dpi=self.config['reporting']['plot_dpi']
        )
        if missing_plot:
            plots['missing_data'] = missing_plot
            print(f"   ✅ Missing data plot saved")
        
        # Confusion matrix for classification
        if problem_type == 'classification':
            plot_path = plots_dir / f"{run_id}_confusion_matrix.png"
            # Use original (non-encoded) labels for confusion matrix display
            plots['confusion_matrix'] = plot_confusion_matrix(
                y_val_original if self.label_encoder is not None else y_val,
                y_val_pred_original if self.label_encoder is not None else y_val_pred,
                save_path=str(plot_path),
                dpi=self.config['reporting']['plot_dpi']
            )
            print(f"   ✅ Confusion matrix plot saved")
        
        # Step 9: Generate text summaries
        print("\n📝 Step 9: Generating summaries...")
        
        model_summary = generate_model_summary(
            best_result,
            problem_type,
            metric_name,
            best_result.score  # Use CV score from leaderboard
        )
        
        # Use aggregated importance for text summary to match the plot
        feature_importance_summary = generate_feature_importance_summary(
            display_importance,  # Use aggregated importance
            top_k=8
        )
        
        data_summary_text = generate_data_summary(data_info)
        
        recommendations = generate_recommendations(
            problem_type,
            best_result,
            best_result.score,  # Use CV score from leaderboard
            data_info
        )
        
        # Step 10: Save model
        print("\n💾 Step 10: Saving model...")
        
        # Create full pipeline (preprocessing + model)
        from sklearn.pipeline import Pipeline as SkPipeline
        
        full_pipeline = SkPipeline([
            ('preprocessor', self.preprocessor.pipeline),
            ('model', best_model)
        ])
        
        # Save model with label encoder (for classification tasks)
        model_artifacts = {
            'pipeline': full_pipeline,
            'label_encoder': self.label_encoder,
            'problem_type': problem_type,
            'feature_names': X.columns.tolist(),
            'target_column': target_column,
            'class_labels': class_labels.tolist() if class_labels is not None else None,
            'metric_name': metric_name
        }
        
        model_path = models_dir / f"{run_id}_model.joblib"
        joblib.dump(model_artifacts, model_path)
        print(f"   ✅ Model saved to: {model_path}")
        
        # Step 11: Generate report
        print("\n📄 Step 11: Generating HTML report...")
        
        report_builder = ReportBuilder(output_dir=str(reports_dir))
        
        # Get model family summary
        model_family_summary = self.leaderboard.get_model_family_summary()
        
        context = report_builder.prepare_context(
            data_info=data_info,
            problem_type=problem_type,
            target_column=target_column,
            leaderboard_data=self.leaderboard.to_dict(tolerance=tolerance),
            best_model_info={
                'model_name': best_result.model_name,
                'metric_name': metric_name,
                'metric_value': best_result.score  # Use CV score from leaderboard
            },
            feature_importance_df=display_importance,  # Use aggregated importance
            plots=plots,
            warnings=warnings_list,
            model_summary=model_summary,
            feature_importance_summary=feature_importance_summary,
            recommendations=recommendations,
            model_path=str(model_path),
            selection_info=selection_info,
            model_family_summary=model_family_summary,
            top_contenders=[{
                'model_name': m.model_name,
                'score': m.score,
                'simplicity': MODEL_SIMPLICITY.get(m.model_name, 999)
            } for m in top_contenders]
        )
        
        context['run_id'] = run_id
        
        if report_name is None:
            report_name = f"automl_report_{run_id}.html"
        
        report_path = report_builder.build_report(context, report_name)
        print(f"   ✅ Report saved to: {report_path}")
        
        # Summary
        print("\n" + "="*80)
        print("✅ AutoML Pipeline Completed Successfully!")
        print("="*80)
        print(f"\n📊 Best Model: {best_result.model_name}")
        print(f"📈 {metric_name.upper()}: {best_result.score:.4f}")
        print(f"\n📁 Outputs:")
        print(f"   - Model: {model_path}")
        print(f"   - Report: {report_path}")
        print("\n")
        
        # Store results
        self.results = {
            'run_id': run_id,
            'problem_type': problem_type,
            'best_model_name': best_result.model_name,
            'best_params': best_result.params,
            'metric_name': metric_name,
            'metric_value': best_result.score,
            'all_metrics': holdout_metrics,
            'model_path': str(model_path),
            'report_path': report_path,
            'leaderboard': self.leaderboard.to_dict(),
            'feature_importance': feature_importance_df.to_dict(orient='records')
        }
        
        return self.results


def run_automl_job(
    data_path: str,
    target_column: str,
    output_dir: str = "output",
    config: Optional[Dict] = None,
    metric: str = "auto",
    report_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run AutoML pipeline.
    
    Args:
        data_path: Path to CSV data file
        target_column: Name of target column
        output_dir: Directory for outputs
        config: Optional configuration dictionary
        metric: Primary metric ('auto' or specific metric name)
        report_name: Optional custom report name
        
    Returns:
        Dictionary with results and paths
    """
    orchestrator = AutoMLOrchestrator(config=config)
    return orchestrator.run(
        data_path=data_path,
        target_column=target_column,
        output_dir=output_dir,
        metric=metric,
        report_name=report_name
    )


__all__ = ["AutoMLOrchestrator", "run_automl_job"]
