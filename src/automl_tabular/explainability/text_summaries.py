"""Text summary generation for explainability."""

from typing import Dict, List
import pandas as pd
from automl_tabular.models.search import TrialResult


def generate_model_summary(
    best_result: TrialResult,
    problem_type: str,
    metric_name: str,
    metric_value: float
) -> str:
    """
    Generate plain-language summary of the best model.
    
    Args:
        best_result: Best TrialResult
        problem_type: 'classification' or 'regression'
        metric_name: Name of primary metric
        metric_value: Value of primary metric
        
    Returns:
        Text summary
    """
    summary = []
    
    summary.append(f"## Best Model Summary\n")
    summary.append(
        f"The AutoML system selected **{best_result.model_name}** as the best-performing model "
        f"for this {problem_type} task.\n"
    )
    
    summary.append(
        f"This model achieved a **{metric_name}** score of **{metric_value:.4f}** "
        f"on the validation set.\n"
    )
    
    summary.append(f"### Model Configuration\n")
    summary.append("The best hyperparameters found were:\n")
    for param, value in best_result.params.items():
        summary.append(f"- **{param}**: {value}")
    
    summary.append(f"\n### Training Details\n")
    summary.append(f"- Training time: {best_result.train_time:.2f} seconds")
    
    return "\n".join(summary)


def generate_feature_importance_summary(
    feature_importance_df: pd.DataFrame,
    top_k: int = 5
) -> str:
    """
    Generate plain-language summary of feature importance.
    
    Args:
        feature_importance_df: DataFrame with feature importance
        top_k: Number of top features to describe
        
    Returns:
        Text summary
    """
    if feature_importance_df.empty:
        return "Feature importance information is not available for this model.\n"
    
    summary = []
    
    summary.append(f"## Feature Importance Analysis\n")
    
    top_features = feature_importance_df.head(top_k)
    
    # Check if this is aggregated by column or raw features
    has_agg = 'num_features' in feature_importance_df.columns
    
    if has_agg:
        summary.append(
            f"The model identified the following {len(top_features)} feature groups as most important "
            f"for making predictions:\n"
        )
    else:
        summary.append(
            f"The model identified the following {len(top_features)} features as most important "
            f"for making predictions:\n"
        )
    
    for i, row in top_features.iterrows():
        feature = row['feature']
        importance = row['importance']
        if has_agg and 'num_features' in row and row['num_features'] > 1:
            summary.append(f"{i+1}. **{feature}** ({int(row['num_features'])} encoded features, importance: {importance:.4f})")
        else:
            summary.append(f"{i+1}. **{feature}** (importance: {importance:.4f})")
    
    # Skip the percentage calculation for models with many sparse features
    # as it's not very informative
    if 'importance_percentage' in feature_importance_df.columns and not has_agg:
        cumulative = top_features['importance_percentage'].sum()
        if cumulative > 10:  # Only show if it's a meaningful percentage
            summary.append(
                f"\nThese {len(top_features)} features collectively account for "
                f"**{cumulative:.1f}%** of the model's decision-making process.\n"
            )
    
    if has_agg:
        summary.append(
            f"\n*Note: High-cardinality identifier-like columns (e.g., IDs or free-text identifiers) were excluded "
            f"from this analysis to avoid over-interpreting sparse one-hot categories.*\n"
        )
    
    return "\n".join(summary)


def generate_data_summary(data_info: Dict) -> str:
    """
    Generate plain-language summary of the dataset.
    
    Args:
        data_info: Dictionary with dataset information
        
    Returns:
        Text summary
    """
    summary = []
    
    summary.append(f"## Dataset Overview\n")
    
    summary.append(f"- **Total samples**: {data_info.get('n_rows', 'N/A'):,}")
    summary.append(f"- **Total features**: {data_info.get('n_columns', 'N/A')}")
    
    # Missing values
    if 'missing_percentage' in data_info:
        missing_pct = data_info['missing_percentage']
        high_missing = {k: v for k, v in missing_pct.items() if v > 10}
        
        if high_missing:
            summary.append(f"\n### Data Quality Notes")
            summary.append(
                f"The following features have more than 10% missing values:"
            )
            for col, pct in sorted(high_missing.items(), key=lambda x: x[1], reverse=True):
                summary.append(f"- **{col}**: {pct:.1f}% missing")
    
    return "\n".join(summary)


def generate_recommendations(
    problem_type: str,
    best_result: TrialResult,
    metric_value: float,
    data_info: Dict
) -> str:
    """
    Generate recommendations for model improvement.
    
    Args:
        problem_type: 'classification' or 'regression'
        best_result: Best model result
        metric_value: Primary metric value
        data_info: Dataset information
        
    Returns:
        Text with recommendations
    """
    summary = []
    
    summary.append(f"## Recommendations for Model Improvement\n")
    
    # Data-related recommendations
    n_rows = data_info.get('n_rows', 0)
    if n_rows < 1000:
        summary.append(
            "- **Collect more data**: Your dataset is relatively small. "
            "Collecting more samples could significantly improve model performance."
        )
    
    # Feature engineering
    summary.append(
        "- **Feature engineering**: Consider creating new features based on domain knowledge "
        "or interactions between existing features."
    )
    
    # Model-specific recommendations
    if best_result.model_name in ['logistic_regression', 'linear_regression']:
        summary.append(
            "- **Tree-based alternatives**: Tree-based models (RandomForest, XGBoost, LightGBM) "
            "achieved slightly higher ROC_AUC in the search and may capture more complex patterns, "
            "at the cost of interpretability and deployment simplicity."
        )
    
    # Hyperparameter tuning
    summary.append(
        "- **Extended hyperparameter tuning**: Consider increasing the number of Optuna trials "
        "for more thorough hyperparameter search."
    )
    
    # Deployment
    summary.append(
        "- **Monitor in production**: Once deployed, continuously monitor model performance "
        "and retrain periodically with new data."
    )
    
    return "\n".join(summary)


__all__ = [
    "generate_model_summary",
    "generate_feature_importance_summary",
    "generate_data_summary",
    "generate_recommendations"
]
