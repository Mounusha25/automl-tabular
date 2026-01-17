"""Explainability module for feature importance and visualizations."""

from automl_tabular.explainability.feature_importance import (
    get_feature_importance,
    get_top_features,
    normalize_importance
)
from automl_tabular.explainability.plots import (
    plot_feature_importance,
    plot_target_distribution,
    plot_feature_vs_target,
    plot_correlation_heatmap
)
from automl_tabular.explainability.text_summaries import (
    generate_model_summary,
    generate_feature_importance_summary,
    generate_data_summary,
    generate_recommendations
)

__all__ = [
    "get_feature_importance",
    "get_top_features",
    "normalize_importance",
    "plot_feature_importance",
    "plot_target_distribution",
    "plot_feature_vs_target",
    "plot_correlation_heatmap",
    "generate_model_summary",
    "generate_feature_importance_summary",
    "generate_data_summary",
    "generate_recommendations"
]
