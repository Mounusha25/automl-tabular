"""Plotting utilities for visualization."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    top_k: int = 15,
    save_path: Optional[str] = None,
    dpi: int = 100
) -> str:
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        feature_importance_df: DataFrame with feature and importance columns
        top_k: Number of top features to plot
        save_path: Path to save the plot
        dpi: Resolution for saved plot
        
    Returns:
        Path to saved plot
    """
    if feature_importance_df.empty:
        return None
    
    # Get top K features
    plot_df = feature_importance_df.head(top_k).copy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.35)))
    
    # Horizontal bar chart with gradient colors
    y_pos = np.arange(len(plot_df))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))
    ax.barh(y_pos, plot_df['importance'], color=colors, edgecolor='white', linewidth=0.5)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'])
    ax.invert_yaxis()  # Top feature at the top
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {len(plot_df)} Most Important Features', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(plot_df['importance']):
        ax.text(v, i, f' {v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = 'feature_importance.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_target_distribution(
    y: np.ndarray,
    problem_type: str,
    save_path: Optional[str] = None,
    dpi: int = 100
) -> str:
    """
    Plot target variable distribution.
    
    Args:
        y: Target values
        problem_type: 'classification' or 'regression'
        save_path: Path to save the plot
        dpi: Resolution
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if problem_type == 'classification':
        # Bar chart for classification
        unique, counts = np.unique(y, return_counts=True)
        colors = ['#3498db', '#e74c3c'] if len(unique) == 2 else plt.cm.Set3(np.arange(len(unique)))
        ax.bar(range(len(unique)), counts, color=colors, edgecolor='white', linewidth=1.5, alpha=0.85)
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(unique)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Target Class Distribution', fontsize=14, fontweight='bold')
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(i, count, str(count), ha='center', va='bottom')
    else:
        # Histogram for regression
        ax.hist(y, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Target Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Target Value Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'target_distribution.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_feature_vs_target(
    X: pd.DataFrame,
    y: pd.Series,
    feature_name: str,
    problem_type: str,
    save_path: Optional[str] = None,
    dpi: int = 100
) -> str:
    """
    Plot relationship between a feature and target.
    
    Args:
        X: Feature DataFrame
        y: Target series
        feature_name: Name of feature to plot
        problem_type: 'classification' or 'regression'
        save_path: Path to save plot
        dpi: Resolution
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature_data = X[feature_name]
    
    if problem_type == 'classification':
        # Box plot for classification
        df_plot = pd.DataFrame({
            'feature': feature_data,
            'target': y
        })
        sns.boxplot(data=df_plot, x='target', y='feature', ax=ax, palette='Set2')
        ax.set_xlabel('Target Class', fontsize=12)
        ax.set_ylabel(feature_name, fontsize=12)
    else:
        # Scatter plot for regression
        ax.scatter(feature_data, y, alpha=0.5, color='steelblue')
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel('Target', fontsize=12)
        
        # Add trend line
        try:
            z = np.polyfit(feature_data.dropna(), y[feature_data.notna()], 1)
            p = np.poly1d(z)
            ax.plot(feature_data, p(feature_data), "r--", alpha=0.8, linewidth=2)
        except:
            pass
    
    ax.set_title(f'{feature_name} vs Target', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'{feature_name}_vs_target.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_correlation_heatmap(
    X: pd.DataFrame,
    top_k: int = 20,
    save_path: Optional[str] = None,
    dpi: int = 100
) -> str:
    """
    Plot correlation heatmap for numeric features.
    
    Args:
        X: Feature DataFrame
        top_k: Number of features to include
        save_path: Path to save plot
        dpi: Resolution
        
    Returns:
        Path to saved plot
    """
    # Select numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns[:top_k]
    
    if len(numeric_cols) == 0:
        return None
    
    # Compute correlation
    corr = X[numeric_cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'correlation_heatmap.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return save_path


__all__ = [
    "plot_feature_importance",
    "plot_target_distribution",
    "plot_feature_vs_target",
    "plot_correlation_heatmap",
    "plot_model_comparison",
    "plot_missing_data",
    "plot_confusion_matrix"
]


def plot_model_comparison(
    leaderboard_df: pd.DataFrame,
    metric_name: str = 'score',
    save_path: Optional[str] = None,
    dpi: int = 100
) -> str:
    """
    Plot model performance comparison.
    
    Args:
        leaderboard_df: Leaderboard DataFrame
        metric_name: Name of metric to plot
        save_path: Path to save plot
        dpi: Resolution
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Get top 10 unique models
    top_models = leaderboard_df.groupby('model')[metric_name].max().nlargest(10).reset_index()
    
    # Bar plot with gradient colors
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(top_models)))
    ax.bar(range(len(top_models)), top_models[metric_name], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(top_models)))
    ax.set_xticklabels(top_models['model'], rotation=45, ha='right')
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(top_models[metric_name]):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'model_comparison.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_missing_data(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    dpi: int = 100
) -> str:
    """
    Plot missing data visualization.
    
    Args:
        df: Input DataFrame
        save_path: Path to save plot
        dpi: Resolution
        
    Returns:
        Path to saved plot
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(7, max(4, len(missing) * 0.25)))
    
    missing_pct = (missing / len(df)) * 100
    # Color based on severity: red > 50%, orange 20-50%, blue < 20%
    colors = ['#e74c3c' if x > 50 else '#f39c12' if x > 20 else '#3498db' for x in missing_pct]
    ax.barh(range(len(missing)), missing_pct, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(missing)))
    ax.set_yticklabels(missing.index)
    ax.invert_yaxis()
    ax.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax.set_title('Missing Data by Feature', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(missing_pct):
        ax.text(v, i, f' {v:.1f}% ({missing.iloc[i]} missing)', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'missing_data.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 100
) -> str:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
        dpi: Resolution
        
    Returns:
        Path to saved plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='RdYlGn_r',
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'confusion_matrix.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return save_path
