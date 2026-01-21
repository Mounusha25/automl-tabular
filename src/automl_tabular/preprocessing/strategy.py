"""
Preprocessing strategy selection based on data profiling.

Intelligent selection of imputation, outlier handling, and scaling strategies.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from automl_tabular.preprocessing.profile import FeatureProfile, ProfileResult


@dataclass
class FeaturePreprocessingPlan:
    """Preprocessing plan for a single feature."""
    
    name: str
    dtype: str
    
    # Imputation
    imputation_strategy: str  # "median", "mean", "most_frequent", "constant", "drop", "none"
    imputation_value: Optional[float] = None
    add_missing_indicator: bool = False
    
    # Outlier handling
    outlier_handling: str = "none"  # "none", "clip", "winsorize"
    clip_lower: Optional[float] = None
    clip_upper: Optional[float] = None
    
    # Scaling
    scaling: str = "standard"  # "none", "standard", "robust", "minmax"
    
    # Transform
    transform: str = "none"  # "none", "log1p", "boxcox"
    
    # Human-readable explanation
    explanation: str = ""


@dataclass
class PreprocessingPlan:
    """Complete preprocessing plan for dataset."""
    
    features: Dict[str, FeaturePreprocessingPlan]
    numeric_features: List[str]
    categorical_features: List[str]
    dropped_features: List[str]
    


def build_preprocessing_plan(
    profile: ProfileResult,
    config: Optional[Dict] = None,
    enable_smart_strategies: bool = True
) -> PreprocessingPlan:
    """
    Build intelligent preprocessing plan based on data profile.
    
    Args:
        profile: Dataset profile from profiling module
        config: Optional configuration overrides
        enable_smart_strategies: If False, falls back to simple median/mode
        
    Returns:
        PreprocessingPlan object
    """
    config = config or {}
    
    features = {}
    numeric_features = []
    categorical_features = []
    dropped_features = []
    
    # High missingness threshold from config
    high_missing_threshold = config.get('high_missing_threshold', 0.95)
    
    for name, feat_profile in profile.features.items():
        
        # Handle constant features
        if feat_profile.is_constant:
            plan = FeaturePreprocessingPlan(
                name=name,
                dtype=feat_profile.dtype,
                imputation_strategy="drop",
                explanation=f"Dropped because it has only {feat_profile.n_unique} unique value(s) (no predictive signal)."
            )
            dropped_features.append(name)
            features[name] = plan
            continue
        
        # Handle identifier-like features
        if feat_profile.is_identifier_like:
            plan = FeaturePreprocessingPlan(
                name=name,
                dtype=feat_profile.dtype,
                imputation_strategy="drop",
                explanation=f"Dropped because it looks like an identifier ({feat_profile.unique_ratio*100:.1f}% unique values, name contains 'id')."
            )
            dropped_features.append(name)
            features[name] = plan
            continue
        
        # NUMERIC FEATURES
        if feat_profile.dtype == "numeric":
            numeric_features.append(name)
            
            if enable_smart_strategies:
                plan = _build_numeric_plan(feat_profile, config, high_missing_threshold)
            else:
                # Fallback to simple median
                plan = _build_simple_numeric_plan(feat_profile)
            
            features[name] = plan
        
        # CATEGORICAL FEATURES
        elif feat_profile.dtype == "categorical":
            categorical_features.append(name)
            
            if enable_smart_strategies:
                plan = _build_categorical_plan(feat_profile, config, high_missing_threshold)
            else:
                # Fallback to simple most_frequent
                plan = _build_simple_categorical_plan(feat_profile)
            
            features[name] = plan
    
    return PreprocessingPlan(
        features=features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_features=dropped_features
    )


def _build_numeric_plan(
    profile: FeatureProfile,
    config: Dict,
    high_missing_threshold: float
) -> FeaturePreprocessingPlan:
    """Build smart preprocessing plan for numeric feature."""
    
    mr = profile.missing_ratio
    skew = profile.skewness or 0.0
    oratio = profile.outlier_ratio or 0.0
    
    explanation_parts = []
    
    # === IMPUTATION ===
    if mr == 0:
        imputation_strategy = "none"
        add_missing_indicator = False
        explanation_parts.append("No missing values")
    
    elif mr > high_missing_threshold:
        # Very high missingness - drop or use median with indicator
        drop_high_missing = config.get('drop_high_missing_features', False)
        if drop_high_missing:
            return FeaturePreprocessingPlan(
                name=profile.name,
                dtype="numeric",
                imputation_strategy="drop",
                explanation=f"Dropped due to very high missingness ({mr*100:.1f}% missing)."
            )
        else:
            imputation_strategy = "median"
            add_missing_indicator = True
            explanation_parts.append(f"Very high missingness ({mr*100:.1f}%) - using median + missing indicator")
    
    elif mr > 0.4:
        # High missingness
        imputation_strategy = "median"
        add_missing_indicator = True
        explanation_parts.append(f"High missingness ({mr*100:.1f}%) - using median + missing indicator")
    
    elif mr > 0.05:
        # Moderate missingness
        imputation_strategy = "median"
        add_missing_indicator = True
        explanation_parts.append(f"Moderate missingness ({mr*100:.1f}%) - using median + missing indicator")
    
    else:
        # Low missingness
        if abs(skew) < 1.0:
            imputation_strategy = "mean"
            explanation_parts.append(f"Low missingness ({mr*100:.1f}%), near-normal distribution → mean imputation")
        else:
            imputation_strategy = "median"
            explanation_parts.append(f"Low missingness ({mr*100:.1f}%), skewed (skew={skew:.2f}) → median imputation")
        add_missing_indicator = False
    
    # === OUTLIER HANDLING ===
    if oratio <= 0.05:
        outlier_handling = "none"
        clip_lower = None
        clip_upper = None
    elif oratio <= 0.2:
        outlier_handling = "clip"
        # Use percentile-based clipping
        clip_lower = profile.q1 - 1.5 * (profile.q3 - profile.q1) if profile.q1 and profile.q3 else None
        clip_upper = profile.q3 + 1.5 * (profile.q3 - profile.q1) if profile.q1 and profile.q3 else None
        explanation_parts.append(f"{oratio*100:.1f}% outliers detected (IQR method) - clipping to [Q1-1.5×IQR, Q3+1.5×IQR]")
    else:
        # Heavy outliers
        outlier_handling = "clip"
        clip_lower = profile.q1 - 1.5 * (profile.q3 - profile.q1) if profile.q1 and profile.q3 else None
        clip_upper = profile.q3 + 1.5 * (profile.q3 - profile.q1) if profile.q1 and profile.q3 else None
        explanation_parts.append(f"Heavy outliers ({oratio*100:.1f}% by IQR) - clipping applied")
    
    # === SCALING ===
    if abs(skew) >= 1.0 or oratio > 0.05:
        scaling = "robust"
        explanation_parts.append("Using robust scaling (less sensitive to outliers)")
    else:
        scaling = "standard"
        explanation_parts.append("Using standard scaling")
    
    # === TRANSFORM ===
    transform = "none"
    # Could add log transform for heavy right skew
    # if profile.min_val and profile.min_val > 0 and skew > 1.5:
    #     transform = "log1p"
    #     explanation_parts.append(f"Strongly right-skewed (skew={skew:.2f}) - applying log1p transform")
    
    explanation = f"{profile.name}: " + ". ".join(explanation_parts) + "."
    
    return FeaturePreprocessingPlan(
        name=profile.name,
        dtype="numeric",
        imputation_strategy=imputation_strategy,
        add_missing_indicator=add_missing_indicator,
        outlier_handling=outlier_handling,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        scaling=scaling,
        transform=transform,
        explanation=explanation
    )


def _build_categorical_plan(
    profile: FeatureProfile,
    config: Dict,
    high_missing_threshold: float
) -> FeaturePreprocessingPlan:
    """Build smart preprocessing plan for categorical feature."""
    
    mr = profile.missing_ratio
    explanation_parts = []
    
    # === IDENTIFIER-LIKE HIGH CARDINALITY ===
    # Check if this is an identifier-like feature that should be dropped
    if profile.is_identifier_like or (profile.unique_ratio > 0.95 and profile.is_high_cardinality):
        return FeaturePreprocessingPlan(
            name=profile.name,
            dtype="categorical",
            imputation_strategy="drop",
            explanation=f"High-cardinality identifier-like feature ({profile.n_unique} unique values, {profile.unique_ratio*100:.1f}% unique). Dropped from modeling to avoid sparse features and overfitting."
        )
    
    # === HIGH CARDINALITY (non-identifier) ===
    if profile.is_high_cardinality:
        explanation_parts.append(f"High cardinality ({profile.n_unique} unique values)")
        # Could implement rare category grouping here
    
    # === IMPUTATION ===
    if mr == 0:
        imputation_strategy = "none"
        add_missing_indicator = False
        explanation_parts.append("No missing values")
    
    elif mr > high_missing_threshold:
        return FeaturePreprocessingPlan(
            name=profile.name,
            dtype="categorical",
            imputation_strategy="drop",
            explanation=f"Dropped due to very high missingness ({mr*100:.1f}%)."
        )
    
    elif mr > 0.1:
        # Treat missing as separate category
        imputation_strategy = "constant"
        imputation_value = "__MISSING__"
        add_missing_indicator = False
        explanation_parts.append(f"Missingness ({mr*100:.1f}%) encoded as separate '__MISSING__' category")
    
    else:
        # Low missingness - use mode
        imputation_strategy = "most_frequent"
        add_missing_indicator = False
        explanation_parts.append(f"Low missingness ({mr*100:.1f}%) - using most frequent value")
    
    explanation = f"{profile.name}: " + ". ".join(explanation_parts) + "."
    
    return FeaturePreprocessingPlan(
        name=profile.name,
        dtype="categorical",
        imputation_strategy=imputation_strategy,
        imputation_value=imputation_value if imputation_strategy == "constant" else None,
        add_missing_indicator=add_missing_indicator,
        scaling="none",
        explanation=explanation
    )


def _build_simple_numeric_plan(profile: FeatureProfile) -> FeaturePreprocessingPlan:
    """Simple fallback plan - just median imputation."""
    
    if profile.missing_ratio == 0:
        imputation_strategy = "none"
    else:
        imputation_strategy = "median"
    
    return FeaturePreprocessingPlan(
        name=profile.name,
        dtype="numeric",
        imputation_strategy=imputation_strategy,
        add_missing_indicator=False,
        outlier_handling="none",
        scaling="standard",
        explanation=f"{profile.name}: Standard preprocessing (median imputation, standard scaling)."
    )


def _build_simple_categorical_plan(profile: FeatureProfile) -> FeaturePreprocessingPlan:
    """Simple fallback plan - just mode imputation."""
    
    if profile.missing_ratio == 0:
        imputation_strategy = "none"
    else:
        imputation_strategy = "most_frequent"
    
    return FeaturePreprocessingPlan(
        name=profile.name,
        dtype="categorical",
        imputation_strategy=imputation_strategy,
        add_missing_indicator=False,
        scaling="none",
        explanation=f"{profile.name}: Standard preprocessing (mode imputation, one-hot encoding)."
    )


__all__ = [
    "FeaturePreprocessingPlan",
    "PreprocessingPlan",
    "build_preprocessing_plan"
]
