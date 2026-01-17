"""
AutoML Tabular - Explainable AutoML for Tabular Data

An automated machine learning system that:
- Automatically detects problem type (classification/regression)
- Runs model selection and hyperparameter tuning
- Generates human-readable explanations and reports
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from automl_tabular.orchestrator import run_automl_job

__all__ = ["run_automl_job"]
