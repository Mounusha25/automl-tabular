"""Configuration management for AutoML."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_default_config() -> Dict[str, Any]:
    """Load the default configuration from YAML file."""
    config_path = Path(__file__).parent / "default_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override config into default config."""
    merged = default.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


__all__ = ["load_default_config", "merge_configs"]
