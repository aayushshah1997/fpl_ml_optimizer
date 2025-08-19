"""
Feature engineering pipeline for FPL AI system.

This module provides comprehensive feature engineering capabilities:
- Multi-league data integration and normalization
- Rolling window features (3/5/8 game averages)
- Team and opponent form metrics
- Set piece roles and market signals
- Training and prediction frame building
"""

from .builder import FeatureBuilder
from .touches import calculate_touches_features
from .team_form import calculate_team_form
from .h2h import calculate_h2h_features

__all__ = [
    "FeatureBuilder",
    "calculate_touches_features", 
    "calculate_team_form",
    "calculate_h2h_features",
]
