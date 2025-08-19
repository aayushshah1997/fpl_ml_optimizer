"""
FPL AI - Production-grade Fantasy Premier League analytics system.

A comprehensive machine learning platform for FPL team optimization with:
- Multi-league data integration via FBRef API
- Staged training (Warm Start vs Full ML)
- Advanced risk optimization and Monte Carlo simulation
- 10-week transfer planning with chip timing
- Real-time Streamlit dashboard

Version: 0.1.0
Author: FPL AI Team
"""

__version__ = "0.1.0"
__author__ = "FPL AI Team"

# Core modules
from .common.config import get_config, get_logger
from .common.cache import CacheManager
from .common.timeutil import get_current_gw, get_season_dates

__all__ = [
    "get_config",
    "get_logger", 
    "CacheManager",
    "get_current_gw",
    "get_season_dates",
]
