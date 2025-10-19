"""
Touches and player involvement features.

Calculates features related to player involvement in games,
including touches, passing accuracy, and general play metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


def calculate_touches_features(
    player_data: pd.DataFrame,
    rolling_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Calculate touches and involvement features.
    
    Args:
        player_data: DataFrame with player match data
        rolling_windows: Rolling window sizes to calculate
        
    Returns:
        DataFrame with touches features added
    """
    if player_data.empty:
        return player_data
    
    config = get_config()
    if rolling_windows is None:
        rolling_windows = config.get("training.rolling_windows", [3, 5, 8])
    
    df = player_data.copy()
    
    # Calculate base involvement metrics
    df = _calculate_base_involvement(df)
    
    # Calculate rolling involvement features
    df = _calculate_rolling_involvement(df, rolling_windows)
    
    # Calculate per-90 involvement metrics
    df = _calculate_per90_involvement(df)
    
    logger.debug(f"Calculated touches features for {len(df)} records")
    return df


def _calculate_base_involvement(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate base involvement metrics."""
    # Estimate touches from available FPL data
    # (FPL doesn't provide direct touches data, so we approximate)
    
    # Base touches estimation (rough approximation)
    df['est_touches'] = (
        df.get('minutes', 0) * 0.8 +  # Base touches per minute
        df.get('key_passes', 0) * 3 +  # Key passes suggest more involvement
        df.get('total_shots', 0) * 2 +  # Shots suggest attacking involvement
        df.get('passes_completed', df.get('passes', 0)) * 0.1  # If pass data available
    )
    
    # Estimate passing metrics if not available
    if 'passes_completed' not in df.columns:
        # Rough estimation based on position and game involvement
        position_pass_rates = {
            'GK': 0.8,   # Goalkeepers have high pass completion
            'DEF': 0.85, # Defenders typically high
            'MID': 0.82, # Midfielders vary
            'FWD': 0.75  # Forwards typically lower
        }
        
        base_passes = df.get('minutes', 0) * 0.5  # Rough passes per minute
        df['est_passes'] = base_passes
        
        # Estimate completion rate by position
        # Ensure we get a Series before calling fillna
        position_col = df.get('position', 'MID')
        completion_mapped = position_col.map(position_pass_rates)
        if isinstance(completion_mapped, pd.Series):
            df['est_pass_completion'] = completion_mapped.fillna(0.8)
        else:
            # If it's a scalar, create a Series with the same length
            df['est_pass_completion'] = pd.Series([completion_mapped] * len(df), index=df.index)
        df['passes_completed'] = df['est_passes'] * df['est_pass_completion']
    
    # Calculate involvement intensity
    df['involvement_intensity'] = df['est_touches'] / (df.get('minutes', 1) / 90)
    
    # Attacking involvement
    df['attacking_involvement'] = (
        df.get('shots', 0) + 
        df.get('key_passes', 0) + 
        df.get('big_chances_created', 0) * 2
    )
    
    # Defensive involvement (for defenders)
    df['defensive_involvement'] = (
        df.get('tackles', 0) + 
        df.get('interceptions', 0) + 
        df.get('clearances', 0) * 0.5
    )
    
    return df


def _calculate_rolling_involvement(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Calculate rolling involvement metrics."""
    if 'element_id' not in df.columns:
        return df
    
    # Ensure data is sorted by player and time
    df = df.sort_values(['element_id', 'kickoff_time'])
    
    involvement_cols = [
        'est_touches', 'involvement_intensity', 
        'attacking_involvement', 'defensive_involvement'
    ]
    
    for window in windows:
        for col in involvement_cols:
            if col in df.columns:
                rolling_col = f"{col}_r{window}"
                df[rolling_col] = df.groupby('element_id')[col].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(level=0, drop=True)
    
    return df


def _calculate_per90_involvement(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-90 minute involvement metrics."""
    per90_cols = [
        'est_touches', 'attacking_involvement', 'defensive_involvement'
    ]
    
    for col in per90_cols:
        if col in df.columns:
            per90_col = f"{col}_per90"
            df[per90_col] = df[col] / (df.get('minutes', 90) / 90)
    
    return df


def calculate_creativity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate creativity and chance creation metrics.
    
    Args:
        df: DataFrame with player data
        
    Returns:
        DataFrame with creativity metrics added
    """
    if df.empty:
        return df
    
    # Creativity index (FPL provides this, but we can enhance it)
    base_creativity = df.get('creativity', 0)
    
    # Enhanced creativity score
    df['enhanced_creativity'] = (
        base_creativity * 0.7 +
        df.get('key_passes', 0) * 15 +
        df.get('big_chances_created', 0) * 25 +
        df.get('assists', 0) * 30
    )
    
    # Creativity per 90
    df['creativity_per90'] = df['enhanced_creativity'] / (df.get('minutes', 90) / 90)
    
    return df


def calculate_influence_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate influence and leadership metrics.
    
    Args:
        df: DataFrame with player data
        
    Returns:
        DataFrame with influence metrics added
    """
    if df.empty:
        return df
    
    # Base influence from FPL
    base_influence = df.get('influence', 0)
    
    # Enhanced influence considering various factors
    df['enhanced_influence'] = (
        base_influence * 0.6 +
        df.get('goals_scored', 0) * 20 +
        df.get('assists', 0) * 15 +
        df.get('bonus', 0) * 5 +
        (df.get('minutes', 0) >= 60) * 10  # Full game bonus
    )
    
    # Team influence (how much of team's output player contributes)
    team_stats = df.groupby(['team_id', 'gameweek']).agg({
        'goals_scored': 'sum',
        'assists': 'sum',
        'points': 'sum'
    }).add_suffix('_team')
    
    df = df.merge(team_stats, left_on=['team_id', 'gameweek'], right_index=True, how='left')
    
    # Calculate team contribution rates
    df['goals_share'] = df['goals_scored'] / (df['goals_scored_team'] + 1)
    df['assists_share'] = df['assists'] / (df['assists_team'] + 1)
    df['points_share'] = df['points'] / (df['points_team'] + 1)
    
    return df


def calculate_consistency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consistency and reliability metrics.
    
    Args:
        df: DataFrame with player data
        
    Returns:
        DataFrame with consistency metrics added
    """
    if df.empty or 'element_id' not in df.columns:
        return df
    
    # Calculate rolling standard deviation of points
    for window in [3, 5, 8]:
        points_std_col = f"points_std_r{window}"
        df[points_std_col] = df.groupby('element_id')['points'].rolling(
            window=window, min_periods=2
        ).std().reset_index(level=0, drop=True)
        
        # Consistency score (lower std = more consistent)
        consistency_col = f"consistency_r{window}"
        # Ensure we get a Series before calling fillna
        std_series = df[points_std_col]
        if isinstance(std_series, pd.Series):
            df[consistency_col] = 1 / (1 + std_series.fillna(0))
        else:
            # If it's a scalar, create a Series with the same length
            df[consistency_col] = 1 / (1 + pd.Series([std_series] * len(df), index=df.index))
    
    # Minutes consistency
    for window in [3, 5, 8]:
        minutes_std_col = f"minutes_std_r{window}"
        df[minutes_std_col] = df.groupby('element_id')['minutes'].rolling(
            window=window, min_periods=2
        ).std().reset_index(level=0, drop=True)
        
        # Minutes reliability
        minutes_reliability_col = f"minutes_reliability_r{window}"
        # Ensure we get a Series before calling fillna
        minutes_std_series = df[minutes_std_col]
        if isinstance(minutes_std_series, pd.Series):
            df[minutes_reliability_col] = 1 / (1 + minutes_std_series.fillna(0))
        else:
            # If it's a scalar, create a Series with the same length
            df[minutes_reliability_col] = 1 / (1 + pd.Series([minutes_std_series] * len(df), index=df.index))
    
    return df


def calculate_form_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate form and momentum indicators.
    
    Args:
        df: DataFrame with player data
        
    Returns:
        DataFrame with momentum metrics added
    """
    if df.empty or 'element_id' not in df.columns:
        return df
    
    # Sort by player and time
    df = df.sort_values(['element_id', 'kickoff_time'])
    
    # Calculate trend in points (is player improving or declining?)
    for window in [3, 5]:
        trend_col = f"points_trend_r{window}"
        
        # Use linear regression slope as trend indicator
        def calculate_trend(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            if np.std(y) == 0:
                return 0
            return np.corrcoef(x, y)[0, 1] * np.std(y)
        
        df[trend_col] = df.groupby('element_id')['points'].rolling(
            window=window, min_periods=2
        ).apply(calculate_trend, raw=False).reset_index(level=0, drop=True)
    
    # Recent vs historical performance
    df['recent_vs_season'] = (
        df.get('points_r3', 0) - df.get('points_r8', 0)
    )
    
    return df
