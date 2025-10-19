"""
Rolling features calculator for feature building.

Handles calculation of rolling averages, forms, and trend features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from ...common.config import get_config, get_logger

logger = get_logger(__name__)


class RollingCalculator:
    """Calculates rolling features for player and team data."""
    
    def __init__(self):
        """Initialize rolling calculator."""
        self.config = get_config()
    
    def calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling features for all players.
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with rolling features added
        """
        if df.empty:
            return df
        
        logger.info("Calculating rolling features")
        
        try:
            result_df = df.copy()
            
            # Sort by player and gameweek for rolling calculations
            player_col = 'web_name' if 'web_name' in result_df.columns else 'name' if 'name' in result_df.columns else 'player_name'
            if player_col in result_df.columns and 'gameweek' in result_df.columns:
                result_df = result_df.sort_values([player_col, 'gameweek'])
            
            # Calculate rolling features by position
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                pos_mask = result_df['position'] == position
                if pos_mask.any():
                    pos_data = result_df[pos_mask].copy()
                    pos_data = self._calculate_position_rolling_features(pos_data)
                    result_df.loc[pos_mask, pos_data.columns] = pos_data
            
            # Calculate team-level rolling features
            result_df = self._calculate_team_rolling_features(result_df)
            
            logger.info("Rolling features calculation completed")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating rolling features: {e}")
            return df
    
    def _calculate_position_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling features for a specific position."""
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Define rolling windows
        windows = [3, 5, 10]
        
        # Features to calculate rolling averages for
        rolling_features = [
            'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'saves', 'bonus', 'creativity', 'influence', 'threat', 'ict_index',
            'expected_goals', 'expected_assists', 'expected_saves'
        ]
        
        # Group by player for rolling calculations
        player_col = 'web_name' if 'web_name' in result_df.columns else 'name' if 'name' in result_df.columns else 'player_name'
        for player in result_df[player_col].unique():
            player_mask = result_df[player_col] == player
            player_data = result_df[player_mask].copy()
            
            for window in windows:
                for feature in rolling_features:
                    if feature in player_data.columns:
                        # Calculate rolling mean
                        rolling_mean = player_data[feature].rolling(window=window, min_periods=1).mean()
                        result_df.loc[player_mask, f'{feature}_rolling_{window}'] = rolling_mean
                        
                        # Calculate rolling standard deviation
                        rolling_std = player_data[feature].rolling(window=window, min_periods=1).std()
                        result_df.loc[player_mask, f'{feature}_std_{window}'] = rolling_std
                        
                        # Calculate rolling trend (slope of linear regression)
                        trend = self._calculate_trend(player_data[feature], window)
                        result_df.loc[player_mask, f'{feature}_trend_{window}'] = trend
            
            # Calculate form features (recent performance vs average)
            self._calculate_form_features(result_df, player_mask, player_data)
        
        return result_df
    
    def _calculate_team_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team-level rolling features."""
        if df.empty or 'team_name' not in df.columns:
            return df
        
        result_df = df.copy()
        
        # Team-level features
        team_features = ['total_points', 'goals_scored', 'assists', 'clean_sheets']
        windows = [3, 5]
        
        for team in result_df['team_name'].unique():
            team_mask = result_df['team_name'] == team
            team_data = result_df[team_mask].copy()
            
            # Sort by gameweek
            if 'gameweek' in team_data.columns:
                team_data = team_data.sort_values('gameweek')
                
                for window in windows:
                    for feature in team_features:
                        if feature in team_data.columns:
                            # Team rolling average
                            team_rolling = team_data[feature].rolling(window=window, min_periods=1).mean()
                            result_df.loc[team_mask, f'team_{feature}_rolling_{window}'] = team_rolling
        
        return result_df
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend (slope) for a rolling window."""
        if len(series) < 2:
            return pd.Series([0.0] * len(series), index=series.index)
        
        trends = []
        for i in range(len(series)):
            if i < window - 1:
                # Not enough data for full window
                trends.append(0.0)
            else:
                # Get window data
                window_data = series.iloc[max(0, i-window+1):i+1]
                if len(window_data) > 1:
                    # Calculate slope using linear regression
                    x = np.arange(len(window_data))
                    y = window_data.values
                    slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
                    trends.append(slope)
                else:
                    trends.append(0.0)
        
        return pd.Series(trends, index=series.index)
    
    def _calculate_form_features(self, df: pd.DataFrame, player_mask: pd.Series, player_data: pd.DataFrame):
        """Calculate form-related features."""
        if player_data.empty:
            return
        
        # Recent form (last 3 games vs season average)
        if 'total_points' in player_data.columns and len(player_data) >= 3:
            recent_avg = player_data['total_points'].tail(3).mean()
            season_avg = player_data['total_points'].mean()
            form_ratio = recent_avg / season_avg if season_avg > 0 else 1.0
            df.loc[player_mask, 'form_ratio'] = form_ratio
        
        # Consistency (coefficient of variation)
        if 'total_points' in player_data.columns and len(player_data) >= 5:
            points_std = player_data['total_points'].std()
            points_mean = player_data['total_points'].mean()
            consistency = 1 - (points_std / points_mean) if points_mean > 0 else 0.0
            df.loc[player_mask, 'consistency'] = max(0, consistency)
        
        # Minutes consistency
        if 'minutes' in player_data.columns and len(player_data) >= 3:
            minutes_played = (player_data['minutes'] > 0).sum()
            total_games = len(player_data)
            availability = minutes_played / total_games if total_games > 0 else 0.0
            df.loc[player_mask, 'availability'] = availability
