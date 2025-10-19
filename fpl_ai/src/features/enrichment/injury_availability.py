"""
Injury and availability enrichment module for feature building.

Handles calculation of player availability, injury status, and related features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from ...common.config import get_config, get_logger
from ...providers.injuries import InjuryProvider

logger = get_logger(__name__)


class InjuryAvailabilityEnricher:
    """Enriches data with injury and availability features."""
    
    def __init__(self):
        """Initialize injury availability enricher."""
        self.config = get_config()
        self.injury_provider = InjuryProvider()
    
    def add_availability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add availability and injury-related features.
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with availability features added
        """
        if df.empty:
            return df
        
        logger.info("Adding availability features")
        
        try:
            result_df = df.copy()
            
            # Basic availability features
            result_df = self._add_basic_availability_features(result_df)
            
            # Injury status features
            result_df = self._add_injury_status_features(result_df)
            
            # Minutes availability features
            result_df = self._add_minutes_availability_features(result_df)
            
            logger.info("Availability features added successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding availability features: {e}")
            return df
    
    def _add_basic_availability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic availability indicators."""
        result_df = df.copy()
        
        # Minutes-based availability
        if 'minutes' in df.columns:
            result_df['played'] = (result_df['minutes'] > 0).astype(int)
            result_df['started'] = (result_df['minutes'] >= 60).astype(int)
            result_df['full_game'] = (result_df['minutes'] >= 90).astype(int)
            result_df['came_off_bench'] = ((result_df['minutes'] > 0) & (result_df['minutes'] < 60)).astype(int)
        
        # Availability percentage (if we have multiple gameweeks)
        if 'player_name' in df.columns and 'played' in result_df.columns:
            availability_by_player = df.groupby('player_name')['played'].mean()
            result_df['availability_pct'] = result_df['player_name'].map(availability_by_player)
        
        return result_df
    
    def _add_injury_status_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add injury status features."""
        result_df = df.copy()
        
        try:
            # Get injury data
            injury_data = self.injury_provider.get_injury_data()
            
            if not injury_data.empty:
                # Merge injury data
                result_df = result_df.merge(
                    injury_data, 
                    left_on='element_id', 
                    right_on='element_id', 
                    how='left'
                )
                
                # Create injury indicators
                result_df['injured'] = result_df.get('injury_status', '') != 'available'
                result_df['doubtful'] = result_df.get('injury_status', '').str.contains('doubt', case=False, na=False)
                result_df['injured'] = result_df['injured'].fillna(False)
                result_df['doubtful'] = result_df['doubtful'].fillna(False)
                
                # Injury risk score
                result_df['injury_risk'] = result_df.get('injury_risk', 0.0).fillna(0.0)
            else:
                # Default values if no injury data
                result_df['injured'] = False
                result_df['doubtful'] = False
                result_df['injury_risk'] = 0.0
            
        except Exception as e:
            logger.warning(f"Could not load injury data: {e}")
            result_df['injured'] = False
            result_df['doubtful'] = False
            result_df['injury_risk'] = 0.0
        
        return result_df
    
    def _add_minutes_availability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add minutes-based availability features."""
        result_df = df.copy()
        
        if 'minutes' not in df.columns:
            return result_df
        
        # Minutes categories
        result_df['minutes_category'] = pd.cut(
            result_df['minutes'],
            bins=[-1, 0, 30, 60, 90, 120],
            labels=['No Play', 'Sub', 'Partial', 'Full', 'Extra'],
            include_lowest=True
        )
        
        # Rotation risk (if we have historical data)
        if 'player_name' in df.columns:
            player_minutes = df.groupby('player_name')['minutes']
            result_df['minutes_std'] = result_df['player_name'].map(player_minutes.std())
            result_df['minutes_mean'] = result_df['player_name'].map(player_minutes.mean())
            
            # Rotation risk indicator
            result_df['rotation_risk'] = np.where(
                (result_df['minutes_std'] > 20) & (result_df['minutes_mean'] < 70),
                1, 0
            )
        
        # Expected minutes based on recent form
        if 'minutes' in df.columns and 'player_name' in df.columns:
            # Calculate rolling average minutes (if multiple gameweeks available)
            minutes_rolling = df.groupby('player_name')['minutes'].rolling(window=3, min_periods=1).mean()
            result_df['expected_minutes'] = result_df['minutes']  # Default to current minutes
            
            # If we have historical data, use rolling average
            if len(df) > len(df['player_name'].unique()):
                result_df['expected_minutes'] = df.groupby('player_name')['minutes'].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
        
        return result_df
