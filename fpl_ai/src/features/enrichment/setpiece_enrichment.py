"""
Set piece enrichment module for feature building.

Handles calculation of set piece taker features and related metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from ...common.config import get_config, get_logger
from ...providers.setpiece_roles import SetPieceRolesManager

logger = get_logger(__name__)


class SetPieceEnricher:
    """Enriches data with set piece taker features."""
    
    def __init__(self):
        """Initialize set piece enricher."""
        self.config = get_config()
        self.setpiece_manager = SetPieceRolesManager()
    
    def add_setpiece_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add set piece taker features.
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with set piece features added
        """
        if df.empty:
            return df
        
        logger.info("Adding set piece features")
        
        try:
            result_df = df.copy()
            
            # Basic set piece features
            result_df = self._add_basic_setpiece_features(result_df)
            
            # Set piece taker indicators
            result_df = self._add_setpiece_taker_features(result_df)
            
            # Set piece value features
            result_df = self._add_setpiece_value_features(result_df)
            
            logger.info("Set piece features added successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding set piece features: {e}")
            return df
    
    def _add_basic_setpiece_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic set piece related features."""
        result_df = df.copy()
        
        # Initialize set piece features
        result_df['penalty_taker'] = 0
        result_df['free_kick_taker'] = 0
        result_df['corner_taker'] = 0
        result_df['set_piece_taker'] = 0
        
        # Position-based set piece likelihood
        position_setpiece_map = {
            'GK': 0.1,
            'DEF': 0.2,
            'MID': 0.7,
            'FWD': 0.4
        }
        
        result_df['setpiece_likelihood'] = result_df['position'].map(position_setpiece_map).fillna(0.3)
        
        return result_df
    
    def _add_setpiece_taker_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add set piece taker indicators."""
        result_df = df.copy()
        
        try:
            # Get set piece data
            setpiece_data = self.setpiece_manager.get_setpiece_roles()
            
            if not setpiece_data.empty:
                # Merge set piece data
                result_df = result_df.merge(
                    setpiece_data,
                    left_on='element_id',
                    right_on='element_id',
                    how='left'
                )
                
                # Create set piece indicators
                result_df['penalty_taker'] = result_df.get('penalty_taker', 0).fillna(0)
                result_df['free_kick_taker'] = result_df.get('free_kick_taker', 0).fillna(0)
                result_df['corner_taker'] = result_df.get('corner_taker', 0).fillna(0)
                
                # Combined set piece taker
                result_df['set_piece_taker'] = (
                    result_df['penalty_taker'] + 
                    result_df['free_kick_taker'] + 
                    result_df['corner_taker']
                ).clip(upper=1)
                
            else:
                # Default values based on position and form
                result_df = self._add_default_setpiece_features(result_df)
                
        except Exception as e:
            logger.warning(f"Could not load set piece data: {e}")
            result_df = self._add_default_setpiece_features(result_df)
        
        return result_df
    
    def _add_default_setpiece_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add default set piece features when data is unavailable."""
        result_df = df.copy()
        
        # Position-based defaults
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_mask = result_df['position'] == position
            
            if position == 'MID':
                # Midfielders more likely to be set piece takers
                result_df.loc[pos_mask, 'setpiece_likelihood'] = 0.6
                result_df.loc[pos_mask, 'free_kick_taker'] = 0.3
                result_df.loc[pos_mask, 'corner_taker'] = 0.4
            elif position == 'FWD':
                # Forwards more likely to be penalty takers
                result_df.loc[pos_mask, 'penalty_taker'] = 0.2
                result_df.loc[pos_mask, 'free_kick_taker'] = 0.1
            elif position == 'DEF':
                # Defenders occasionally take free kicks
                result_df.loc[pos_mask, 'free_kick_taker'] = 0.1
        
        # Form-based adjustment
        if 'form' in df.columns:
            high_form_mask = df['form'] > df['form'].quantile(0.75)
            result_df.loc[high_form_mask, 'penalty_taker'] *= 1.2
            result_df.loc[high_form_mask, 'free_kick_taker'] *= 1.2
        
        # Ensure binary indicators
        result_df['penalty_taker'] = (result_df['penalty_taker'] > 0.5).astype(int)
        result_df['free_kick_taker'] = (result_df['free_kick_taker'] > 0.5).astype(int)
        result_df['corner_taker'] = (result_df['corner_taker'] > 0.5).astype(int)
        result_df['set_piece_taker'] = (
            result_df['penalty_taker'] + 
            result_df['free_kick_taker'] + 
            result_df['corner_taker']
        ).clip(upper=1)
        
        return result_df
    
    def _add_setpiece_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add set piece value enhancement features."""
        result_df = df.copy()
        
        # Set piece value multiplier
        if 'total_points' in df.columns:
            # Set piece takers get value boost
            setpiece_boost = 1.0 + (result_df['set_piece_taker'] * 0.15)
            result_df['setpiece_value_multiplier'] = setpiece_boost
            
            # Penalty taker bonus (higher value)
            penalty_boost = 1.0 + (result_df['penalty_taker'] * 0.25)
            result_df['penalty_value_multiplier'] = penalty_boost
        
        # Expected set piece points
        if 'total_points' in df.columns and 'minutes' in df.columns:
            # Estimate set piece contribution to total points
            setpiece_points = np.where(
                result_df['minutes'] > 0,
                result_df['total_points'] * result_df['setpiece_likelihood'] * 0.3,
                0
            )
            result_df['expected_setpiece_points'] = setpiece_points
        
        # Set piece opportunity score
        result_df['setpiece_opportunity_score'] = (
            result_df['penalty_taker'] * 3 +  # Penalties are most valuable
            result_df['free_kick_taker'] * 2 +  # Free kicks second
            result_df['corner_taker'] * 1  # Corners least valuable
        )
        
        return result_df
