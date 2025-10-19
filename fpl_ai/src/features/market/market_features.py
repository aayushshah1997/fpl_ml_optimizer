"""
Market and transfer features module for feature building.

Handles calculation of market-related features like ownership, transfer activity, and value metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from ...common.config import get_config, get_logger

logger = get_logger(__name__)


class MarketFeatureCalculator:
    """Calculates market-related features for players."""
    
    def __init__(self):
        """Initialize market feature calculator."""
        self.config = get_config()
    
    def add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market-related features to the dataset.
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with market features added
        """
        if df.empty:
            return df
        
        logger.info("Adding market features")
        
        try:
            result_df = df.copy()
            
            # Ownership features
            result_df = self._add_ownership_features(result_df)
            
            # Transfer activity features
            result_df = self._add_transfer_features(result_df)
            
            # Value metrics
            result_df = self._add_value_features(result_df)
            
            # Market trends
            result_df = self._add_market_trend_features(result_df)
            
            logger.info("Market features added successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding market features: {e}")
            return df
    
    def _add_ownership_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ownership-related features."""
        if 'selected_by_percent' not in df.columns:
            return df
        
        result_df = df.copy()
        
        # Ownership categories
        result_df['high_ownership'] = (result_df['selected_by_percent'] > 20).astype(int)
        result_df['medium_ownership'] = ((result_df['selected_by_percent'] >= 5) & 
                                       (result_df['selected_by_percent'] <= 20)).astype(int)
        result_df['low_ownership'] = (result_df['selected_by_percent'] < 5).astype(int)
        
        # Ownership by position
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_mask = result_df['position'] == position
            if pos_mask.any():
                pos_ownership = result_df.loc[pos_mask, 'selected_by_percent']
                result_df.loc[pos_mask, 'ownership_rank'] = pos_ownership.rank(ascending=False, method='dense')
        
        # Differential ownership (vs expected based on points)
        if 'total_points' in df.columns and 'now_cost' in df.columns:
            # Calculate expected ownership based on points per million
            df['points_per_million'] = df['total_points'] / df['now_cost'].replace(0, 1)
            expected_ownership = df.groupby('position')['points_per_million'].rank(pct=True) * 100
            result_df['ownership_differential'] = df['selected_by_percent'] - expected_ownership
        
        return result_df
    
    def _add_transfer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transfer activity features."""
        result_df = df.copy()
        
        # Transfer balance features
        if 'transfers_balance' in df.columns:
            result_df['net_transfers_positive'] = (result_df['transfers_balance'] > 0).astype(int)
            result_df['net_transfers_negative'] = (result_df['transfers_balance'] < 0).astype(int)
            result_df['high_transfer_activity'] = (result_df['transfers_balance'].abs() > 1000).astype(int)
        
        if 'transfers_in' in df.columns and 'transfers_out' in df.columns:
            # Transfer velocity (rate of change)
            total_transfers = result_df['transfers_in'] + result_df['transfers_out']
            result_df['transfer_velocity'] = total_transfers / result_df['now_cost'].replace(0, 1)
            
            # Transfer sentiment (in vs out ratio)
            result_df['transfer_sentiment'] = np.where(
                result_df['transfers_out'] > 0,
                result_df['transfers_in'] / result_df['transfers_out'],
                0
            )
        
        return result_df
    
    def _add_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add value-related features."""
        if 'total_points' not in df.columns or 'now_cost' not in df.columns:
            return df
        
        result_df = df.copy()
        
        # Basic value metrics
        result_df['points_per_million'] = result_df['total_points'] / result_df['now_cost'].replace(0, 1)
        result_df['value_form'] = result_df.get('value_form', 0)
        result_df['value_season'] = result_df.get('value_season', 0)
        
        # Value efficiency (points per million vs position average)
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_mask = result_df['position'] == position
            if pos_mask.any():
                pos_ppm = result_df.loc[pos_mask, 'points_per_million']
                pos_avg_ppm = pos_ppm.mean()
                result_df.loc[pos_mask, 'value_efficiency'] = pos_ppm / pos_avg_ppm if pos_avg_ppm > 0 else 1
        
        # Price change indicators (if available)
        if 'now_cost' in df.columns:
            # Calculate price change from previous gameweek (if data available)
            result_df['price_change'] = 0  # Placeholder - would need historical price data
        
        # Value categories
        result_df['high_value'] = (result_df['points_per_million'] > result_df['points_per_million'].quantile(0.75)).astype(int)
        result_df['low_value'] = (result_df['points_per_million'] < result_df['points_per_million'].quantile(0.25)).astype(int)
        
        return result_df
    
    def _add_market_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market trend features."""
        result_df = df.copy()
        
        # Form vs market perception
        if 'form' in df.columns and 'selected_by_percent' in df.columns:
            # Calculate correlation between form and ownership
            form_ownership_corr = df.groupby('position').apply(
                lambda x: x['form'].corr(x['selected_by_percent']) if len(x) > 1 else 0
            )
            
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                pos_mask = result_df['position'] == position
                if pos_mask.any():
                    result_df.loc[pos_mask, 'form_ownership_correlation'] = form_ownership_corr.get(position, 0)
        
        # Market inefficiency indicators
        if 'total_points' in df.columns and 'selected_by_percent' in df.columns:
            # Players with high points but low ownership (potential gems)
            points_rank = df.groupby('position')['total_points'].rank(pct=True)
            ownership_rank = df.groupby('position')['selected_by_percent'].rank(pct=True)
            
            result_df['market_inefficiency'] = points_rank - ownership_rank
            result_df['potential_gem'] = ((result_df['market_inefficiency'] > 0.3) & 
                                        (result_df['selected_by_percent'] < 10)).astype(int)
        
        return result_df
