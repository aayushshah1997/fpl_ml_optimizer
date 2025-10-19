"""
Historical data loading module for feature building.

Handles loading and processing of historical season data from multiple sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ...common.config import get_config, get_logger
from ...common.cache import get_cache
from ...common.timeutil import get_season_dates
from ...providers.fbrapi_client import FBRAPIClient
from ...providers.fpl_map import FPLMapper
from ...providers.league_strength import strength_and_weight_mult, log_seen_leagues

logger = get_logger(__name__)


class HistoricalDataLoader:
    """Loads and processes historical season data."""
    
    def __init__(self):
        """Initialize historical data loader."""
        self.config = get_config()
        self.cache = get_cache()
        self.fbr_client = FBRAPIClient()
        self.fpl_mapper = FPLMapper()
    
    def get_historical_season_data(self, season: str, start_gw: int, end_gw: int) -> pd.DataFrame:
        """
        Get historical season data for specified gameweeks.
        
        Args:
            season: Season string (e.g., '2023-24')
            start_gw: Starting gameweek
            end_gw: Ending gameweek
            
        Returns:
            DataFrame with historical player data
        """
        logger.info(f"Loading historical data for {season} GW{start_gw}-{end_gw}")
        
        try:
            # Load individual season data
            season_data = self._load_individual_season_data(season, start_gw, end_gw)
            
            if season_data.empty:
                logger.warning(f"No data found for season {season}")
                return pd.DataFrame()
            
            # Map vaastav schema to standard format
            season_data = self._map_vaastav_schema(season_data)
            
            # Apply league strength scaling
            season_data = self._apply_league_strength_scaling(season_data)
            
            logger.info(f"Loaded {len(season_data)} historical records for {season}")
            return season_data
            
        except Exception as e:
            logger.error(f"Error loading historical data for {season}: {e}")
            return pd.DataFrame()
    
    def _load_individual_season_data(self, season: str, start_gw: int, end_gw: int) -> pd.DataFrame:
        """Load data for individual season from vaastav files."""
        try:
            data_dir = Path(self.config.get("io", {}).get("data_dir", "fpl_ai/data"))
            season_dir = data_dir / "vaastav" / "data" / season
            
            if not season_dir.exists():
                logger.warning(f"Season directory not found: {season_dir}")
                return pd.DataFrame()
            
            # Load all CSV files for the season
            csv_files = list(season_dir.glob("*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in {season_dir}")
                return pd.DataFrame()
            
            all_data = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Error reading {csv_file}: {e}")
                    continue
            
            if not all_data:
                return pd.DataFrame()
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Filter by gameweek if specified
            if 'GW' in combined_df.columns:
                combined_df = combined_df[
                    (combined_df['GW'] >= start_gw) & 
                    (combined_df['GW'] <= end_gw)
                ]
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading individual season data for {season}: {e}")
            return pd.DataFrame()
    
    def _map_vaastav_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map vaastav data schema to standard format."""
        if df.empty:
            return df
        
        try:
            # Create a copy to avoid modifying original
            mapped_df = df.copy()
            
            # Standardize column names
            column_mapping = {
                'name': 'player_name',
                'team': 'team_name',
                'position': 'position',
                'GW': 'gameweek',
                'points': 'total_points',
                'minutes': 'minutes',
                'goals_scored': 'goals',
                'assists': 'assists',
                'clean_sheets': 'clean_sheets',
                'saves': 'saves',
                'bonus': 'bonus_points',
                'yellow_cards': 'yellow_cards',
                'red_cards': 'red_cards',
                'own_goals': 'own_goals',
                'penalties_missed': 'penalties_missed',
                'penalties_saved': 'penalties_saved'
            }
            
            # Rename columns that exist
            for old_name, new_name in column_mapping.items():
                if old_name in mapped_df.columns:
                    mapped_df = mapped_df.rename(columns={old_name: new_name})
            
            # Add season identifier
            mapped_df['season'] = self._extract_season_from_data(df)
            
            # Add position mapping
            if 'position' in mapped_df.columns:
                position_map = {
                    1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD',
                    'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'
                }
                mapped_df['position'] = mapped_df['position'].map(position_map).fillna('UNKNOWN')
            
            # Ensure numeric columns are properly typed
            numeric_columns = [
                'gameweek', 'total_points', 'minutes', 'goals', 'assists',
                'clean_sheets', 'saves', 'bonus_points', 'yellow_cards',
                'red_cards', 'own_goals', 'penalties_missed', 'penalties_saved'
            ]
            
            for col in numeric_columns:
                if col in mapped_df.columns:
                    mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce').fillna(0)
            
            # Add derived features
            mapped_df['goals_conceded'] = mapped_df.get('goals_conceded', 0)
            mapped_df['creativity'] = mapped_df.get('creativity', 0)
            mapped_df['influence'] = mapped_df.get('influence', 0)
            mapped_df['threat'] = mapped_df.get('threat', 0)
            
            return mapped_df
            
        except Exception as e:
            logger.error(f"Error mapping vaastav schema: {e}")
            return df
    
    def _extract_season_from_data(self, df: pd.DataFrame) -> str:
        """Extract season from data if available."""
        # Try to extract from filename or data
        # For now, return a default season
        return "2023-24"
    
    def _apply_league_strength_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply league strength scaling to historical data."""
        if df.empty:
            return df
        
        try:
            # Get league strength multipliers
            strength_data = strength_and_weight_mult()
            
            if not strength_data:
                logger.warning("No league strength data available")
                return df
            
            # Apply scaling based on team strength
            scaled_df = df.copy()
            
            # Log seen leagues for tracking
            unique_teams = scaled_df['team_name'].unique()
            log_seen_leagues(unique_teams)
            
            # Apply strength multipliers to key metrics
            strength_columns = ['total_points', 'goals', 'assists', 'clean_sheets']
            
            for team in unique_teams:
                team_mask = scaled_df['team_name'] == team
                strength_mult = strength_data.get(team, 1.0)
                
                for col in strength_columns:
                    if col in scaled_df.columns:
                        scaled_df.loc[team_mask, col] = scaled_df.loc[team_mask, col] * strength_mult
            
            return scaled_df
            
        except Exception as e:
            logger.error(f"Error applying league strength scaling: {e}")
            return df
