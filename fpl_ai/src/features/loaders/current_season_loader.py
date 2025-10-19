"""
Current season data loading module for feature building.

Handles loading and processing of current season data from FPL API and other sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ...common.config import get_config, get_logger
from ...common.cache import get_cache
from ...common.timeutil import get_current_season
from ...providers.fpl_api import FPLAPIClient
from ...providers.fpl_map import FPLMapper

logger = get_logger(__name__)


class CurrentSeasonLoader:
    """Loads and processes current season data."""
    
    def __init__(self):
        """Initialize current season loader."""
        self.config = get_config()
        self.cache = get_cache()
        self.fpl_api = FPLAPIClient()
        self.fpl_mapper = FPLMapper()
    
    def get_current_season_data(self, start_gw: int, end_gw: int) -> pd.DataFrame:
        """
        Get current season data for specified gameweeks.
        
        Args:
            start_gw: Starting gameweek
            end_gw: Ending gameweek
            
        Returns:
            DataFrame with current season player data
        """
        logger.info(f"Loading current season data for GW{start_gw}-{end_gw}")
        
        try:
            # Get current season bootstrap data
            bootstrap_data = self.fpl_api.get_bootstrap_data()
            if not bootstrap_data:
                logger.error("Failed to get bootstrap data")
                return pd.DataFrame()
            
            # Get player data
            player_data = self._get_current_player_data(bootstrap_data)
            
            if player_data.empty:
                logger.warning("No current season player data found")
                return pd.DataFrame()
            
            # Add season identifier
            player_data['season'] = get_current_season()
            
            logger.info(f"Loaded {len(player_data)} current season records")
            return player_data
            
        except Exception as e:
            logger.error(f"Error loading current season data: {e}")
            return pd.DataFrame()
    
    def _get_current_player_data(self, bootstrap_data: Dict) -> pd.DataFrame:
        """Extract current player data from bootstrap."""
        try:
            if 'elements' not in bootstrap_data:
                logger.error("No elements data in bootstrap")
                return pd.DataFrame()
            
            elements = bootstrap_data['elements']
            teams = {team['id']: team for team in bootstrap_data.get('teams', [])}
            
            player_data = []
            for player in elements:
                team_id = player.get('team', 0)
                team_info = teams.get(team_id, {})
                
                player_record = {
                    'element_id': player.get('id', 0),
                    'player_name': player.get('web_name', ''),
                    'first_name': player.get('first_name', ''),
                    'second_name': player.get('second_name', ''),
                    'team_name': team_info.get('short_name', ''),
                    'team_id': team_id,
                    'position': self._map_position(player.get('element_type', 0)),
                    'now_cost': player.get('now_cost', 0) / 10.0,  # Convert to millions
                    'selected_by_percent': player.get('selected_by_percent', 0.0),
                    'form': player.get('form', 0.0),
                    'total_points': player.get('total_points', 0),
                    'minutes': player.get('minutes', 0),
                    'goals_scored': player.get('goals_scored', 0),
                    'assists': player.get('assists', 0),
                    'clean_sheets': player.get('clean_sheets', 0),
                    'saves': player.get('saves', 0),
                    'bonus': player.get('bonus', 0),
                    'yellow_cards': player.get('yellow_cards', 0),
                    'red_cards': player.get('red_cards', 0),
                    'own_goals': player.get('own_goals', 0),
                    'penalties_missed': player.get('penalties_missed', 0),
                    'penalties_saved': player.get('penalties_saved', 0),
                    'goals_conceded': player.get('goals_conceded', 0),
                    'creativity': player.get('creativity', 0),
                    'influence': player.get('influence', 0),
                    'threat': player.get('threat', 0),
                    'ict_index': player.get('ict_index', 0.0),
                    'transfers_in': player.get('transfers_in', 0),
                    'transfers_out': player.get('transfers_out', 0),
                    'transfers_balance': player.get('transfers_in', 0) - player.get('transfers_out', 0),
                    'value_form': player.get('value_form', 0.0),
                    'value_season': player.get('value_season', 0.0),
                    'points_per_game': player.get('points_per_game', 0.0),
                    'starts': player.get('starts', 0),
                    'expected_goals': player.get('expected_goals', 0.0),
                    'expected_assists': player.get('expected_assists', 0.0),
                    'expected_goal_involvements': player.get('expected_goal_involvements', 0.0),
                    'expected_goals_conceded': player.get('expected_goals_conceded', 0.0),
                    'expected_saves': player.get('expected_saves', 0.0),
                    'expected_clean_sheets': player.get('expected_clean_sheets', 0.0)
                }
                
                player_data.append(player_record)
            
            df = pd.DataFrame(player_data)
            
            # Filter out players with no minutes or invalid data
            df = df[df['minutes'] > 0]
            df = df[df['player_name'].notna()]
            df = df[df['team_name'] != '']
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting current player data: {e}")
            return pd.DataFrame()
    
    def _map_position(self, element_type: int) -> str:
        """Map FPL element type to position string."""
        position_map = {
            1: 'GK',
            2: 'DEF', 
            3: 'MID',
            4: 'FWD'
        }
        return position_map.get(element_type, 'UNKNOWN')
    
    def get_player_master_data(self) -> pd.DataFrame:
        """Load player master data if available."""
        try:
            data_dir = Path(self.config.get("io", {}).get("data_dir", "fpl_ai/data"))
            player_master_file = data_dir / "player_master.parquet"
            
            if player_master_file.exists():
                df = pd.read_parquet(player_master_file)
                logger.info(f"Loaded player master data: {len(df)} records")
                return df
            else:
                logger.info("Player master file not found, using bootstrap data only")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading player master data: {e}")
            return pd.DataFrame()
