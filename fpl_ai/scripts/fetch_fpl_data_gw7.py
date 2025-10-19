"""
Fetch real FPL data from API and save in vaastav format.

This script fetches live gameweek data from the FPL API and saves it
in the same format as the vaastav dataset for seamless integration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fpl_ai.src.providers.fpl_api import FPLAPIClient
from fpl_ai.src.common.config import get_logger

logger = get_logger(__name__)


def fetch_fpl_data_for_gw(gw: int) -> pd.DataFrame:
    """Fetch all player data from FPL API for a specific gameweek."""
    try:
        client = FPLAPIClient()
        
        # Get bootstrap data for player and team mapping
        bootstrap = client.get_bootstrap_data()
        if not bootstrap:
            logger.error("Could not get bootstrap data from FPL API")
            return pd.DataFrame()
        
        # Get live gameweek data
        live_data = client.get_gameweek_live(gw)
        if not live_data:
            logger.error(f"Could not get live data for GW{gw} from FPL API")
            return pd.DataFrame()
        
        # Create mappings
        players_df = pd.DataFrame(bootstrap['elements'])
        teams_df = pd.DataFrame(bootstrap['teams'])
        team_map = dict(zip(teams_df['id'], teams_df['name']))
        
        # Process live data into vaastav format
        vaastav_data = []
        
        for player_data in live_data['elements']:
            player_id = player_data['id']
            player_stats = player_data['stats']
            
            # Find player info in bootstrap
            player_info = players_df[players_df['id'] == player_id]
            if player_info.empty:
                continue
                
            player_info = player_info.iloc[0]
            
            # Map position codes to names
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            position = position_map.get(player_info['element_type'], 'UNK')
            
            # Create vaastav-style row
            row = {
                'name': player_info['web_name'],
                'position': position,
                'team': team_map.get(player_info['team'], f"Team_{player_info['team']}"),
                'xP': float(player_stats.get('expected_goals', 0)) + float(player_stats.get('expected_assists', 0)),
                'assists': player_stats['assists'],
                'bonus': player_stats['bonus'],
                'bps': player_stats['bps'],
                'clean_sheets': player_stats['clean_sheets'],
                'clearances_blocks_interceptions': player_stats.get('clearances_blocks_interceptions', 0),
                'creativity': float(player_stats.get('creativity', 0)),
                'defensive_contribution': player_stats.get('defensive_contribution', 0),
                'element': player_id,
                'expected_assists': float(player_stats.get('expected_assists', 0)),
                'expected_goal_involvements': float(player_stats.get('expected_goal_involvements', 0)),
                'expected_goals': float(player_stats.get('expected_goals', 0)),
                'expected_goals_conceded': float(player_stats.get('expected_goals_conceded', 0)),
                'fixture': player_data.get('explain', [{}])[0].get('fixture', 0) if player_data.get('explain') else 0,
                'goals_conceded': player_stats['goals_conceded'],
                'goals_scored': player_stats['goals_scored'],
                'ict_index': float(player_stats.get('ict_index', 0)),
                'influence': float(player_stats.get('influence', 0)),
                'kickoff_time': '',  # Not available in live data
                'minutes': player_stats['minutes'],
                'modified': player_data.get('modified', False),
                'opponent_team': 0,  # Not available in live data
                'own_goals': player_stats['own_goals'],
                'penalties_missed': player_stats['penalties_missed'],
                'penalties_saved': player_stats['penalties_saved'],
                'recoveries': player_stats.get('recoveries', 0),
                'red_cards': player_stats['red_cards'],
                'round': gw,
                'saves': player_stats['saves'],
                'selected': 0,  # Not available in live data
                'starts': player_stats.get('starts', 1 if player_stats['minutes'] > 0 else 0),
                'tackles': player_stats.get('tackles', 0),
                'team_a_score': 0,  # Not available in live data
                'team_h_score': 0,  # Not available in live data
                'threat': float(player_stats.get('threat', 0)),
                'total_points': player_stats['total_points'],
                'transfers_balance': 0,  # Not available in live data
                'transfers_in': 0,  # Not available in live data
                'transfers_out': 0,  # Not available in live data
                'value': player_info['now_cost'] / 10.0,  # Convert from tenths
                'was_home': False,  # Not available in live data
                'yellow_cards': player_stats['yellow_cards']
            }
            
            vaastav_data.append(row)
        
        result_df = pd.DataFrame(vaastav_data)
        logger.info(f"Fetched FPL API data for GW{gw}: {len(result_df)} players")
        return result_df
        
    except Exception as e:
        logger.error(f"Error fetching FPL API data for GW{gw}: {e}")
        return pd.DataFrame()


def save_vaastav_format(df: pd.DataFrame, gw: int):
    """Save DataFrame in vaastav format."""
    try:
        # Create directory if it doesn't exist
        vaastav_dir = Path("data/vaastav/data/2025-26/gws")
        vaastav_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual gameweek file
        gw_file = vaastav_dir / f"gw{gw}.csv"
        df.to_csv(gw_file, index=False)
        logger.info(f"Saved vaastav format data to {gw_file}")
        
        # Update merged_gw.csv if it exists
        merged_file = vaastav_dir / "merged_gw.csv"
        if merged_file.exists():
            # Read existing merged data
            existing_df = pd.read_csv(merged_file)
            
            # Remove any existing data for this gameweek
            existing_df = existing_df[existing_df['round'] != gw]
            
            # Add new data
            df_with_gw = df.copy()
            df_with_gw['GW'] = gw  # Add GW column for merged file
            
            # Combine and save
            combined_df = pd.concat([existing_df, df_with_gw], ignore_index=True)
            combined_df.to_csv(merged_file, index=False)
            logger.info(f"Updated merged_gw.csv with GW{gw} data")
        else:
            # Create new merged file
            df_with_gw = df.copy()
            df_with_gw['GW'] = gw
            df_with_gw.to_csv(merged_file, index=False)
            logger.info(f"Created new merged_gw.csv with GW{gw} data")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving vaastav format data for GW{gw}: {e}")
        return False


def fetch_and_save_gw7():
    """Fetch GW7 data from FPL API and save in vaastav format."""
    logger.info("Starting GW7 data fetch from FPL API...")
    
    # Fetch data
    gw7_data = fetch_fpl_data_for_gw(7)
    
    if gw7_data.empty:
        logger.error("Failed to fetch GW7 data from FPL API")
        return False
    
    # Save in vaastav format
    success = save_vaastav_format(gw7_data, 7)
    
    if success:
        logger.info(f"Successfully saved GW7 data: {len(gw7_data)} players")
        
        # Show sample statistics
        logger.info(f"GW7 stats: avg points={gw7_data['total_points'].mean():.1f}, "
                   f"max={gw7_data['total_points'].max()}, "
                   f"min={gw7_data['total_points'].min()}")
        
        # Show position breakdown
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_data = gw7_data[gw7_data['position'] == position]
            if not pos_data.empty:
                logger.info(f"  {position}: {len(pos_data)} players, "
                           f"avg points: {pos_data['total_points'].mean():.1f}")
        
        return True
    else:
        logger.error("Failed to save GW7 data")
        return False


if __name__ == "__main__":
    success = fetch_and_save_gw7()
    if success:
        print("✅ GW7 data successfully fetched and saved in vaastav format!")
    else:
        print("❌ Failed to fetch and save GW7 data")
