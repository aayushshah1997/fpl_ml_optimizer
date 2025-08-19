"""
FPL picks and team management client.

Provides functionality for retrieving user picks, team composition,
and GW1 baseline team data for planning.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from ..common.config import get_config, get_logger
from ..common.cache import get_cache
from .fpl_api import FPLAPIClient

logger = get_logger(__name__)


class FPLPicksClient:
    """
    Client for managing FPL picks and team state.
    """
    
    def __init__(self):
        """Initialize FPL picks client."""
        self.config = get_config()
        self.cache = get_cache()
        self.fpl_api = FPLAPIClient()
        
        logger.info("FPL Picks client initialized")
    
    def get_gw1_baseline_team(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get GW1 team as baseline for planning.
        
        Args:
            force_refresh: Force refresh from API
            
        Returns:
            Dictionary with team state (squad, bank, purchase prices)
        """
        cache_path = self.config.cache_dir / "team_state_gw1.json"
        
        # Try to load from cache first
        if cache_path.exists() and not force_refresh:
            try:
                with open(cache_path, 'r') as f:
                    team_state = json.load(f)
                logger.info("Loaded GW1 baseline team from cache")
                return team_state
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Invalid cached GW1 team data, refreshing from API")
        
        # Fetch from API
        team_data = self.fpl_api.get_my_team(gameweek=1)
        if not team_data:
            logger.error("Could not retrieve GW1 team from FPL API")
            return None
        
        picks_data = team_data['picks']
        entry_data = team_data['entry']
        
        if not picks_data or 'picks' not in picks_data:
            logger.error("Invalid picks data structure")
            return None
        
        # Process picks
        squad = []
        for pick in picks_data['picks']:
            squad.append({
                'element': pick['element'],
                'purchase_price': pick.get('purchase_price', 0),  # In tenths
                'selling_price': pick.get('selling_price', 0),   # In tenths
                'is_captain': pick.get('is_captain', False),
                'is_vice_captain': pick.get('is_vice_captain', False),
                'multiplier': pick.get('multiplier', 1)
            })
        
        # Get bank balance
        bank = entry_data.get('bank', 0) / 10 if entry_data else 0  # Convert from tenths
        
        # Calculate team value
        team_value = sum(p['selling_price'] for p in squad) / 10 + bank
        
        team_state = {
            'gameweek': 1,
            'squad': squad,
            'bank': bank,
            'team_value': team_value,
            'free_transfers': 1,
            'created_at': pd.Timestamp.now().isoformat(),
            'source': 'fpl_api'
        }
        
        # Save to cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(team_state, f, indent=2)
            logger.info(f"Saved GW1 baseline team to {cache_path}")
        except Exception as e:
            logger.warning(f"Could not save GW1 team to cache: {e}")
        
        return team_state
    
    def save_gw1_baseline_from_csv(self, csv_data: pd.DataFrame) -> bool:
        """
        Save GW1 baseline from uploaded CSV data.
        
        Args:
            csv_data: DataFrame with columns: element, purchase_price, selling_price, is_captain, bank
            
        Returns:
            True if saved successfully
        """
        try:
            squad = []
            bank = 0
            
            for _, row in csv_data.iterrows():
                # Handle bank row (if present)
                if pd.isna(row.get('element')) and not pd.isna(row.get('bank')):
                    bank = float(row['bank'])
                    continue
                
                # Process player row
                squad.append({
                    'element': int(row['element']),
                    'purchase_price': int(row.get('purchase_price', 0)),  # Tenths
                    'selling_price': int(row.get('selling_price', 0)),    # Tenths
                    'is_captain': bool(row.get('is_captain', False)),
                    'is_vice_captain': bool(row.get('is_vice_captain', False)),
                    'multiplier': 2 if row.get('is_captain', False) else 1
                })
            
            if len(squad) != 15:
                logger.error(f"Invalid squad size: {len(squad)} (expected 15)")
                return False
            
            # Calculate team value
            team_value = sum(p['selling_price'] for p in squad) / 10 + bank
            
            team_state = {
                'gameweek': 1,
                'squad': squad,
                'bank': bank,
                'team_value': team_value,
                'free_transfers': 1,
                'created_at': pd.Timestamp.now().isoformat(),
                'source': 'csv_upload'
            }
            
            # Save to cache
            cache_path = self.config.cache_dir / "team_state_gw1.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'w') as f:
                json.dump(team_state, f, indent=2)
            
            logger.info(f"Saved GW1 baseline from CSV: {len(squad)} players, bank £{bank}m")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save GW1 baseline from CSV: {e}")
            return False
    
    def get_current_team(self, gameweek: Optional[int] = None) -> Optional[Dict]:
        """
        Get current team for specified gameweek.
        
        Args:
            gameweek: Gameweek number (uses current if None)
            
        Returns:
            Current team data
        """
        if gameweek is None:
            # Get current gameweek
            bootstrap = self.fpl_api.get_bootstrap_data()
            if bootstrap:
                gameweek = next(
                    (event['id'] for event in bootstrap.get('events', []) if event.get('is_current')),
                    1
                )
            else:
                gameweek = 1
        
        return self.fpl_api.get_my_team(gameweek)
    
    def get_team_changes_since_gw1(self, current_gw: int) -> Optional[Dict]:
        """
        Get all transfers and changes since GW1.
        
        Args:
            current_gw: Current gameweek
            
        Returns:
            Dictionary with transfer history and net changes
        """
        # Get transfer history
        transfers_data = self.fpl_api.get_entry_transfers()
        if not transfers_data:
            return None
        
        transfers = transfers_data.get('transfers', [])
        
        # Process transfers
        transfers_out = []
        transfers_in = []
        total_cost = 0
        
        for transfer in transfers:
            if transfer.get('event', 999) <= current_gw:
                transfers_out.append(transfer['element_out'])
                transfers_in.append(transfer['element_in'])
                total_cost += transfer.get('element_out_cost', 0)
        
        return {
            'transfers_out': transfers_out,
            'transfers_in': transfers_in,
            'total_transfers': len(transfers),
            'total_cost': total_cost,
            'net_transfers': len(transfers_in) - len(transfers_out)
        }
    
    def validate_team_formation(self, squad: List[Dict], bootstrap_data: Dict) -> Tuple[bool, str]:
        """
        Validate team formation is legal.
        
        Args:
            squad: List of player picks
            bootstrap_data: FPL bootstrap data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get player positions
        elements = {elem['id']: elem for elem in bootstrap_data['elements']}
        
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        team_counts = {}
        
        for pick in squad:
            element_id = pick['element']
            if element_id not in elements:
                return False, f"Invalid player ID: {element_id}"
            
            player = elements[element_id]
            position = position_map[player['element_type']]
            position_counts[position] += 1
            
            # Count players per team
            team_id = player['team']
            team_counts[team_id] = team_counts.get(team_id, 0) + 1
        
        # Check position limits
        limits = {'GK': (2, 2), 'DEF': (5, 5), 'MID': (5, 5), 'FWD': (3, 3)}
        for pos, (min_count, max_count) in limits.items():
            count = position_counts[pos]
            if not (min_count <= count <= max_count):
                return False, f"Invalid {pos} count: {count} (must be {min_count}-{max_count})"
        
        # Check team limits (max 3 per team)
        for team_id, count in team_counts.items():
            if count > 3:
                team_name = next(
                    (team['name'] for team in bootstrap_data['teams'] if team['id'] == team_id),
                    f"Team {team_id}"
                )
                return False, f"Too many players from {team_name}: {count} (max 3)"
        
        return True, "Valid formation"
    
    def calculate_team_value(self, squad: List[Dict]) -> float:
        """
        Calculate total team value including bank.
        
        Args:
            squad: List of player picks with selling prices
            
        Returns:
            Total team value in millions
        """
        squad_value = sum(pick.get('selling_price', 0) for pick in squad) / 10
        return squad_value
    
    def get_picks_summary(self, gameweek: int) -> Optional[Dict]:
        """
        Get comprehensive picks summary for a gameweek.
        
        Args:
            gameweek: Gameweek number
            
        Returns:
            Dictionary with picks, formation, and team stats
        """
        picks_data = self.fpl_api.get_entry_picks(gameweek=gameweek)
        if not picks_data:
            return None
        
        bootstrap = self.fpl_api.get_bootstrap_data()
        if not bootstrap:
            return None
        
        picks = picks_data.get('picks', [])
        
        # Get player details
        elements = {elem['id']: elem for elem in bootstrap['elements']}
        teams = {team['id']: team for team in bootstrap['teams']}
        
        # Process picks
        squad_details = []
        for pick in picks:
            element_id = pick['element']
            if element_id in elements:
                player = elements[element_id]
                team = teams[player['team']]
                
                squad_details.append({
                    'element_id': element_id,
                    'web_name': player['web_name'],
                    'team': team['short_name'],
                    'position': player['element_type'],
                    'now_cost': player['now_cost'],
                    'selling_price': pick.get('selling_price', 0),
                    'is_captain': pick.get('is_captain', False),
                    'is_vice_captain': pick.get('is_vice_captain', False),
                    'multiplier': pick.get('multiplier', 1)
                })
        
        return {
            'gameweek': gameweek,
            'squad': squad_details,
            'entry_history': picks_data.get('entry_history', {}),
            'active_chip': picks_data.get('active_chip'),
            'automatic_subs': picks_data.get('automatic_subs', [])
        }
