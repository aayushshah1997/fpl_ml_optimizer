"""
Unified team building module - single source of truth for FPL team construction.

Consolidates squad building, position limits, team constraints, and budget optimization
to eliminate duplication across the codebase.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


class TeamBuilder:
    """
    Unified team builder for FPL squad construction.
    """
    
    def __init__(self):
        """Initialize team builder with FPL constraints."""
        self.config = get_config()
        
        # FPL position constraints
        self.position_limits = {
            'GK': 2,  # Exactly 2 goalkeepers
            'DEF': 5,  # Exactly 5 defenders
            'MID': 5,  # Exactly 5 midfielders
            'FWD': 3   # Exactly 3 forwards
        }
        
        self.squad_size = 15
        self.max_players_per_team = 3
        self.total_budget = 100.0
        
        logger.info("Team builder initialized")
    
    def build_squad(
        self,
        predictions_df: pd.DataFrame,
        budget: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Build a complete FPL squad using greedy value-based selection.
        
        Args:
            predictions_df: DataFrame with player predictions including:
                - element_id: Player ID
                - web_name: Player name
                - position: GK/DEF/MID/FWD
                - team_name: Team name
                - proj_points: Projected points
                - now_cost: Player cost
            budget: Budget limit (default: 100.0)
            
        Returns:
            List of player dictionaries forming a valid FPL squad
        """
        if predictions_df.empty:
            logger.warning("Empty predictions DataFrame provided")
            return []
        
        budget = budget or self.total_budget
        logger.info(f"Building squad with budget £{budget:.1f}M")
        
        # Clean and prepare data
        squad_data = self._prepare_prediction_data(predictions_df)
        if squad_data.empty:
            return []
        
        # Build squad using greedy selection
        squad = self._greedy_squad_selection(squad_data, budget)
        
        # Validate squad
        is_valid, error_msg = self._validate_squad(squad)
        if not is_valid:
            logger.error(f"Invalid squad built: {error_msg}")
            return []
        
        logger.info(f"Squad built: {len(squad)} players, cost £{self._calculate_squad_cost(squad):.1f}M")
        return squad
    
    def _prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean prediction data for team building."""
        # Ensure required columns exist
        required_cols = ['element_id', 'web_name', 'position', 'team_name', 'now_cost']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Handle proj_points - use mean_points if proj_points not available
        if 'proj_points' not in df.columns:
            if 'mean_points' in df.columns:
                df = df.copy()
                df['proj_points'] = df['mean_points']
                logger.info("Using mean_points as proj_points")
            else:
                logger.error("Neither proj_points nor mean_points column found")
                return pd.DataFrame()
        
        # Clean data
        clean_df = df.copy()
        
        # Remove duplicates by element_id
        clean_df = clean_df.drop_duplicates(subset=['element_id'])
        
        # Filter valid positions
        valid_positions = list(self.position_limits.keys())
        clean_df = clean_df[clean_df['position'].isin(valid_positions)]
        
        # Handle missing values
        clean_df['proj_points'] = pd.to_numeric(clean_df['proj_points'], errors='coerce').fillna(0.0)
        clean_df['now_cost'] = pd.to_numeric(clean_df['now_cost'], errors='coerce').fillna(5.0)
        clean_df['team_name'] = clean_df['team_name'].fillna('Unknown')
        
        # Calculate value (points per million)
        clean_df['value'] = clean_df['proj_points'] / clean_df['now_cost']
        
        # Sort by value for greedy selection
        clean_df = clean_df.sort_values('value', ascending=False)
        
        logger.info(f"Prepared {len(clean_df)} players for team building")
        return clean_df
    
    def _greedy_squad_selection(
        self,
        data: pd.DataFrame,
        budget: float
    ) -> List[Dict[str, Any]]:
        """Build squad using greedy value-based selection."""
        squad = []
        team_counts = {}  # Track players per team
        used_budget = 0.0
        
        # First pass: Fill minimum requirements for each position
        for position, count in self.position_limits.items():
            pos_data = data[data['position'] == position]
            
            for _, player in pos_data.iterrows():
                # Check if we already have enough players of this position
                current_pos_count = len([p for p in squad if p['position'] == position])
                if current_pos_count >= count:
                    break
                
                player_cost = player['now_cost']
                team_name = player['team_name']
                
                # Check constraints
                if (team_counts.get(team_name, 0) < self.max_players_per_team and
                    used_budget + player_cost <= budget):
                    
                    player_dict = {
                        'element_id': player['element_id'],
                        'web_name': player['web_name'],
                        'position': position,
                        'team_name': team_name,
                        'proj_points': player['proj_points'],
                        'now_cost': player_cost,
                        'value': player['value']
                    }
                    
                    squad.append(player_dict)
                    team_counts[team_name] = team_counts.get(team_name, 0) + 1
                    used_budget += player_cost
        
        # Second pass: Fill remaining slots to reach 15 players
        remaining_slots = self.squad_size - len(squad)
        if remaining_slots > 0:
            # Get all players not yet selected
            selected_ids = {p['element_id'] for p in squad}
            available = data[~data['element_id'].isin(selected_ids)]
            
            for _, player in available.iterrows():
                if len(squad) >= self.squad_size:
                    break
                
                player_cost = player['now_cost']
                team_name = player['team_name']
                
                if (team_counts.get(team_name, 0) < self.max_players_per_team and
                    used_budget + player_cost <= budget):
                    
                    player_dict = {
                        'element_id': player['element_id'],
                        'web_name': player['web_name'],
                        'position': player['position'],
                        'team_name': team_name,
                        'proj_points': player['proj_points'],
                        'now_cost': player_cost,
                        'value': player['value']
                    }
                    
                    squad.append(player_dict)
                    team_counts[team_name] = team_counts.get(team_name, 0) + 1
                    used_budget += player_cost
        
        # Third pass: Try to upgrade players with remaining budget
        if len(squad) == self.squad_size and used_budget < budget * 0.9:
            squad = self._try_upgrades(squad, data, budget, team_counts)
        
        return squad
    
    def _try_upgrades(
        self,
        squad: List[Dict],
        data: pd.DataFrame,
        budget: float,
        team_counts: Dict[str, int]
    ) -> List[Dict]:
        """Try to upgrade players with remaining budget."""
        remaining_budget = budget - self._calculate_squad_cost(squad)
        
        if remaining_budget <= 0:
            return squad
        
        # Get available players not in squad
        squad_ids = {p['element_id'] for p in squad}
        available = data[~data['element_id'].isin(squad_ids)]
        
        # Sort by value for upgrade attempts
        available = available.sort_values('value', ascending=False)
        
        for _, upgrade_player in available.iterrows():
            if remaining_budget <= 0:
                break
            
            upgrade_cost = upgrade_player['now_cost']
            upgrade_team = upgrade_player['team_name']
            upgrade_position = upgrade_player['position']
            
            # Find a player to replace (lowest value in same position)
            same_pos_players = [p for p in squad if p['position'] == upgrade_position]
            if not same_pos_players:
                continue
            
            # Sort by value to find worst player
            same_pos_players.sort(key=lambda x: x['value'])
            worst_player = same_pos_players[0]
            
            # Check if upgrade is worthwhile
            cost_difference = upgrade_cost - worst_player['now_cost']
            value_improvement = upgrade_player['value'] - worst_player['value']
            
            if (value_improvement > 0 and
                cost_difference <= remaining_budget and
                team_counts.get(upgrade_team, 0) < self.max_players_per_team):
                
                # Perform upgrade
                squad.remove(worst_player)
                squad.append({
                    'element_id': upgrade_player['element_id'],
                    'web_name': upgrade_player['web_name'],
                    'position': upgrade_position,
                    'team_name': upgrade_team,
                    'proj_points': upgrade_player['proj_points'],
                    'now_cost': upgrade_cost,
                    'value': upgrade_player['value']
                })
                
                # Update team counts
                old_team = worst_player['team_name']
                team_counts[old_team] = max(0, team_counts.get(old_team, 1) - 1)
                team_counts[upgrade_team] = team_counts.get(upgrade_team, 0) + 1
                
                remaining_budget -= cost_difference
                logger.info(f"Upgraded {worst_player['web_name']} -> {upgrade_player['web_name']}")
        
        return squad
    
    def _validate_squad(self, squad: List[Dict]) -> Tuple[bool, str]:
        """Validate squad meets FPL constraints."""
        if len(squad) != self.squad_size:
            return False, f"Squad size {len(squad)} != {self.squad_size}"
        
        # Check position counts
        position_counts = {}
        team_counts = {}
        
        for player in squad:
            pos = player['position']
            team = player['team_name']
            
            position_counts[pos] = position_counts.get(pos, 0) + 1
            team_counts[team] = team_counts.get(team, 0) + 1
        
        # Validate position limits
        for position, required in self.position_limits.items():
            actual = position_counts.get(position, 0)
            if actual != required:
                return False, f"{position} count {actual} != {required}"
        
        # Validate team limits
        for team, count in team_counts.items():
            if count > self.max_players_per_team:
                return False, f"Team {team} has {count} players (max {self.max_players_per_team})"
        
        return True, "Valid squad"
    
    def _calculate_squad_cost(self, squad: List[Dict]) -> float:
        """Calculate total cost of squad."""
        return sum(player['now_cost'] for player in squad)


# Convenience function for backward compatibility
def build_squad_from_predictions(
    predictions_df: pd.DataFrame,
    budget: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Build FPL squad from predictions DataFrame.
    
    Args:
        predictions_df: DataFrame with player predictions
        budget: Budget limit (default: 100.0)
        
    Returns:
        List of player dictionaries forming valid FPL squad
    """
    builder = TeamBuilder()
    return builder.build_squad(predictions_df, budget)
