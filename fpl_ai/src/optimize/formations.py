"""
Formation validation and legal team composition checking.

Ensures team selections comply with FPL rules including position limits,
team limits, and budget constraints.
"""

from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


class FormationValidator:
    """
    Validator for FPL team formations and constraints.
    """
    
    def __init__(self):
        """Initialize formation validator."""
        self.config = get_config()
        
        # FPL constraints
        self.position_limits = {
            'GK': (2, 2),    # Min 2, Max 2
            'DEF': (5, 5),   # Min 5, Max 5
            'MID': (5, 5),   # Min 5, Max 5
            'FWD': (3, 3)    # Min 3, Max 3
        }
        
        self.max_players_per_team = 3
        self.squad_size = 15
        self.starting_xi_size = 11
        
        # Valid formations (DEF-MID-FWD)
        self.valid_formations = [
            (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1),
            (5, 2, 3), (5, 3, 2), (5, 4, 1)
        ]
        
        logger.info("Formation validator initialized")
    
    def validate_squad(
        self,
        squad: List[Dict],
        bootstrap_data: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Validate complete 15-player squad.
        
        Args:
            squad: List of player dictionaries with element_id
            bootstrap_data: FPL bootstrap data for player info
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(squad) != self.squad_size:
            return False, f"Squad must have exactly {self.squad_size} players (got {len(squad)})"
        
        # Get player information
        if bootstrap_data:
            players_info = self._get_players_info(squad, bootstrap_data)
        else:
            # Assume squad already has necessary info
            players_info = squad
        
        # Check position limits
        position_valid, position_error = self._validate_positions(players_info)
        if not position_valid:
            return False, position_error
        
        # Check team limits
        team_valid, team_error = self._validate_team_limits(players_info)
        if not team_valid:
            return False, team_error
        
        return True, "Valid squad"
    
    def validate_starting_xi(
        self,
        starting_xi: List[Dict],
        bootstrap_data: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Validate starting XI formation.
        
        Args:
            starting_xi: List of 11 players for starting lineup
            bootstrap_data: FPL bootstrap data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(starting_xi) != self.starting_xi_size:
            return False, f"Starting XI must have exactly {self.starting_xi_size} players (got {len(starting_xi)})"
        
        # Get player information
        if bootstrap_data:
            players_info = self._get_players_info(starting_xi, bootstrap_data)
        else:
            players_info = starting_xi
        
        # Check formation validity
        formation_valid, formation_error = self._validate_formation(players_info)
        if not formation_valid:
            return False, formation_error
        
        # Check goalkeeper
        gk_valid, gk_error = self._validate_goalkeeper(players_info)
        if not gk_valid:
            return False, gk_error
        
        return True, "Valid starting XI"
    
    def validate_budget(
        self,
        squad: List[Dict],
        budget: float,
        bootstrap_data: Optional[Dict] = None
    ) -> Tuple[bool, str, float]:
        """
        Validate squad fits within budget.
        
        Args:
            squad: List of player dictionaries
            budget: Available budget in millions
            bootstrap_data: FPL bootstrap data
            
        Returns:
            Tuple of (is_valid, error_message, total_cost)
        """
        total_cost = 0.0
        
        for player in squad:
            # Get player cost
            if 'now_cost' in player:
                cost = player['now_cost'] / 10  # Convert from tenths to millions
            elif bootstrap_data and 'element_id' in player:
                element_id = player['element_id']
                player_data = next(
                    (p for p in bootstrap_data['elements'] if p['id'] == element_id),
                    None
                )
                if player_data:
                    cost = player_data['now_cost'] / 10
                else:
                    return False, f"Player {element_id} not found", 0.0
            else:
                return False, "Player cost information not available", 0.0
            
            total_cost += cost
        
        if total_cost > budget:
            return False, f"Squad cost £{total_cost:.1f}m exceeds budget £{budget:.1f}m", total_cost
        
        return True, f"Squad cost £{total_cost:.1f}m within budget", total_cost
    
    def get_valid_formations(self) -> List[Tuple[int, int, int]]:
        """Get list of valid FPL formations."""
        return self.valid_formations.copy()
    
    def suggest_formation(self, players_info: List[Dict]) -> Optional[Tuple[int, int, int]]:
        """
        Suggest valid formation for given players.
        
        Args:
            players_info: List of player information
            
        Returns:
            Suggested formation tuple (DEF, MID, FWD) or None
        """
        if len(players_info) != self.starting_xi_size:
            return None
        
        # Count positions (excluding GK)
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for player in players_info:
            position = player.get('position', player.get('element_type_name', ''))
            if position in position_counts:
                position_counts[position] += 1
        
        # Check if we have exactly 1 GK
        if position_counts['GK'] != 1:
            return None
        
        # Get formation
        formation = (position_counts['DEF'], position_counts['MID'], position_counts['FWD'])
        
        # Check if formation is valid
        if formation in self.valid_formations:
            return formation
        
        return None
    
    def _get_players_info(
        self,
        squad: List[Dict],
        bootstrap_data: Dict
    ) -> List[Dict]:
        """Get enhanced player information from bootstrap data."""
        players_info = []
        
        elements = {elem['id']: elem for elem in bootstrap_data['elements']}
        teams = {team['id']: team for team in bootstrap_data['teams']}
        positions = {pos['id']: pos for pos in bootstrap_data['element_types']}
        
        for player in squad:
            element_id = player.get('element_id', player.get('id'))
            
            if element_id in elements:
                elem_data = elements[element_id]
                team_data = teams.get(elem_data['team'], {})
                pos_data = positions.get(elem_data['element_type'], {})
                
                player_info = player.copy()
                player_info.update({
                    'element_id': element_id,
                    'web_name': elem_data['web_name'],
                    'position': pos_data.get('singular_name', ''),
                    'team_id': elem_data['team'],
                    'team_name': team_data.get('name', ''),
                    'now_cost': elem_data['now_cost']
                })
                
                players_info.append(player_info)
            else:
                logger.warning(f"Player {element_id} not found in bootstrap data")
        
        return players_info
    
    def _validate_positions(self, players_info: List[Dict]) -> Tuple[bool, str]:
        """Validate position limits."""
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for player in players_info:
            position = player.get('position', '')
            if position in position_counts:
                position_counts[position] += 1
            else:
                return False, f"Unknown position: {position}"
        
        # Check limits
        for position, (min_count, max_count) in self.position_limits.items():
            count = position_counts[position]
            if not (min_count <= count <= max_count):
                return False, f"Invalid {position} count: {count} (must be {min_count}-{max_count})"
        
        return True, "Position limits satisfied"
    
    def _validate_team_limits(self, players_info: List[Dict]) -> Tuple[bool, str]:
        """Validate team limits."""
        team_counts = {}
        
        for player in players_info:
            team_id = player.get('team_id')
            team_name = player.get('team_name', f'Team {team_id}')
            
            if team_id:
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
        
        # Check limits
        for team_id, count in team_counts.items():
            if count > self.max_players_per_team:
                team_name = next(
                    (p['team_name'] for p in players_info if p.get('team_id') == team_id),
                    f'Team {team_id}'
                )
                return False, f"Too many players from {team_name}: {count} (max {self.max_players_per_team})"
        
        return True, "Team limits satisfied"
    
    def _validate_formation(self, players_info: List[Dict]) -> Tuple[bool, str]:
        """Validate starting XI formation."""
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for player in players_info:
            position = player.get('position', '')
            if position in position_counts:
                position_counts[position] += 1
        
        # Must have exactly 1 GK
        if position_counts['GK'] != 1:
            return False, f"Must have exactly 1 goalkeeper (got {position_counts['GK']})"
        
        # Check formation
        formation = (position_counts['DEF'], position_counts['MID'], position_counts['FWD'])
        
        if formation not in self.valid_formations:
            return False, f"Invalid formation {formation}. Valid: {self.valid_formations}"
        
        return True, f"Valid formation: {formation}"
    
    def _validate_goalkeeper(self, players_info: List[Dict]) -> Tuple[bool, str]:
        """Validate goalkeeper selection."""
        goalkeepers = [p for p in players_info if p.get('position') == 'GK']
        
        if len(goalkeepers) != 1:
            return False, f"Must have exactly 1 goalkeeper (got {len(goalkeepers)})"
        
        return True, "Goalkeeper validation passed"
    
    def get_position_requirements(self, formation: Tuple[int, int, int]) -> Dict[str, int]:
        """
        Get position requirements for a formation.
        
        Args:
            formation: Tuple of (DEF, MID, FWD)
            
        Returns:
            Dictionary of position requirements
        """
        def_count, mid_count, fwd_count = formation
        
        return {
            'GK': 1,
            'DEF': def_count,
            'MID': mid_count,
            'FWD': fwd_count
        }
    
    def optimize_formation_from_players(
        self,
        available_players: List[Dict],
        target_formation: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[List[Dict], Tuple[int, int, int]]:
        """
        Select best XI from available players for a target formation.
        
        Args:
            available_players: List of available players with scores
            target_formation: Preferred formation, or None for automatic
            
        Returns:
            Tuple of (selected_xi, formation_used)
        """
        if not available_players:
            return [], (0, 0, 0)
        
        # Group players by position
        by_position = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
        
        for player in available_players:
            position = player.get('position', '')
            if position in by_position:
                by_position[position].append(player)
        
        # Sort by score (assuming 'score' field exists)
        for position in by_position:
            by_position[position].sort(
                key=lambda x: x.get('score', x.get('proj_points', 0)),
                reverse=True
            )
        
        # Try formations in order of preference
        formations_to_try = [target_formation] if target_formation else self.valid_formations
        
        for formation in formations_to_try:
            if formation is None:
                continue
                
            requirements = self.get_position_requirements(formation)
            
            # Check if we have enough players for this formation
            can_form = all(
                len(by_position[pos]) >= req_count
                for pos, req_count in requirements.items()
            )
            
            if can_form:
                # Select players for this formation
                selected_xi = []
                
                for position, req_count in requirements.items():
                    selected_xi.extend(by_position[position][:req_count])
                
                return selected_xi, formation
        
        # If no formation works, return empty
        logger.warning("Could not form valid XI from available players")
        return [], (0, 0, 0)
    
    def calculate_formation_flexibility(
        self,
        squad: List[Dict]
    ) -> Dict[Tuple[int, int, int], int]:
        """
        Calculate how many valid XIs can be formed for each formation.
        
        Args:
            squad: 15-player squad
            
        Returns:
            Dictionary mapping formations to number of possible XIs
        """
        if len(squad) != 15:
            return {}
        
        # Group players by position
        by_position = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
        
        for player in squad:
            position = player.get('position', '')
            if position in by_position:
                by_position[position].append(player)
        
        flexibility = {}
        
        for formation in self.valid_formations:
            requirements = self.get_position_requirements(formation)
            
            # Check if formation is possible
            possible = all(
                len(by_position[pos]) >= req_count
                for pos, req_count in requirements.items()
            )
            
            if possible:
                # Calculate number of combinations
                from math import comb
                
                combinations = 1
                for position, req_count in requirements.items():
                    available = len(by_position[position])
                    if available >= req_count:
                        combinations *= comb(available, req_count)
                
                flexibility[formation] = combinations
            else:
                flexibility[formation] = 0
        
        return flexibility
