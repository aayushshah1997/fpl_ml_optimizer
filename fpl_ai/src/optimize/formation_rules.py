"""
Unified formation validation rules - single source of truth for FPL team constraints.

Consolidates all formation validation logic to eliminate duplication across the codebase.
"""

from typing import Dict, List, Tuple, Any, Optional
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


class FormationRules:
    """
    Unified formation and team constraint validator.
    """
    
    def __init__(self):
        """Initialize formation rules with FPL constraints."""
        self.config = get_config()
        
        # FPL position constraints
        self.position_limits = {
            'GK': (2, 2),    # Min 2, Max 2
            'DEF': (5, 5),   # Min 5, Max 5
            'MID': (5, 5),   # Min 5, Max 5
            'FWD': (3, 3)    # Min 3, Max 3
        }
        
        self.max_players_per_team = 3
        self.squad_size = 15
        self.starting_xi_size = 11
        
        # Valid FPL formations (DEF-MID-FWD) - Prioritizing 3-4 defense for better FPL scoring
        self.valid_formations = [
            (3, 5, 2),  # 3-5-2 - Best for FPL (3 DEF, 5 MID, 2 FWD)
            (3, 4, 3),  # 3-4-3 - Strong 3 DEF formation
            (4, 4, 2),  # 4-4-2 - Classic balanced formation
            (4, 3, 3),  # 4-3-3 - 4 DEF with 3 FWD
            (4, 5, 1),  # 4-5-1 - 4 DEF with 5 MID
            (5, 3, 2),  # 5-3-2 - Avoid if possible (5 DEF)
            (5, 4, 1)   # 5-4-1 - Avoid if possible (5 DEF)
        ]
        
        logger.info("Formation rules initialized")
    
    def validate_squad(
        self,
        squad: List[Dict],
        bootstrap_data: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Validate squad meets FPL constraints.
        
        Args:
            squad: List of player dictionaries
            bootstrap_data: Optional FPL bootstrap data for additional validation
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not squad:
            return False, "Empty squad"
        
        if len(squad) != self.squad_size:
            return False, f"Squad size {len(squad)} != {self.squad_size}"
        
        # Get player positions and teams
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        team_counts = {}
        
        for player in squad:
            # Handle different data formats
            if isinstance(player, dict):
                position = player.get('position', '')
                team_name = player.get('team_name', 'Unknown')
                element_id = player.get('element_id', 0)
            else:
                # Handle DataFrame row format
                position = getattr(player, 'position', '')
                team_name = getattr(player, 'team_name', 'Unknown')
                element_id = getattr(player, 'element_id', 0)
            
            if not position:
                return False, f"Player {element_id} missing position"
            
            if position not in position_counts:
                return False, f"Invalid position '{position}' for player {element_id}"
            
            position_counts[position] += 1
            team_counts[team_name] = team_counts.get(team_name, 0) + 1
        
        # Check position limits
        for position, (min_count, max_count) in self.position_limits.items():
            count = position_counts[position]
            if not (min_count <= count <= max_count):
                return False, f"Invalid {position} count: {count} (must be {min_count}-{max_count})"
        
        # Check team limits
        for team_name, count in team_counts.items():
            if count > self.max_players_per_team:
                return False, f"Too many players from {team_name}: {count} (max {self.max_players_per_team})"
        
        # Additional validation with bootstrap data if provided
        if bootstrap_data:
            return self._validate_with_bootstrap(squad, bootstrap_data)
        
        return True, "Valid squad"
    
    def validate_starting_xi(
        self,
        starting_xi: List[Dict],
        squad: Optional[List[Dict]] = None
    ) -> Tuple[bool, str]:
        """
        Validate starting XI meets FPL formation constraints.
        
        Args:
            starting_xi: List of starting XI players
            squad: Optional full squad for validation
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not starting_xi:
            return False, "Empty starting XI"
        
        if len(starting_xi) != self.starting_xi_size:
            return False, f"Starting XI size {len(starting_xi)} != {self.starting_xi_size}"
        
        # Count positions
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        
        for player in starting_xi:
            position = player.get('position', '')
            if position not in position_counts:
                return False, f"Invalid position '{position}' in starting XI"
            position_counts[position] += 1
        
        # Check goalkeeper count
        if position_counts['GK'] != 1:
            return False, f"Starting XI must have exactly 1 GK, got {position_counts['GK']}"
        
        # Check formation validity
        formation = (position_counts['DEF'], position_counts['MID'], position_counts['FWD'])
        if formation not in self.valid_formations:
            return False, f"Invalid formation {formation[0]}-{formation[1]}-{formation[2]}"
        
        # Validate all players are in squad if provided
        if squad:
            squad_ids = {p.get('element_id') for p in squad}
            xi_ids = {p.get('element_id') for p in starting_xi}
            
            if not xi_ids.issubset(squad_ids):
                missing = xi_ids - squad_ids
                return False, f"Starting XI contains players not in squad: {missing}"
        
        return True, "Valid starting XI"
    
    def validate_budget(
        self,
        squad: List[Dict],
        max_budget: float = 100.0
    ) -> Tuple[bool, str]:
        """
        Validate squad is within budget constraints.
        
        Args:
            squad: List of player dictionaries
            max_budget: Maximum allowed budget
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not squad:
            return False, "Empty squad"
        
        total_cost = 0.0
        for player in squad:
            cost = player.get('now_cost', 0.0)
            if isinstance(cost, str):
                try:
                    cost = float(cost)
                except ValueError:
                    return False, f"Invalid cost format: {cost}"
            total_cost += cost
        
        if total_cost > max_budget:
            return False, f"Squad cost £{total_cost:.1f}M exceeds budget £{max_budget:.1f}M"
        
        return True, f"Within budget: £{total_cost:.1f}M / £{max_budget:.1f}M"
    
    def _validate_with_bootstrap(
        self,
        squad: List[Dict],
        bootstrap_data: Dict
    ) -> Tuple[bool, str]:
        """
        Additional validation using FPL bootstrap data.
        
        Args:
            squad: List of player dictionaries
            bootstrap_data: FPL bootstrap data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if 'elements' not in bootstrap_data:
            return True, "No bootstrap elements data for validation"
        
        # Create player lookup
        elements = {elem['id']: elem for elem in bootstrap_data['elements']}
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        for player in squad:
            element_id = player.get('element_id')
            if element_id not in elements:
                return False, f"Player {element_id} not found in FPL data"
            
            element = elements[element_id]
            expected_position = position_map.get(element['element_type'], 'UNKNOWN')
            actual_position = player.get('position', '')
            
            if expected_position != actual_position:
                return False, f"Position mismatch for player {element_id}: expected {expected_position}, got {actual_position}"
        
        return True, "Validated with bootstrap data"
    
    def get_formation_requirements(self, formation: Tuple[int, int, int]) -> Dict[str, int]:
        """
        Get position requirements for a specific formation.
        
        Args:
            formation: Tuple of (defenders, midfielders, forwards)
            
        Returns:
            Dictionary mapping positions to required counts
        """
        def_count, mid_count, fwd_count = formation
        
        return {
            'GK': 1,
            'DEF': def_count,
            'MID': mid_count,
            'FWD': fwd_count
        }


# Convenience functions for backward compatibility
def validate_squad_constraints(
    squad: List[Dict],
    bootstrap_data: Optional[Dict] = None
) -> Tuple[bool, str]:
    """
    Validate squad meets FPL constraints.
    
    Args:
        squad: List of player dictionaries
        bootstrap_data: Optional FPL bootstrap data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    rules = FormationRules()
    return rules.validate_squad(squad, bootstrap_data)


def validate_formation(
    starting_xi: List[Dict],
    squad: Optional[List[Dict]] = None
) -> Tuple[bool, str]:
    """
    Validate starting XI formation.
    
    Args:
        starting_xi: List of starting XI players
        squad: Optional full squad
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    rules = FormationRules()
    return rules.validate_starting_xi(starting_xi, squad)


def validate_budget_constraints(
    squad: List[Dict],
    max_budget: float = 100.0
) -> Tuple[bool, str]:
    """
    Validate squad budget constraints.
    
    Args:
        squad: List of player dictionaries
        max_budget: Maximum budget
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    rules = FormationRules()
    return rules.validate_budget(squad, max_budget)
