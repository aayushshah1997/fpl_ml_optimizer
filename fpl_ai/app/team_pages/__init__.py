"""
Team pages modules for the FPL AI dashboard.

Modular components for team prediction, optimization, and display.
"""

from .optimization import optimize_team_selection, create_simple_team
from .transfers import suggest_transfers, calculate_transfer_value
from .captaincy import select_captain_and_vice_captain, calculate_captain_value, get_captain_alternatives, analyze_captain_form
from .comparison import compare_teams, find_team_differences, analyze_position_balance
from .display import (
    display_player_fixtures, display_team_summary, display_formation_grid,
    display_captain_selection, display_transfer_suggestions, create_position_summary
)

__all__ = [
    # Optimization
    'optimize_team_selection',
    'create_simple_team',
    
    # Transfers
    'suggest_transfers',
    'calculate_transfer_value',
    
    # Captaincy
    'select_captain_and_vice_captain',
    'calculate_captain_value',
    'get_captain_alternatives',
    'analyze_captain_form',
    
    # Comparison
    'compare_teams',
    'find_team_differences',
    'analyze_position_balance',
    
    # Display
    'display_player_fixtures',
    'display_team_summary',
    'display_formation_grid',
    'display_captain_selection',
    'display_transfer_suggestions',
    'create_position_summary'
]
