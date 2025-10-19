"""
Team comparison logic for the FPL AI dashboard.

Handles comparison between user team and optimized team.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


def compare_teams(
    user_team: List[Dict],
    optimized_team: Dict[str, Any],
    predictions_df: pd.DataFrame
) -> Dict[str, Any]:
    """Compare user team with optimized team."""
    if not user_team or not optimized_team or predictions_df.empty:
        return {"error": "Missing team data for comparison"}
    
    try:
        # Get user team projections
        user_ids = [p.get('element_id') for p in user_team if p.get('element_id') is not None]
        user_predictions = predictions_df[predictions_df['element_id'].isin(user_ids)]
        
        # Get optimized team projections
        opt_starting_xi = optimized_team.get('starting_xi', [])
        opt_ids = [p.get('element_id') for p in opt_starting_xi if p.get('element_id') is not None]
        opt_predictions = predictions_df[predictions_df['element_id'].isin(opt_ids)]
        
        # Calculate total projections
        user_total = user_predictions['proj_points'].sum() if not user_predictions.empty else 0
        opt_total = opt_predictions['proj_points'].sum() if not opt_predictions.empty else 0
        
        # Calculate costs
        user_cost = sum(p.get('cost', 0) for p in user_team)
        opt_cost = optimized_team.get('total_cost', 0)
        
        # Calculate value metrics
        user_value = user_total / max(user_cost, 0.1)
        opt_value = opt_total / max(opt_cost, 0.1)
        
        # Find differences
        points_difference = opt_total - user_total
        cost_difference = opt_cost - user_cost
        
        return {
            "user_total_points": user_total,
            "optimized_total_points": opt_total,
            "points_difference": points_difference,
            "user_cost": user_cost,
            "optimized_cost": opt_cost,
            "cost_difference": cost_difference,
            "user_value": user_value,
            "optimized_value": opt_value,
            "value_improvement": opt_value - user_value,
            "user_team_size": len(user_team),
            "optimized_team_size": len(opt_starting_xi)
        }
        
    except Exception as e:
        return {"error": str(e)}


def find_team_differences(
    user_team: List[Dict],
    optimized_team: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    """Find specific differences between user and optimized teams."""
    if not user_team or not optimized_team:
        return {"added": [], "removed": [], "unchanged": []}
    
    try:
        user_ids = {p.get('element_id') for p in user_team if p.get('element_id') is not None}
        opt_starting_xi = optimized_team.get('starting_xi', [])
        opt_ids = {p.get('element_id') for p in opt_starting_xi if p.get('element_id') is not None}
        
        # Players added to optimized team
        added_ids = opt_ids - user_ids
        added_players = [p for p in opt_starting_xi if p.get('element_id') in added_ids]
        
        # Players removed from user team
        removed_ids = user_ids - opt_ids
        removed_players = [p for p in user_team if p.get('element_id') in removed_ids]
        
        # Players in both teams
        common_ids = user_ids & opt_ids
        unchanged_players = [p for p in user_team if p.get('element_id') in common_ids]
        
        return {
            "added": added_players,
            "removed": removed_players,
            "unchanged": unchanged_players
        }
        
    except Exception as e:
        return {"added": [], "removed": [], "unchanged": [], "error": str(e)}


def analyze_position_balance(
    user_team: List[Dict],
    optimized_team: Dict[str, Any],
    predictions_df: pd.DataFrame
) -> Dict[str, Any]:
    """Analyze position balance between teams."""
    if not user_team or not optimized_team or predictions_df.empty:
        return {"error": "Missing data for position analysis"}
    
    try:
        positions = ['GK', 'DEF', 'MID', 'FWD']
        user_balance = {pos: 0 for pos in positions}
        opt_balance = {pos: 0 for pos in positions}
        
        # Count user team positions
        for player in user_team:
            pos = player.get('position', 'MID')
            if pos in user_balance:
                user_balance[pos] += 1
        
        # Count optimized team positions
        opt_starting_xi = optimized_team.get('starting_xi', [])
        for player in opt_starting_xi:
            pos = player.get('position', 'MID')
            if pos in opt_balance:
                opt_balance[pos] += 1
        
        # Calculate position strengths
        position_analysis = {}
        for pos in positions:
            user_count = user_balance[pos]
            opt_count = opt_balance[pos]
            
            # Get projections for position
            user_pos_players = [p for p in user_team if p.get('position') == pos]
            opt_pos_players = [p for p in opt_starting_xi if p.get('position') == pos]
            
            user_pos_proj = sum(
                predictions_df[predictions_df['element_id'] == p.get('element_id')]['proj_points'].iloc[0]
                for p in user_pos_players
                if not predictions_df[predictions_df['element_id'] == p.get('element_id')].empty
            )
            
            opt_pos_proj = sum(
                predictions_df[predictions_df['element_id'] == p.get('element_id')]['proj_points'].iloc[0]
                for p in opt_pos_players
                if not predictions_df[predictions_df['element_id'] == p.get('element_id')].empty
            )
            
            position_analysis[pos] = {
                "user_count": user_count,
                "optimized_count": opt_count,
                "user_projection": user_pos_proj,
                "optimized_projection": opt_pos_proj,
                "projection_difference": opt_pos_proj - user_pos_proj
            }
        
        return position_analysis
        
    except Exception as e:
        return {"error": str(e)}
