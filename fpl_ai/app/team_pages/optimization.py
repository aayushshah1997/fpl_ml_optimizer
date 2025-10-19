"""
Team optimization logic for the FPL AI dashboard.

Handles team selection, optimization, and related calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from ..data_loaders import load_predictions_cached, enrich_with_current_fpl_data


def optimize_team_selection(predictions_df: pd.DataFrame, objective: str = "mean") -> Dict[str, Any]:
    """Optimize team selection using the optimizer."""
    try:
        from fpl_ai.src.optimize.optimizer import TeamOptimizer
        
        # Initialize optimizer
        optimizer = TeamOptimizer()
        
        # Prepare data for optimization
        opt_data = predictions_df.copy()
        
        # Ensure required columns exist
        required_cols = ['element_id', 'position', 'now_cost', 'team_name', 'web_name']
        for col in required_cols:
            if col not in opt_data.columns:
                st.error(f"Missing required column: {col}")
                return {"error": f"Missing column: {col}"}
        
        # Set the score column based on objective
        if objective == "mean":
            score_col = "mean_points" if "mean_points" in opt_data.columns else "proj_points"
        elif objective == "median":
            score_col = "median_points" if "median_points" in opt_data.columns else "proj_points"
        else:
            score_col = "proj_points"
        
        if score_col not in opt_data.columns:
            st.error(f"Missing score column: {score_col}")
            return {"error": f"Missing score column: {score_col}"}
        
        # Build squad
        squad = optimizer._greedy_selection(opt_data, budget=100.0, formation_preference=None, score_col=score_col)
        
        if not squad:
            return {"error": "Failed to build squad"}
        
        # Optimize starting XI
        starting_xi_result = optimizer.optimize_starting_xi(squad, opt_data)
        
        return {
            "squad": squad,
            "starting_xi": starting_xi_result.get("starting_xi", []),
            "formation": starting_xi_result.get("formation", (3, 4, 3)),
            "expected_points": starting_xi_result.get("expected_points", 0),
            "total_cost": sum(player.get("cost", 0) for player in squad),
            "bank": 100.0 - sum(player.get("cost", 0) for player in squad)
        }
        
    except Exception as e:
        import streamlit as st
        st.error(f"Team optimization failed: {e}")
        return {"error": str(e)}


def create_simple_team(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Create a budget-optimized team selection using unified team builder."""
    try:
        from fpl_ai.src.optimize.team_builder import build_squad_from_predictions
        from fpl_ai.src.optimize.formations import FormationValidator
        
        # Use unified team builder
        team_data = build_squad_from_predictions(predictions_df, budget=100.0)
        
        if not team_data:
            return {"error": "Failed to build team"}
        
        # Create formation validator
        validator = FormationValidator()
        
        # Optimize formation from the selected squad
        starting_xi, formation = validator.optimize_formation_from_players(team_data)
        
        # Calculate expected points for starting XI
        expected_points = 0
        if starting_xi:
            # Get predictions for starting XI players
            xi_ids = [player['element_id'] for player in starting_xi]
            xi_predictions = predictions_df[predictions_df['element_id'].isin(xi_ids)]
            
            if not xi_predictions.empty:
                # Use mean_points if available, otherwise proj_points
                points_col = 'mean_points' if 'mean_points' in xi_predictions.columns else 'proj_points'
                expected_points = xi_predictions[points_col].sum()
        
        # Calculate total cost and bank
        total_cost = sum(player.get('cost', 0) for player in team_data)
        bank = max(0, 100.0 - total_cost)
        
        return {
            "squad": team_data,
            "starting_xi": starting_xi,
            "formation": formation,
            "expected_points": expected_points,
            "total_cost": total_cost,
            "bank": bank
        }
        
    except Exception as e:
        import streamlit as st
        st.error(f"Simple team creation failed: {e}")
        return {"error": str(e)}
