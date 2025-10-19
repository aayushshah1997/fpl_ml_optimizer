"""
Transfer suggestion logic for the FPL AI dashboard.

Handles transfer recommendations and budget calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


def suggest_transfers(
    user_team: List[Dict],
    predictions_df: pd.DataFrame,
    optimized: Dict[str, Any]
) -> Dict[str, Any]:
    """Suggest simple action based on user's previous GW team.

    - Computes bank from user squad costs (assumes £100m total budget)
    - Finds best single transfer upgrade within budget for same-position swap
    - If no meaningful gain, recommend saving transfer
    """
    if not user_team or predictions_df is None or predictions_df.empty:
        return {"action": "No data", "details": []}

    # Map costs and projections for user players
    pred_index = predictions_df.set_index('element_id') if 'element_id' in predictions_df.columns else None

    def get_cost(pid: int) -> float:
        if pred_index is not None and pid in pred_index.index:
            c = float(pred_index.loc[pid].get('now_cost', 5.0))
            return c/10.0 if c > 20 else c
        return 5.0

    def get_proj(pid: int) -> float:
        if pred_index is not None and pid in pred_index.index:
            return float(pred_index.loc[pid].get('proj_points', pred_index.loc[pid].get('mean_points', 0.0)))
        return 0.0

    def get_pos(pid: int, fallback: str = 'MID') -> str:
        if pred_index is not None and pid in pred_index.index:
            return str(pred_index.loc[pid].get('position', fallback))
        return fallback

    user_ids = [p.get('element_id') for p in user_team if p.get('element_id') is not None]
    user_cost = sum(get_cost(pid) for pid in user_ids)
    bank = max(0.0, 100.0 - user_cost)

    # Build candidate pool not already in team
    candidates = predictions_df[~predictions_df['element_id'].isin(user_ids)].copy()
    if candidates.empty:
        return {"action": "Save transfer", "details": ["No viable replacements found."]}

    # Normalize costs to £M
    cand_cost = candidates['now_cost'].copy()
    if cand_cost.max() > 20:
        candidates['now_cost'] = candidates['now_cost'] / 10.0

    best = {"gain": -1e9}
    for p in user_ids:
        pos = get_pos(p)
        p_cost = get_cost(p)
        p_proj = get_proj(p)
        # Same-position candidates within budget
        pool = candidates[candidates.get('position') == pos]
        affordable = pool[pool['now_cost'] <= (p_cost + bank)]
        if affordable.empty:
            continue
        # Best upgrade by projected gain
        top = affordable.sort_values('proj_points', ascending=False).head(1)
        if not top.empty:
            row = top.iloc[0]
            gain = float(row.get('proj_points', 0.0)) - p_proj
            if gain > best.get('gain', -1e9):
                best = {
                    "out_id": p,
                    "out_proj": p_proj,
                    "out_cost": p_cost,
                    "in_id": int(row['element_id']),
                    "in_name": row.get('web_name', 'Unknown'),
                    "in_team": row.get('team_name', 'Unknown'),
                    "in_pos": pos,
                    "in_cost": float(row['now_cost']),
                    "in_proj": float(row.get('proj_points', 0.0)),
                    "gain": gain
                }

    # Decision threshold: if expected gain < 1.0 point, recommend saving
    if best.get('gain', 0) <= 1.0:
        return {
            "action": "Save transfer",
            "details": [
                f"Bank: £{bank:.1f}m | Best 1-FT gain ≤ 1.0 pts (gain {best.get('gain', 0):.2f})"
            ]
        }

    # Otherwise, suggest the best single transfer
    return {
        "action": "1 transfer",
        "details": [
            f"OUT (cost £{best['out_cost']:.1f}m): id {best['out_id']} → IN {best['in_name']} ({best['in_team']}, {best['in_pos']})",
            f"IN cost £{best['in_cost']:.1f}m | Gain ≈ {best['gain']:.2f} pts",
            f"Bank before: £{bank:.1f}m | After: £{bank - (best['in_cost'] - best['out_cost']):.1f}m"
        ]
    }


def calculate_transfer_value(
    player_out: Dict,
    player_in: Dict,
    predictions_df: pd.DataFrame
) -> Dict[str, Any]:
    """Calculate the value of a specific transfer."""
    try:
        # Get projections for both players
        out_proj = predictions_df[
            predictions_df['element_id'] == player_out['element_id']
        ]['proj_points'].iloc[0] if not predictions_df.empty else 0
        
        in_proj = predictions_df[
            predictions_df['element_id'] == player_in['element_id']
        ]['proj_points'].iloc[0] if not predictions_df.empty else 0
        
        # Calculate net gain
        points_gain = in_proj - out_proj
        cost_change = player_in.get('cost', 0) - player_out.get('cost', 0)
        
        return {
            "points_gain": points_gain,
            "cost_change": cost_change,
            "value_per_million": points_gain / max(abs(cost_change), 0.1) if cost_change != 0 else 0,
            "out_projection": out_proj,
            "in_projection": in_proj
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "points_gain": 0,
            "cost_change": 0,
            "value_per_million": 0,
            "out_projection": 0,
            "in_projection": 0
        }
