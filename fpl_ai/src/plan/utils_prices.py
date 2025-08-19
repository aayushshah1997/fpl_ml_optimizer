"""
FPL pricing utilities for transfer planning.

Handles player price calculations including purchase prices, selling prices,
and FPL's selling value rules for transfer planning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


def fpl_sell_value(purchase_price_tenths: float, current_price_tenths: float) -> float:
    """
    Calculate FPL selling value based on purchase and current price.
    
    FPL Rule: Selling price = purchase_price + floor(max(0, price_rise) / 2)
    
    Args:
        purchase_price_tenths: Price when player was purchased (in tenths of millions)
        current_price_tenths: Current market price (in tenths of millions)
        
    Returns:
        Selling price in tenths of millions
    """
    price_rise = current_price_tenths - purchase_price_tenths
    profit = max(0, price_rise) / 2
    selling_price = purchase_price_tenths + np.floor(profit)
    
    return selling_price


def calculate_team_value(squad: List[Dict], include_bank: bool = True) -> Dict[str, float]:
    """
    Calculate total team value including selling prices.
    
    Args:
        squad: List of players with purchase/current prices
        include_bank: Whether to include bank balance
        
    Returns:
        Dictionary with team value breakdown
    """
    total_selling_value = 0
    total_purchase_value = 0
    total_current_value = 0
    bank = 0
    
    for player in squad:
        purchase_price = player.get('purchase_price', player.get('now_cost', 0))
        current_price = player.get('now_cost', purchase_price)
        
        if isinstance(purchase_price, str):
            purchase_price = float(purchase_price)
        if isinstance(current_price, str):
            current_price = float(current_price)
        
        # Convert to tenths if needed
        if purchase_price < 20:  # Likely in millions
            purchase_price *= 10
        if current_price < 20:  # Likely in millions
            current_price *= 10
        
        selling_price = fpl_sell_value(purchase_price, current_price)
        
        total_purchase_value += purchase_price
        total_current_value += current_price
        total_selling_value += selling_price
        
        # Update player with selling price
        player['selling_price'] = selling_price / 10  # Convert back to millions
    
    if include_bank:
        bank = squad[0].get('bank', 0) if squad else 0
    
    return {
        'total_selling_value': total_selling_value / 10,  # Convert to millions
        'total_purchase_value': total_purchase_value / 10,
        'total_current_value': total_current_value / 10,
        'available_budget': (total_selling_value / 10) + bank,
        'bank': bank,
        'profit_loss': (total_selling_value - total_purchase_value) / 10
    }


def can_afford_transfer(
    squad: List[Dict],
    transfer_out_id: int,
    transfer_in_cost: float,
    free_transfers: int = 0,
    bank: float = 0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a transfer is affordable.
    
    Args:
        squad: Current squad
        transfer_out_id: Player ID to transfer out
        transfer_in_cost: Cost of player to transfer in (in millions)
        free_transfers: Number of free transfers available
        bank: Current bank balance
        
    Returns:
        Tuple of (can_afford, details_dict)
    """
    # Find player to transfer out
    transfer_out_player = None
    for player in squad:
        if player.get('element_id') == transfer_out_id:
            transfer_out_player = player
            break
    
    if not transfer_out_player:
        return False, {'error': 'Player to transfer out not found in squad'}
    
    # Calculate selling price
    purchase_price = transfer_out_player.get('purchase_price', transfer_out_player.get('now_cost', 0))
    current_price = transfer_out_player.get('now_cost', purchase_price)
    
    # Convert to tenths for calculation
    if purchase_price < 20:
        purchase_price *= 10
    if current_price < 20:
        current_price *= 10
    if transfer_in_cost < 20:
        transfer_in_cost_tenths = transfer_in_cost * 10
    else:
        transfer_in_cost_tenths = transfer_in_cost
    
    selling_price_tenths = fpl_sell_value(purchase_price, current_price)
    selling_price = selling_price_tenths / 10
    
    # Calculate cost
    transfer_cost = transfer_in_cost - selling_price
    hit_cost = 0 if free_transfers > 0 else 4  # 4 points for a hit
    
    # Check affordability
    available_funds = bank + selling_price
    can_afford = available_funds >= transfer_in_cost
    
    details = {
        'transfer_out_player': transfer_out_player.get('web_name', ''),
        'transfer_out_selling_price': selling_price,
        'transfer_in_cost': transfer_in_cost,
        'net_cost': transfer_cost,
        'available_funds': available_funds,
        'hit_cost': hit_cost,
        'total_cost_points': hit_cost,
        'requires_hit': free_transfers == 0,
        'can_afford': can_afford
    }
    
    return can_afford, details


def calculate_transfer_budget(
    squad: List[Dict],
    num_transfers: int,
    transfer_out_ids: List[int],
    bank: float = 0
) -> Dict[str, float]:
    """
    Calculate available budget for multiple transfers.
    
    Args:
        squad: Current squad
        num_transfers: Number of transfers to make
        transfer_out_ids: List of player IDs to transfer out
        bank: Current bank balance
        
    Returns:
        Budget calculation details
    """
    if len(transfer_out_ids) != num_transfers:
        logger.warning(f"Mismatch: {num_transfers} transfers but {len(transfer_out_ids)} players specified")
    
    total_selling_value = 0
    transfer_out_details = []
    
    for player_id in transfer_out_ids:
        # Find player in squad
        player = None
        for p in squad:
            if p.get('element_id') == player_id:
                player = p
                break
        
        if player:
            purchase_price = player.get('purchase_price', player.get('now_cost', 0))
            current_price = player.get('now_cost', purchase_price)
            
            # Convert to tenths
            if purchase_price < 20:
                purchase_price *= 10
            if current_price < 20:
                current_price *= 10
            
            selling_price_tenths = fpl_sell_value(purchase_price, current_price)
            selling_price = selling_price_tenths / 10
            
            total_selling_value += selling_price
            transfer_out_details.append({
                'element_id': player_id,
                'name': player.get('web_name', ''),
                'purchase_price': purchase_price / 10,
                'current_price': current_price / 10,
                'selling_price': selling_price
            })
        else:
            logger.warning(f"Player {player_id} not found in squad")
    
    available_budget = bank + total_selling_value
    
    return {
        'available_budget': available_budget,
        'bank': bank,
        'total_selling_value': total_selling_value,
        'transfer_out_details': transfer_out_details,
        'budget_per_player': available_budget / num_transfers if num_transfers > 0 else 0
    }


def optimize_transfer_value(
    squad: List[Dict],
    transfer_candidates: pd.DataFrame,
    num_transfers: int = 1,
    max_cost_per_transfer: Optional[float] = None,
    bank: float = 0
) -> List[Dict[str, Any]]:
    """
    Find optimal transfers based on value (points per cost).
    
    Args:
        squad: Current squad
        transfer_candidates: DataFrame of potential transfers
        num_transfers: Number of transfers to optimize
        max_cost_per_transfer: Maximum cost per transfer
        bank: Available bank balance
        
    Returns:
        List of optimal transfer combinations
    """
    if transfer_candidates.empty:
        return []
    
    # Current squad performance
    squad_df = pd.DataFrame(squad)
    if 'element_id' in squad_df.columns and 'proj_points' in transfer_candidates.columns:
        # Merge to get current projections
        squad_with_proj = squad_df.merge(
            transfer_candidates[['element_id', 'proj_points', 'position']],
            on='element_id',
            how='left'
        )
        squad_with_proj['proj_points'] = squad_with_proj['proj_points'].fillna(2.0)
    else:
        squad_with_proj = squad_df.copy()
        squad_with_proj['proj_points'] = 2.0  # Default
    
    optimal_transfers = []
    
    # Group by position for position-based optimization
    positions = ['GK', 'DEF', 'MID', 'FWD']
    
    for position in positions:
        # Current players in position
        position_players = squad_with_proj[
            squad_with_proj.get('position', '') == position
        ].copy()
        
        if position_players.empty:
            continue
        
        # Transfer candidates in position
        position_candidates = transfer_candidates[
            transfer_candidates['position'] == position
        ].copy()
        
        if position_candidates.empty:
            continue
        
        # Exclude current squad players
        current_ids = set(position_players['element_id'].tolist())
        position_candidates = position_candidates[
            ~position_candidates['element_id'].isin(current_ids)
        ]
        
        # Calculate transfer value for each swap
        for _, current_player in position_players.iterrows():
            current_proj = current_player.get('proj_points', 2.0)
            
            for _, candidate in position_candidates.iterrows():
                candidate_cost = candidate.get('now_cost', 5.0)
                if candidate_cost > 20:  # Convert from tenths
                    candidate_cost /= 10
                
                # Check if transfer is affordable
                affordable, details = can_afford_transfer(
                    squad, 
                    current_player['element_id'],
                    candidate_cost,
                    free_transfers=1,  # Assume FT available
                    bank=bank
                )
                
                if affordable:
                    # Calculate transfer value
                    points_gain = candidate['proj_points'] - current_proj
                    net_cost = details['net_cost']
                    
                    # Value metric: points gained per £M spent (if spending)
                    if net_cost > 0:
                        value_ratio = points_gain / net_cost
                    else:
                        value_ratio = points_gain * 10  # Bonus for gaining money
                    
                    # Apply cost constraint
                    if max_cost_per_transfer and net_cost > max_cost_per_transfer:
                        continue
                    
                    transfer_option = {
                        'transfer_out': {
                            'element_id': current_player['element_id'],
                            'name': current_player.get('web_name', ''),
                            'position': position,
                            'proj_points': current_proj,
                            'selling_price': details['transfer_out_selling_price']
                        },
                        'transfer_in': {
                            'element_id': candidate['element_id'],
                            'name': candidate.get('web_name', ''),
                            'position': position,
                            'proj_points': candidate['proj_points'],
                            'cost': candidate_cost
                        },
                        'points_gain': points_gain,
                        'net_cost': net_cost,
                        'value_ratio': value_ratio,
                        'affordable': affordable,
                        'details': details
                    }
                    
                    optimal_transfers.append(transfer_option)
    
    # Sort by value ratio (best value first)
    optimal_transfers.sort(key=lambda x: x['value_ratio'], reverse=True)
    
    # Return top options
    return optimal_transfers[:num_transfers * 3]  # Top 3 options per transfer slot


def validate_squad_budget(squad: List[Dict], max_budget: float = 100.0) -> Dict[str, Any]:
    """
    Validate that squad is within budget constraints.
    
    Args:
        squad: Squad to validate
        max_budget: Maximum allowed budget
        
    Returns:
        Validation results
    """
    if not squad:
        return {'valid': False, 'error': 'Empty squad'}
    
    total_cost = 0
    cost_breakdown = {}
    
    for player in squad:
        cost = player.get('now_cost', 0)
        position = player.get('position', 'UNKNOWN')
        
        # Convert to millions if needed
        if cost > 20:
            cost /= 10
        
        total_cost += cost
        
        if position not in cost_breakdown:
            cost_breakdown[position] = {'count': 0, 'total_cost': 0}
        
        cost_breakdown[position]['count'] += 1
        cost_breakdown[position]['total_cost'] += cost
    
    is_valid = total_cost <= max_budget
    
    return {
        'valid': is_valid,
        'total_cost': total_cost,
        'max_budget': max_budget,
        'remaining_budget': max_budget - total_cost,
        'cost_breakdown': cost_breakdown,
        'average_cost': total_cost / len(squad) if squad else 0,
        'error': None if is_valid else f"Squad cost £{total_cost:.1f}M exceeds budget £{max_budget:.1f}M"
    }


def estimate_price_changes(
    player_data: pd.DataFrame,
    gameweeks_ahead: int = 5
) -> pd.DataFrame:
    """
    Estimate future player price changes based on ownership and performance.
    
    Args:
        player_data: DataFrame with player data
        gameweeks_ahead: Number of gameweeks to predict ahead
        
    Returns:
        DataFrame with price change predictions
    """
    if player_data.empty:
        return pd.DataFrame()
    
    predictions = player_data.copy()
    
    # Simple price change model based on:
    # 1. Ownership changes
    # 2. Recent performance
    # 3. Projected performance
    
    # Calculate price change probability
    if 'selected_by_percent' in predictions.columns:
        ownership = predictions['selected_by_percent']
        
        # High ownership players more likely to drop if performing poorly
        # Low ownership players more likely to rise if performing well
        
        recent_form = predictions.get('r3_points_per_game', predictions.get('proj_points', 3.0))
        
        # Price rise probability (simplified)
        rise_prob = np.clip(
            (recent_form - 3.0) * 0.2 + (20 - ownership) * 0.01,
            0, 0.3
        )
        
        # Price drop probability
        drop_prob = np.clip(
            (3.0 - recent_form) * 0.15 + (ownership - 10) * 0.005,
            0, 0.2
        )
        
        # Expected price change over period
        expected_rises = rise_prob * gameweeks_ahead * 0.3  # Max 1 rise per ~3 GWs
        expected_drops = drop_prob * gameweeks_ahead * 0.2  # Max 1 drop per ~5 GWs
        
        predictions['price_rise_prob'] = rise_prob
        predictions['price_drop_prob'] = drop_prob
        predictions['expected_price_change'] = (expected_rises - expected_drops) * 0.1  # £0.1M changes
        predictions['predicted_price'] = predictions['now_cost'] + predictions['expected_price_change']
    else:
        # No ownership data - assume minimal price changes
        predictions['price_rise_prob'] = 0.05
        predictions['price_drop_prob'] = 0.05
        predictions['expected_price_change'] = 0.0
        predictions['predicted_price'] = predictions['now_cost']
    
    return predictions


def calculate_transfer_roi(
    transfer_details: Dict[str, Any],
    gameweeks_held: int = 5
) -> Dict[str, float]:
    """
    Calculate return on investment for a transfer.
    
    Args:
        transfer_details: Transfer details from optimize_transfer_value
        gameweeks_held: Number of gameweeks to hold the new player
        
    Returns:
        ROI calculation
    """
    points_gain = transfer_details.get('points_gain', 0)
    net_cost = transfer_details.get('net_cost', 0)
    
    # Total points gain over period
    total_points_gain = points_gain * gameweeks_held
    
    # ROI calculations
    if net_cost > 0:
        roi_points_per_million = total_points_gain / net_cost
        payback_gameweeks = net_cost / points_gain if points_gain > 0 else float('inf')
    else:
        roi_points_per_million = float('inf')  # Gaining money and points
        payback_gameweeks = 0
    
    # Hit-adjusted ROI (if transfer requires a hit)
    hit_cost = transfer_details.get('details', {}).get('hit_cost', 0)
    hit_adjusted_gain = total_points_gain - hit_cost
    
    return {
        'total_points_gain': total_points_gain,
        'roi_points_per_million': roi_points_per_million,
        'payback_gameweeks': payback_gameweeks,
        'hit_adjusted_gain': hit_adjusted_gain,
        'hit_cost': hit_cost,
        'net_cost': net_cost,
        'worth_hit': hit_adjusted_gain > 0
    }
