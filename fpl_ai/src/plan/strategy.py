"""
Auto-transfer strategy optimization engine.

Evaluates whether to roll free transfers or make 0/1/2 transfers based on:
- Expected points improvement from team changes
- Hit costs and bank utility for future planning
- Risk-adjusted scoring using Monte Carlo CVaR
- Position constraints and budget limits
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import logging

from ..common.config import load_settings, get_logger
from ..optimize.optimizer import TeamOptimizer
from ..modeling.mc_sim import simulate_player_matrix

logger = get_logger(__name__)


@dataclass
class StrategyDecision:
    """Container for transfer strategy decision and metadata."""
    transfers: List[Tuple[int, int]]  # (out_player_id, in_player_id) pairs
    hits: int                         # Number of hits required
    roll: bool                        # Whether to roll free transfer instead
    captain_id: int                   # Selected captain player_id
    vice_id: int                      # Selected vice-captain player_id
    exp_points: float                 # Expected points for the gameweek
    risk_adj: float                   # Risk-adjusted score (CVaR)
    bank_after: float                 # Remaining bank after transfers
    details: Dict                     # Additional details and reasoning


def bank_utility(bank: float, weight: float) -> float:
    """
    Calculate utility from keeping money in the bank for future upgrades.
    Uses concave utility function so extra bank has diminishing returns.
    
    Args:
        bank: Available bank money
        weight: Weight for bank utility in optimization
        
    Returns:
        Utility value from bank amount
    """
    return weight * np.sqrt(max(bank, 0.0))


def cvar_score(samples: np.ndarray, alpha: float = 0.2) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) for risk assessment.
    
    Args:
        samples: Array of scenario outcomes
        alpha: Confidence level (e.g. 0.2 for 20th percentile tail)
        
    Returns:
        CVaR value (mean of worst alpha% outcomes)
    """
    if samples.size == 0:
        return 0.0
    
    threshold = np.quantile(samples, alpha)
    tail = samples[samples <= threshold]
    return float(tail.mean()) if tail.size > 0 else float(samples.mean())


def evaluate_one_step(
    team_df: pd.DataFrame, 
    market_df: pd.DataFrame, 
    bank: float, 
    free_transfers: int, 
    settings: Dict
) -> StrategyDecision:
    """
    Evaluate optimal transfer strategy for one gameweek.
    
    Considers 0/1/2 same-position swaps within budget and club limits.
    Scoring: risk-adjusted XI points (MC CVaR) - hit costs + bank utility.
    
    Args:
        team_df: Current 15-man squad with projections
        market_df: All available players with projections
        bank: Available money in bank
        free_transfers: Number of free transfers available
        settings: Configuration settings
        
    Returns:
        StrategyDecision with optimal transfers and metadata
    """
    try:
        cfg = settings.get("strategy", {})
        hit_cost = int(cfg.get("hit_cost", 4))
        roll_thresh = float(cfg.get("roll_threshold_points", 2.0))
        bank_weight = float(cfg.get("bank_future_weight", 0.25))
        max_transfers = int(cfg.get("max_transfers_per_gw", 2))
        per_pos = cfg.get("shortlist", {}).get("per_pos", 20)
        
        mc_config = settings.get("mc", {})
        num_scenarios = int(mc_config.get("num_scenarios", 2000))
        seed = int(mc_config.get("seed", 42))
        minutes_uncertainty = float(mc_config.get("minutes_uncertainty", 0.20))
        cvar_alpha = float(mc_config.get("cvar_alpha", 0.2))
        
        logger.debug(f"Evaluating strategy: bank={bank:.1f}, FTs={free_transfers}")
        
        # 0) Baseline: current team with no transfers
        optimizer = TeamOptimizer()
        baseline_result = optimizer.optimize_team(team_df, budget=100.0, objective="mean")
        
        if not baseline_result or not baseline_result.get("squad"):
            logger.warning("Failed to optimize baseline team")
            return StrategyDecision([], 0, True, 0, 0, 0.0, 0.0, bank, {"reason": "baseline_failed"})
        
        # Get starting XI from baseline
        baseline_xi = optimizer.optimize_starting_xi(
            baseline_result["squad"], team_df, objective="mean"
        )
        
        if not baseline_xi or not baseline_xi.get("starting_xi"):
            logger.warning("Failed to get baseline XI")
            return StrategyDecision([], 0, True, 0, 0, 0.0, 0.0, bank, {"reason": "xi_failed"})
        
        # Calculate baseline score using Monte Carlo
        baseline_players = pd.DataFrame(baseline_xi["starting_xi"])
        baseline_scenarios, _ = simulate_player_matrix(
            baseline_players, num_scenarios, seed, minutes_uncertainty, settings
        )
        
        if baseline_scenarios.size == 0:
            logger.warning("Failed to generate baseline scenarios")
            return StrategyDecision([], 0, True, 0, 0, 0.0, 0.0, bank, {"reason": "mc_failed"})
        
        # Simple captain selection for baseline (highest mean)
        baseline_captain_idx = np.argmax(baseline_players.get("proj_points", baseline_players.get("proj", np.zeros(len(baseline_players)))).values)
        baseline_team_scores = baseline_scenarios.sum(axis=1) + baseline_scenarios[:, baseline_captain_idx]
        baseline_cvar = cvar_score(baseline_team_scores, cvar_alpha)
        baseline_mean = float(baseline_team_scores.mean())
        
        # Include bank utility in baseline score
        baseline_score = baseline_cvar + bank_utility(bank, bank_weight)
        
        best_decision = StrategyDecision(
            transfers=[], 
            hits=0, 
            roll=True, 
            captain_id=int(baseline_players.iloc[baseline_captain_idx].get("element_id", 0)),
            vice_id=int(baseline_players.iloc[baseline_captain_idx].get("element_id", 0)),  # Simplified
            exp_points=baseline_mean, 
            risk_adj=baseline_cvar, 
            bank_after=bank,
            details={"reason": "roll", "baseline_score": baseline_score}
        )
        best_score = baseline_score
        
        # Build candidate pools by position for transfers
        candidates = {}
        positions = ["GK", "DEF", "MID", "FWD"]
        
        for pos in positions:
            pos_market = market_df[market_df.get("position") == pos]
            if not pos_market.empty:
                # Get top candidates by projection
                proj_col = "proj_points" if "proj_points" in pos_market.columns else "proj"
                candidates[pos] = pos_market.nlargest(per_pos, proj_col)
            else:
                candidates[pos] = pd.DataFrame()
        
        logger.debug(f"Built candidate pools: {[(pos, len(cands)) for pos, cands in candidates.items()]}")
        
        # Try 1 transfer (same position swaps)
        for _, current_player in team_df.iterrows():
            pos = current_player.get("position")
            current_cost = float(current_player.get("now_cost", current_player.get("cost", 0)))
            current_id = current_player.get("element_id", current_player.get("player_id"))
            
            if pos not in candidates or candidates[pos].empty:
                continue
            
            for _, target_player in candidates[pos].iterrows():
                target_id = target_player.get("element_id", target_player.get("player_id"))
                target_cost = float(target_player.get("now_cost", target_player.get("cost", 0)))
                target_team = target_player.get("team_name", target_player.get("team"))
                
                # Skip if same player or same team (basic constraint)
                if target_id == current_id or target_team == current_player.get("team_name", current_player.get("team")):
                    continue
                
                cost_delta = target_cost - current_cost
                hits_needed = max(0, 1 - free_transfers)
                
                # Budget check
                if cost_delta > bank + hits_needed * hit_cost * 0.1:  # Crude budget gate
                    continue
                
                # Create new team with transfer
                new_team = team_df.copy()
                
                # Update player in team
                mask = (new_team.get("element_id") == current_id) | (new_team.get("player_id") == current_id)
                if mask.any():
                    for col in ["element_id", "player_id", "web_name", "team_name", "position", "now_cost", "cost"]:
                        if col in new_team.columns and col in target_player.index:
                            new_team.loc[mask, col] = target_player[col]
                    
                    # Update projection
                    proj_col = "proj_points" if "proj_points" in new_team.columns else "proj"
                    target_proj_col = "proj_points" if "proj_points" in target_player.index else "proj"
                    if proj_col in new_team.columns and target_proj_col in target_player.index:
                        new_team.loc[mask, proj_col] = target_player[target_proj_col]
                
                # Evaluate new team
                try:
                    new_result = optimizer.optimize_team(new_team, budget=100.0, objective="mean")
                    if not new_result or not new_result.get("squad"):
                        continue
                    
                    new_xi = optimizer.optimize_starting_xi(
                        new_result["squad"], new_team, objective="mean"
                    )
                    if not new_xi or not new_xi.get("starting_xi"):
                        continue
                    
                    new_players = pd.DataFrame(new_xi["starting_xi"])
                    new_scenarios, _ = simulate_player_matrix(
                        new_players, num_scenarios, seed, minutes_uncertainty, settings
                    )
                    
                    if new_scenarios.size == 0:
                        continue
                    
                    # Captain selection (simplified)
                    new_captain_idx = np.argmax(new_players.get("proj_points", new_players.get("proj", np.zeros(len(new_players)))).values)
                    new_team_scores = new_scenarios.sum(axis=1) + new_scenarios[:, new_captain_idx]
                    new_cvar = cvar_score(new_team_scores, cvar_alpha)
                    new_mean = float(new_team_scores.mean())
                    
                    # Calculate score including costs and bank utility
                    new_bank = bank - max(0.0, cost_delta)
                    hit_penalty = hits_needed * hit_cost
                    new_score = new_cvar - hit_penalty + bank_utility(new_bank, bank_weight)
                    
                    if new_score > best_score + 1e-6:  # Small epsilon for numerical stability
                        best_score = new_score
                        best_decision = StrategyDecision(
                            transfers=[(int(current_id), int(target_id))],
                            hits=hits_needed,
                            roll=False,
                            captain_id=int(new_players.iloc[new_captain_idx].get("element_id", 0)),
                            vice_id=int(new_players.iloc[new_captain_idx].get("element_id", 0)),
                            exp_points=new_mean,
                            risk_adj=new_cvar,
                            bank_after=new_bank,
                            details={
                                "reason": "1T_improve",
                                "score_improvement": new_score - baseline_score,
                                "cost_delta": cost_delta
                            }
                        )
                        
                        logger.debug(f"Better 1T found: {current_player.get('web_name', 'Unknown')} -> {target_player.get('web_name', 'Unknown')}, score: {new_score:.2f}")
                
                except Exception as e:
                    logger.debug(f"Error evaluating 1T: {e}")
                    continue
        
        # Try 2 transfers (simplified heuristic for performance)
        if max_transfers >= 2 and free_transfers >= 1:
            # Sample a few positions to avoid combinatorial explosion
            sample_players = team_df.sample(min(5, len(team_df)), random_state=42)
            
            for _, first_out in sample_players.iterrows():
                pos1 = first_out.get("position")
                if pos1 not in candidates or candidates[pos1].empty:
                    continue
                
                # Try first transfer
                for _, first_in in candidates[pos1].head(5).iterrows():  # Limit candidates
                    if first_in.get("element_id") == first_out.get("element_id"):
                        continue
                    
                    # Apply first transfer
                    temp_team = team_df.copy()
                    first_id = first_out.get("element_id", first_out.get("player_id"))
                    mask1 = (temp_team.get("element_id") == first_id) | (temp_team.get("player_id") == first_id)
                    
                    if not mask1.any():
                        continue
                    
                    # Update temp team with first transfer
                    for col in ["element_id", "player_id", "web_name", "team_name", "position", "now_cost", "cost"]:
                        if col in temp_team.columns and col in first_in.index:
                            temp_team.loc[mask1, col] = first_in[col]
                    
                    # Try second transfer from same position
                    remaining_pos1 = temp_team[temp_team.get("position") == pos1]
                    if remaining_pos1.empty:
                        continue
                    
                    second_out = remaining_pos1.nsmallest(1, "proj_points" if "proj_points" in remaining_pos1.columns else "proj").iloc[0]
                    
                    for _, second_in in candidates[pos1].head(5).iterrows():
                        second_in_id = second_in.get("element_id", second_in.get("player_id"))
                        if second_in_id in temp_team.get("element_id", temp_team.get("player_id", [])).values:
                            continue
                        
                        # Apply second transfer
                        final_team = temp_team.copy()
                        second_id = second_out.get("element_id", second_out.get("player_id"))
                        mask2 = (final_team.get("element_id") == second_id) | (final_team.get("player_id") == second_id)
                        
                        if not mask2.any():
                            continue
                        
                        for col in ["element_id", "player_id", "web_name", "team_name", "position", "now_cost", "cost"]:
                            if col in final_team.columns and col in second_in.index:
                                final_team.loc[mask2, col] = second_in[col]
                        
                        # Evaluate 2-transfer team
                        try:
                            final_result = optimizer.optimize_team(final_team, budget=100.0, objective="mean")
                            if not final_result or not final_result.get("squad"):
                                continue
                            
                            final_xi = optimizer.optimize_starting_xi(
                                final_result["squad"], final_team, objective="mean"
                            )
                            if not final_xi or not final_xi.get("starting_xi"):
                                continue
                            
                            final_players = pd.DataFrame(final_xi["starting_xi"])
                            final_scenarios, _ = simulate_player_matrix(
                                final_players, num_scenarios, seed, minutes_uncertainty, settings
                            )
                            
                            if final_scenarios.size == 0:
                                continue
                            
                            final_captain_idx = np.argmax(final_players.get("proj_points", final_players.get("proj", np.zeros(len(final_players)))).values)
                            final_team_scores = final_scenarios.sum(axis=1) + final_scenarios[:, final_captain_idx]
                            final_cvar = cvar_score(final_team_scores, cvar_alpha)
                            final_mean = float(final_team_scores.mean())
                            
                            hits_needed = max(0, 2 - free_transfers)
                            hit_penalty = hits_needed * hit_cost
                            final_score = final_cvar - hit_penalty + bank_utility(bank, bank_weight)  # Simplified bank calc
                            
                            if final_score > best_score + 1e-6:
                                best_score = final_score
                                best_decision = StrategyDecision(
                                    transfers=[
                                        (int(first_out.get("element_id", 0)), int(first_in.get("element_id", 0))),
                                        (int(second_out.get("element_id", 0)), int(second_in.get("element_id", 0)))
                                    ],
                                    hits=hits_needed,
                                    roll=False,
                                    captain_id=int(final_players.iloc[final_captain_idx].get("element_id", 0)),
                                    vice_id=int(final_players.iloc[final_captain_idx].get("element_id", 0)),
                                    exp_points=final_mean,
                                    risk_adj=final_cvar,
                                    bank_after=bank,  # Simplified
                                    details={
                                        "reason": "2T_improve",
                                        "score_improvement": final_score - baseline_score
                                    }
                                )
                                
                                logger.debug(f"Better 2T found, score: {final_score:.2f}")
                        
                        except Exception as e:
                            logger.debug(f"Error evaluating 2T: {e}")
                            continue
        
        # Apply roll threshold check
        if best_decision.roll and (best_decision.exp_points - baseline_mean) < roll_thresh:
            best_decision.details["reason"] = "roll_threshold"
            logger.debug(f"Below roll threshold: {best_decision.exp_points - baseline_mean:.2f} < {roll_thresh}")
        
        logger.info(f"Strategy decision: {best_decision.details['reason']}, transfers: {len(best_decision.transfers)}, score: {best_score:.2f}")
        return best_decision
        
    except Exception as e:
        logger.error(f"Error in strategy evaluation: {e}")
        return StrategyDecision([], 0, True, 0, 0, 0.0, 0.0, bank, {"reason": "error", "error": str(e)})


def load_team_state_for_backtest(gw: int) -> Tuple[pd.DataFrame, float, int]:
    """
    Load team state for backtesting at a specific gameweek.
    
    This is a placeholder implementation that would need to be integrated
    with your actual backtesting infrastructure to maintain team state
    across gameweeks.
    
    Args:
        gw: Gameweek number
        
    Returns:
        Tuple of (team_df, bank, free_transfers)
    """
    try:
        # This would normally load from your backtest state management
        # For now, return a stub that indicates missing implementation
        logger.warning(f"load_team_state_for_backtest stub called for GW {gw}")
        
        # Return minimal valid state to prevent crashes
        empty_team = pd.DataFrame({
            "element_id": range(1, 16),
            "player_id": range(1, 16),
            "web_name": [f"Player_{i}" for i in range(1, 16)],
            "position": ["GK"] + ["DEF"]*5 + ["MID"]*5 + ["FWD"]*4,
            "team_name": ["Team_A"] * 15,
            "now_cost": [4.5] * 15,
            "cost": [4.5] * 15,
            "proj_points": [2.0] * 15,
            "proj": [2.0] * 15
        })
        
        return empty_team, 1.0, 1  # Empty team, £1M bank, 1 FT
        
    except Exception as e:
        logger.error(f"Error loading team state for GW {gw}: {e}")
        # Return safe defaults
        return pd.DataFrame(), 1.0, 1


def simulate_transfer_strategy(
    initial_team: pd.DataFrame,
    predictions_by_gw: Dict[int, pd.DataFrame],
    initial_bank: float = 1.0,
    horizon: int = 10,
    settings: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Simulate transfer strategy over multiple gameweeks.
    
    Args:
        initial_team: Starting 15-man squad
        predictions_by_gw: Predictions for each gameweek in horizon
        initial_bank: Starting bank amount
        horizon: Number of gameweeks to simulate
        settings: Configuration settings
        
    Returns:
        Dictionary with strategy simulation results
    """
    if settings is None:
        settings = load_settings()
    
    try:
        results = {
            "decisions": [],
            "scores": [],
            "bank_trajectory": [initial_bank],
            "total_hits": 0,
            "total_points": 0.0
        }
        
        current_team = initial_team.copy()
        current_bank = initial_bank
        free_transfers = 1
        
        for gw_offset in range(horizon):
            gw = min(predictions_by_gw.keys()) + gw_offset
            
            if gw not in predictions_by_gw:
                logger.warning(f"No predictions available for GW {gw}")
                continue
            
            market = predictions_by_gw[gw]
            
            # Make transfer decision
            decision = evaluate_one_step(current_team, market, current_bank, free_transfers, settings)
            
            # Apply transfers to team state
            for out_id, in_id in decision.transfers:
                # Update team with transfers
                if in_id in market.get("element_id", market.get("player_id", [])).values:
                    new_player = market[
                        (market.get("element_id") == in_id) | (market.get("player_id") == in_id)
                    ].iloc[0]
                    
                    mask = (current_team.get("element_id") == out_id) | (current_team.get("player_id") == out_id)
                    if mask.any():
                        for col in ["element_id", "player_id", "web_name", "team_name", "position", "now_cost", "cost"]:
                            if col in current_team.columns and col in new_player.index:
                                current_team.loc[mask, col] = new_player[col]
            
            # Update bank and free transfers
            current_bank = decision.bank_after
            if decision.roll:
                free_transfers = min(2, free_transfers + 1)  # Accumulate FTs (max 2)
            else:
                free_transfers = 1  # Reset to 1 after making transfers
            
            # Track results
            results["decisions"].append(decision)
            results["scores"].append(decision.risk_adj)
            results["bank_trajectory"].append(current_bank)
            results["total_hits"] += decision.hits
            results["total_points"] += decision.exp_points
        
        logger.info(f"Strategy simulation completed: {len(results['decisions'])} gameweeks, {results['total_hits']} total hits")
        return results
        
    except Exception as e:
        logger.error(f"Error in strategy simulation: {e}")
        return {"error": str(e)}
