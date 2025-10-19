"""
Team selection optimizer for FPL.

Optimizes team selection using predictions, considering budget constraints,
formations, and risk preferences with Monte Carlo simulation support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from itertools import combinations
from ..common.config import get_config, get_logger
from ..common.cache import get_cache
from .formations import FormationValidator
from ..common.performance_tracker import ModelPerformanceTracker

logger = get_logger(__name__)


class TeamOptimizer:
    """
    Optimizer for FPL team selection and captain choice.
    """
    
    def __init__(self):
        """Initialize team optimizer."""
        self.config = get_config()
        self.cache = get_cache()
        self.validator = FormationValidator()
        self.performance_tracker = ModelPerformanceTracker()
        
        # Optimization parameters
        self.risk_lambda = self.config.get("mc.lambda_risk", 0.20)
        self.budget = 100.0  # Default FPL budget
        
        logger.info("Team optimizer initialized")
    
    def optimize_team(
        self,
        predictions: pd.DataFrame,
        budget: float = 100.0,
        objective: str = "mean",
        risk_lambda: Optional[float] = None,
        formation_preference: Optional[Tuple[int, int, int]] = None,
        exclude_players: Optional[List[int]] = None,
        include_players: Optional[List[int]] = None,
        gameweek: Optional[int] = None,
        save_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize team selection.
        
        Args:
            predictions: DataFrame with player predictions
            budget: Available budget in millions
            objective: Optimization objective ('mean', 'risk_adjusted', 'monte_carlo')
            risk_lambda: Risk penalty parameter (overrides config)
            formation_preference: Preferred formation tuple
            exclude_players: Player IDs to exclude
            include_players: Player IDs that must be included
            
        Returns:
            Optimization results dictionary
        """
        logger.info(f"Optimizing team with budget £{budget}m, objective: {objective}")
        
        if predictions.empty:
            logger.error("No predictions provided for optimization")
            return {}
        
        # Filter and prepare data
        opt_data = self._prepare_optimization_data(
            predictions, budget, exclude_players, include_players
        )
        
        if opt_data.empty:
            logger.error("No valid players for optimization")
            return {}
        
        # Run optimization based on objective
        if objective == "mean":
            result = self._optimize_mean_points(opt_data, budget, formation_preference)
        elif objective == "risk_adjusted":
            lambda_risk = risk_lambda or self.risk_lambda
            result = self._optimize_risk_adjusted(opt_data, budget, lambda_risk, formation_preference)
        elif objective == "monte_carlo":
            result = self._optimize_monte_carlo(opt_data, budget, formation_preference)
        else:
            logger.error(f"Unknown objective: {objective}")
            return {}
        
        if result:
            # Generate starting XI from the 15-man squad
            logger.info(f"Passing squad to starting XI optimization: {len(result['squad'])} players")
            # Debug: Check first player's position data
            if result['squad']:
                first_player = result['squad'][0]
                logger.info(f"First player position debug: '{first_player.get('position', 'MISSING')}'")
            
            xi_result = self.optimize_starting_xi(
                result['squad'], predictions, formation_preference, objective
            )
            
            # Merge starting XI into main result
            if xi_result:
                result.update(xi_result)
            else:
                logger.warning("Could not generate starting XI from squad")
                result['starting_xi'] = []
                result['formation'] = None
            
            # Validate result
            valid, error = self.validator.validate_squad(result['squad'])
            result['validation'] = {'valid': valid, 'error': error}
            
            # Add optimization metadata
            result['metadata'] = {
                'objective': objective,
                'budget_used': result.get('total_cost', 0),
                'budget_available': budget,
                'risk_lambda': risk_lambda or self.risk_lambda,
                'formation_preference': formation_preference
            }
            
            logger.info(f"Optimization completed: {result['metadata']}")
            
            # Save predictions for performance tracking if requested
            if save_predictions and gameweek is not None:
                try:
                    squad = result.get('squad', [])
                    starting_xi = result.get('starting_xi', [])
                    captain = result.get('captain', {})
                    vice_captain = result.get('vice_captain', {})
                    formation_data = result.get('formation', (0, 0, 0))
                    
                    # Handle both tuple and string formations
                    if isinstance(formation_data, tuple) and len(formation_data) == 3:
                        formation_tuple = formation_data
                        formation_str = f"{formation_data[0]}-{formation_data[1]}-{formation_data[2]}"
                    elif isinstance(formation_data, str) and '-' in formation_data:
                        parts = formation_data.split('-')
                        if len(parts) == 3:
                            formation_tuple = (int(parts[0]), int(parts[1]), int(parts[2]))
                            formation_str = formation_data
                        else:
                            formation_tuple = (0, 0, 0)
                            formation_str = "0-0-0"
                    else:
                        formation_tuple = (0, 0, 0)
                        formation_str = "0-0-0"
                    
                    self.performance_tracker.save_predicted_team(
                        gameweek=gameweek,
                        squad=squad,
                        starting_xi=starting_xi,
                        captain=captain,
                        vice_captain=vice_captain,
                        predictions_df=predictions,
                        formation=formation_str
                    )
                    logger.info(f"Saved predictions for performance tracking - GW{gameweek}")
                except Exception as e:
                    logger.warning(f"Failed to save predictions for performance tracking: {e}")
        
        return result
    
    def optimize_starting_xi(
        self,
        squad: List[Dict],
        predictions: pd.DataFrame,
        formation: Optional[Tuple[int, int, int]] = None,
        objective: str = "mean"
    ) -> Dict[str, Any]:
        """
        Optimize starting XI selection using unified formation validator.
        
        Args:
            squad: 15-player squad
            predictions: Player predictions
            formation: Target formation or None for automatic
            objective: Optimization objective
            
        Returns:
            Starting XI optimization results
        """
        logger.info("Optimizing starting XI selection using formation validator")
        logger.info(f"Received squad: {len(squad)} players")
        
        # Use the canonical formation validator
        starting_xi, actual_formation = self.validator.optimize_formation_from_players(
            squad, formation
        )
        
        if not starting_xi:
            logger.error("Formation validator returned empty starting XI")
            return {
                'starting_xi': [],
                'formation': (0, 0, 0),
                'expected_points': 0.0,
                'method': 'formation_validator_failed'
            }
        
        # Calculate expected points
        expected_points = sum(p.get('proj_points', 0.0) for p in starting_xi)
        
        logger.info(f"Starting XI optimized: {actual_formation}, {expected_points:.1f} points")
        
        return {
            'starting_xi': starting_xi,
            'formation': actual_formation,
            'expected_points': expected_points,
            'method': 'formation_validator'
        }
    
    def _prepare_optimization_data(
        self,
        predictions: pd.DataFrame,
        budget: float,
        exclude_players: Optional[List[int]],
        include_players: Optional[List[int]]
    ) -> pd.DataFrame:
        """Prepare data for optimization."""
        opt_data = predictions.copy()
        
        # Map mean_points to proj_points if needed for backward compatibility
        if 'mean_points' in opt_data.columns and 'proj_points' not in opt_data.columns:
            opt_data['proj_points'] = opt_data['mean_points']
            logger.debug("Mapped mean_points to proj_points for optimization")
        
        # Remove duplicates first - keep the best version of each player
        initial_count = len(opt_data)
        opt_data = opt_data.drop_duplicates(subset=['element_id'], keep='first')
        if len(opt_data) < initial_count:
            logger.info(f"Removed {initial_count - len(opt_data)} duplicate players")
        
        # Filter out excluded players
        if exclude_players:
            opt_data = opt_data[~opt_data['element_id'].isin(exclude_players)]
        
        # Ensure required columns exist
        required_cols = ['element_id', 'position', 'now_cost', 'proj_points']
        missing_cols = [col for col in required_cols if col not in opt_data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Convert cost to millions if needed
        if opt_data['now_cost'].max() > 20:  # Likely in tenths
            opt_data['now_cost'] = opt_data['now_cost'] / 10
        
        # Filter by budget (remove players too expensive)
        opt_data = opt_data[opt_data['now_cost'] <= budget]
        
        # Add value metrics
        opt_data['value_per_point'] = opt_data['now_cost'] / (opt_data['proj_points'] + 0.1)
        opt_data['points_per_million'] = opt_data['proj_points'] / opt_data['now_cost']
        
        # Sort by projected points (best first)
        opt_data = opt_data.sort_values('proj_points', ascending=False)
        
        logger.debug(f"Prepared optimization data: {len(opt_data)} players")
        return opt_data
    
    def _optimize_mean_points(
        self,
        opt_data: pd.DataFrame,
        budget: float,
        formation_preference: Optional[Tuple[int, int, int]]
    ) -> Dict[str, Any]:
        """Optimize for maximum expected points with budget utilization and premium constraints."""
        logger.debug("Running mean points optimization with budget constraints")
        
        # Add budget utilization penalty and premium constraints
        opt_data = self._enhance_objective_with_constraints(opt_data, budget)
        
        # Use greedy approach for speed, then local search
        best_squad = self._greedy_selection(opt_data, budget, formation_preference)
        
        if not best_squad:
            return {}
        
        # Try local improvements with budget constraints
        improved_squad = self._local_search_with_constraints(best_squad, opt_data, budget)
        
        # Validate premium requirements
        improved_squad = self._enforce_premium_constraints(improved_squad, opt_data, budget)
        
        # Final validation
        if len(improved_squad) != 15:
            logger.error(f"Final squad size is incorrect: {len(improved_squad)}/15 players")
            logger.error(f"Position breakdown: GK={len([p for p in improved_squad if p['position']=='GK'])}, DEF={len([p for p in improved_squad if p['position']=='DEF'])}, MID={len([p for p in improved_squad if p['position']=='MID'])}, FWD={len([p for p in improved_squad if p['position']=='FWD'])}")
            return {}
        
        return {
            'squad': improved_squad,
            'total_cost': sum(p['now_cost'] for p in improved_squad),
            'expected_points': sum(p['proj_points'] for p in improved_squad),
            'method': 'budget_constrained_greedy_with_local_search'
        }
    
    def _optimize_risk_adjusted(
        self,
        opt_data: pd.DataFrame,
        budget: float,
        risk_lambda: float,
        formation_preference: Optional[Tuple[int, int, int]]
    ) -> Dict[str, Any]:
        """Optimize for risk-adjusted returns."""
        logger.debug(f"Running risk-adjusted optimization (λ={risk_lambda})")
        
        # Calculate risk-adjusted scores
        if 'prediction_std' in opt_data.columns:
            opt_data['risk_adjusted_score'] = (
                opt_data['proj_points'] - risk_lambda * opt_data['prediction_std']
            )
        else:
            # Use position-based risk estimates
            position_risks = {'GK': 1.8, 'DEF': 2.0, 'MID': 2.8, 'FWD': 3.0}
            # Ensure we get a Series before calling fillna
            risk_mapped = opt_data['position'].map(position_risks)
            if isinstance(risk_mapped, pd.Series):
                opt_data['estimated_risk'] = risk_mapped.fillna(2.5)
            else:
                # If it's a scalar, create a Series with the same length
                opt_data['estimated_risk'] = pd.Series([risk_mapped] * len(opt_data), index=opt_data.index)
            opt_data['risk_adjusted_score'] = (
                opt_data['proj_points'] - risk_lambda * opt_data['estimated_risk']
            )
        
        # Sort by risk-adjusted score
        opt_data = opt_data.sort_values('risk_adjusted_score', ascending=False)
        
        # Use greedy selection with risk-adjusted scores
        best_squad = self._greedy_selection(
            opt_data, budget, formation_preference, score_col='risk_adjusted_score'
        )
        
        if not best_squad:
            return {}
        
        return {
            'squad': best_squad,
            'total_cost': sum(p['now_cost'] for p in best_squad),
            'expected_points': sum(p['proj_points'] for p in best_squad),
            'risk_adjusted_score': sum(p.get('risk_adjusted_score', 0) for p in best_squad),
            'method': 'risk_adjusted_greedy'
        }
    
    def _optimize_monte_carlo(
        self,
        opt_data: pd.DataFrame,
        budget: float,
        formation_preference: Optional[Tuple[int, int, int]]
    ) -> Dict[str, Any]:
        """Optimize using Monte Carlo simulation results."""
        logger.debug("Running Monte Carlo optimization")
        
        # Check if MC results are available
        mc_cols = ['mean', 'p10', 'p90', 'cvar']
        if any(col in opt_data.columns for col in mc_cols):
            # Use Monte Carlo metrics
            if 'cvar' in opt_data.columns:
                score_col = 'cvar'
            elif 'mean' in opt_data.columns:
                score_col = 'mean'
            else:
                score_col = 'proj_points'
            
            opt_data = opt_data.sort_values(score_col, ascending=False)
        else:
            # Fallback to regular optimization
            logger.warning("No Monte Carlo results available, using mean optimization")
            return self._optimize_mean_points(opt_data, budget, formation_preference)
        
        # Use greedy selection
        best_squad = self._greedy_selection(
            opt_data, budget, formation_preference, score_col=score_col
        )
        
        if not best_squad:
            return {}
        
        return {
            'squad': best_squad,
            'total_cost': sum(p['now_cost'] for p in best_squad),
            'expected_points': sum(p['proj_points'] for p in best_squad),
            'monte_carlo_score': sum(p.get(score_col, 0) for p in best_squad),
            'method': f'monte_carlo_{score_col}'
        }
    
    def _greedy_selection(
        self,
        opt_data: pd.DataFrame,
        budget: float,
        formation_preference: Optional[Tuple[int, int, int]],
        score_col: str = 'proj_points'
    ) -> List[Dict]:
        """Greedy player selection using unified team builder."""
        from .team_builder import build_squad_from_predictions
        
        # Use the unified team builder
        squad = build_squad_from_predictions(opt_data, budget)
        
        if not squad:
            logger.warning("Team builder returned empty squad")
            return []
        
        logger.info(f"Successfully built 15-player squad: GK={len([p for p in squad if p['position']=='GK'])}, DEF={len([p for p in squad if p['position']=='DEF'])}, MID={len([p for p in squad if p['position']=='MID'])}, FWD={len([p for p in squad if p['position']=='FWD'])}")
        return squad
    
    def _local_search(
        self,
        initial_squad: List[Dict],
        opt_data: pd.DataFrame,
        budget: float,
        max_iterations: int = 100
    ) -> List[Dict]:
        """Local search for squad improvement."""
        current_squad = initial_squad.copy()
        current_score = sum(p['proj_points'] for p in current_squad)
        
        # Create lookup for faster access
        available_players = opt_data.to_dict('records')
        squad_ids = {p['element_id'] for p in current_squad}
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try single player swaps
            for i, current_player in enumerate(current_squad):
                current_pos = current_player['position']
                
                # Find potential replacements
                for candidate in available_players:
                    if (candidate['element_id'] not in squad_ids and
                        candidate['position'] == current_pos):
                        
                        # Check if swap improves score and fits budget
                        cost_diff = candidate['now_cost'] - current_player['now_cost']
                        current_cost = sum(p['now_cost'] for p in current_squad)
                        
                        if current_cost + cost_diff <= budget:
                            score_diff = candidate['proj_points'] - current_player['proj_points']
                            
                            if score_diff > 0.1:  # Meaningful improvement
                                # Make the swap
                                squad_ids.remove(current_player['element_id'])
                                squad_ids.add(candidate['element_id'])
                                current_squad[i] = candidate
                                current_score += score_diff
                                improved = True
                                break
                
                if improved:
                    break
        
        logger.debug(f"Local search completed: {iterations} iterations, score: {current_score:.2f}")
        return current_squad
    
    def _enhance_objective_with_constraints(
        self,
        opt_data: pd.DataFrame,
        budget: float
    ) -> pd.DataFrame:
        """Enhance objective function with budget utilization and premium constraints."""
        opt_data = opt_data.copy()
        
        # Budget utilization penalty (soft constraint)
        # Encourage spending closer to full budget
        budget_penalty_lambda = 0.5  # Configurable
        
        # Premium player bonus (encourage premium picks)
        premium_threshold = {
            'GK': 5.5,   # Premium GK threshold
            'DEF': 6.0,  # Premium DEF threshold  
            'MID': 9.0,  # Premium MID threshold
            'FWD': 9.5   # Premium FWD threshold
        }
        
        # Add premium bonus to scores
        opt_data['premium_bonus'] = 0
        for pos, threshold in premium_threshold.items():
            mask = (opt_data['position'] == pos) & (opt_data['now_cost'] >= threshold)
            opt_data.loc[mask, 'premium_bonus'] = 2.0  # 2 point bonus for premium players
        
        # Enhanced scoring function
        opt_data['enhanced_score'] = (
            opt_data['proj_points'] + 
            opt_data['premium_bonus'] +
            budget_penalty_lambda * (opt_data['now_cost'] / budget)
        )
        
        # Sort by enhanced score
        opt_data = opt_data.sort_values('enhanced_score', ascending=False)
        
        logger.debug("Enhanced objective with budget and premium constraints")
        return opt_data
    
    def _local_search_with_constraints(
        self,
        initial_squad: List[Dict],
        opt_data: pd.DataFrame,
        budget: float,
        max_iterations: int = 100
    ) -> List[Dict]:
        """Local search with budget utilization constraints."""
        current_squad = initial_squad.copy()
        current_score = sum(p.get('enhanced_score', p['proj_points']) for p in current_squad)
        current_cost = sum(p['now_cost'] for p in current_squad)
        
        # Budget floor - prefer teams that spend more
        budget_floor = 98.0  # Minimum £98m spend
        budget_penalty = 0.5 if current_cost < budget_floor else 0
        
        available_players = opt_data.to_dict('records')
        squad_ids = {p['element_id'] for p in current_squad}
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try single player swaps that improve budget utilization
            for i, current_player in enumerate(current_squad):
                current_pos = current_player['position']
                
                for candidate in available_players:
                    if (candidate['element_id'] not in squad_ids and
                        candidate['position'] == current_pos):
                        
                        # Check if swap improves enhanced score and budget utilization
                        cost_diff = candidate['now_cost'] - current_player['now_cost']
                        new_cost = current_cost + cost_diff
                        
                        if new_cost <= budget:
                            # Calculate enhanced score difference
                            current_enhanced = current_player.get('enhanced_score', current_player['proj_points'])
                            candidate_enhanced = candidate.get('enhanced_score', candidate['proj_points'])
                            score_diff = candidate_enhanced - current_enhanced
                            
                            # Bonus for better budget utilization
                            if new_cost >= budget_floor and current_cost < budget_floor:
                                score_diff += 1.0  # Bonus for reaching budget floor
                            elif new_cost > current_cost:
                                score_diff += 0.2  # Small bonus for spending more
                            
                            if score_diff > 0.1:  # Meaningful improvement
                                squad_ids.remove(current_player['element_id'])
                                squad_ids.add(candidate['element_id'])
                                current_squad[i] = candidate
                                current_cost = new_cost
                                current_score += score_diff
                                improved = True
                                break
                
                if improved:
                    break
        
        logger.debug(f"Local search completed: {iterations} iterations, final cost: £{current_cost:.1f}m")
        return current_squad
    
    def _enforce_premium_constraints(
        self,
        squad: List[Dict],
        opt_data: pd.DataFrame,
        budget: float
    ) -> List[Dict]:
        """Enforce minimum premium player requirements."""
        # Count premium players by position
        premium_threshold = {
            'GK': 5.5, 'DEF': 6.0, 'MID': 9.0, 'FWD': 9.5
        }
        
        premium_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for player in squad:
            pos = player['position']
            if pos in premium_threshold and player['now_cost'] >= premium_threshold[pos]:
                premium_counts[pos] += 1
        
        # Minimum requirements
        min_premium_attackers = 2  # At least 2 premium MID/FWD
        min_premium_mid_fwd = 1    # At least 1 premium MID and 1 premium FWD
        
        current_premium_attackers = premium_counts['MID'] + premium_counts['FWD']
        has_premium_mid = premium_counts['MID'] >= 1
        has_premium_fwd = premium_counts['FWD'] >= 1
        
        # If constraints not met, try to upgrade
        if current_premium_attackers < min_premium_attackers or not has_premium_mid or not has_premium_fwd:
            logger.info(f"Premium constraints not met. Current: {premium_counts}")
            squad = self._upgrade_to_meet_premium_constraints(
                squad, opt_data, budget, premium_threshold
            )
        
        return squad
    
    def _upgrade_to_meet_premium_constraints(
        self,
        squad: List[Dict],
        opt_data: pd.DataFrame,
        budget: float,
        premium_threshold: Dict[str, float]
    ) -> List[Dict]:
        """Upgrade players to meet premium constraints."""
        current_cost = sum(p['now_cost'] for p in squad)
        remaining_budget = budget - current_cost
        
        # Find cheapest non-premium players to replace
        non_premium = [p for p in squad if p['now_cost'] < premium_threshold.get(p['position'], 0)]
        non_premium.sort(key=lambda x: x['now_cost'])  # Cheapest first
        
        # Find premium replacements
        # Ensure we get a Series before calling fillna
        threshold_mapped = opt_data['position'].map(premium_threshold)
        if isinstance(threshold_mapped, pd.Series):
            threshold_series = threshold_mapped.fillna(0)
        else:
            # If it's a scalar, create a Series with the same length
            threshold_series = pd.Series([threshold_mapped] * len(opt_data), index=opt_data.index)
        
        available_premium = opt_data[
            opt_data['now_cost'] >= threshold_series
        ].copy()
        
        if available_premium.empty:
            logger.warning("No premium players available for upgrade")
            return squad
        
        # Try to upgrade cheapest non-premiums to premium
        upgraded_squad = squad.copy()
        
        for player in non_premium:
            if remaining_budget <= 0:
                break
                
            pos = player['position']
            premium_for_pos = available_premium[available_premium['position'] == pos]
            
            if premium_for_pos.empty:
                continue
            
            # Find cheapest premium upgrade
            premium_upgrade = premium_for_pos.nsmallest(1, 'now_cost').iloc[0]
            cost_diff = premium_upgrade['now_cost'] - player['now_cost']
            
            if cost_diff <= remaining_budget:
                # Make the upgrade - replace the player
                for i, squad_player in enumerate(upgraded_squad):
                    if squad_player['element_id'] == player['element_id']:
                        upgraded_squad[i] = premium_upgrade.to_dict()
                        remaining_budget -= cost_diff
                        logger.info(f"Upgraded {player['web_name']} (£{player['now_cost']:.1f}m) to {premium_upgrade['web_name']} (£{premium_upgrade['now_cost']:.1f}m)")
                        break
        
        # Remove any remaining duplicates after upgrades
        unique_squad = []
        seen_ids = set()
        for player in upgraded_squad:
            player_id = player.get('element_id')
            if player_id and player_id not in seen_ids:
                unique_squad.append(player)
                seen_ids.add(player_id)
            elif player_id in seen_ids:
                logger.warning(f"Removed duplicate player: {player.get('web_name', 'Unknown')} (ID: {player_id})")
        
        upgraded_squad = unique_squad
        
        # Ensure we still have exactly 15 players after upgrades
        if len(upgraded_squad) != 15:
            logger.warning(f"Squad size changed during upgrade: {len(upgraded_squad)}/15 players")
            # If we lost players during upgrade, we need to add them back
            if len(upgraded_squad) < 15:
                logger.error("Upgrade process reduced squad size - this should not happen")
                # Try to rebuild the squad to 15 players
                missing_players = 15 - len(upgraded_squad)
                logger.info(f"Attempting to add {missing_players} missing players back to squad")
                
                # Get available players not in current squad
                current_ids = {p['element_id'] for p in upgraded_squad}
                available_players = opt_data[~opt_data['element_id'].isin(current_ids)].copy()
                
                if not available_players.empty:
                    # Sort by projection and add best available players
                    available_players = available_players.sort_values('proj_points', ascending=False)
                    
                    for _, player in available_players.iterrows():
                        if len(upgraded_squad) >= 15:
                            break
                        
                        player_dict = player.to_dict()
                        # Check if adding this player maintains budget constraints
                        new_cost = sum(p['now_cost'] for p in upgraded_squad) + player_dict['now_cost']
                        if new_cost <= budget:
                            upgraded_squad.append(player_dict)
                            logger.info(f"Added {player_dict['web_name']} back to squad")
                
                # If still not 15 players, return original squad
                if len(upgraded_squad) != 15:
                    logger.error(f"Could not restore squad to 15 players: {len(upgraded_squad)}/15")
                    return squad
        
        # Final validation - ensure no duplicates
        final_squad = []
        seen_ids = set()
        for player in upgraded_squad:
            player_id = player.get('element_id')
            if player_id and player_id not in seen_ids:
                final_squad.append(player)
                seen_ids.add(player_id)
        
        if len(final_squad) != 15:
            logger.error(f"Final squad validation failed: {len(final_squad)}/15 players")
            return squad
        
        return final_squad
    
    def _optimize_captain(
        self,
        starting_xi: List[Dict],
        predictions: pd.DataFrame
    ) -> Dict[str, Any]:
        """Optimize captain and vice-captain selection."""
        # Work directly with the starting XI data - it should already have the needed fields
        xi_data = []
        
        for player in starting_xi:
            player_data = player.copy()
            
            # Ensure we have proj_points (use mean_points as fallback)
            if 'proj_points' not in player_data or player_data['proj_points'] == 0:
                player_data['proj_points'] = player_data.get('mean_points', 0)
            
            xi_data.append(player_data)
        
        # Convert to DataFrame for easier sorting
        xi_df = pd.DataFrame(xi_data)
        
        # Sort by projected points (highest first)
        if 'proj_points' in xi_df.columns:
            xi_df = xi_df.sort_values('proj_points', ascending=False)
        else:
            # Fallback to mean_points
            xi_df = xi_df.sort_values('mean_points', ascending=False)
        
        # Captain is highest expected points
        captain = xi_df.iloc[0].to_dict()
        
        # Vice-captain considerations (not same team as captain ideally)
        vc_candidates = xi_df.iloc[1:].copy()
        
        # Prefer different team for vice-captain
        captain_team = captain.get('team_id', captain.get('team_name'))
        
        # Get team column that exists
        team_col = 'team_id' if 'team_id' in vc_candidates.columns else 'team_name'
        if team_col in vc_candidates.columns:
            different_team = vc_candidates[vc_candidates[team_col] != captain_team]
        else:
            different_team = vc_candidates  # Fallback if no team column
        
        if not different_team.empty:
            vice_captain = different_team.iloc[0].to_dict()
        else:
            vice_captain = vc_candidates.iloc[0].to_dict()
        
        return {
            'captain': captain,
            'vice_captain': vice_captain,
            'captain_expected_points': captain.get('proj_points', 0) * 2,  # Captain gets 2x
            'vice_captain_expected_points': vice_captain.get('proj_points', 0)
        }
