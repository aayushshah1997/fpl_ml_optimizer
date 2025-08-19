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
        include_players: Optional[List[int]] = None
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
        
        return result
    
    def optimize_starting_xi(
        self,
        squad: List[Dict],
        predictions: pd.DataFrame,
        formation: Optional[Tuple[int, int, int]] = None,
        objective: str = "mean"
    ) -> Dict[str, Any]:
        """
        Optimize starting XI selection from squad.
        
        Args:
            squad: 15-player squad
            predictions: Player predictions
            formation: Target formation or None for automatic
            objective: Optimization objective
            
        Returns:
            Starting XI optimization results
        """
        logger.info("Optimizing starting XI selection")
        
        # Convert squad to DataFrame for easier manipulation
        squad_df = pd.DataFrame(squad)
        
        # Merge with predictions
        if 'element_id' in squad_df.columns:
            squad_data = squad_df.merge(
                predictions, 
                left_on='element_id', 
                right_on='element_id',
                how='left'
            )
        else:
            logger.error("Squad data missing element_id")
            return {}
        
        # Optimize formation and selection
        if formation:
            selected_xi, actual_formation = self.validator.optimize_formation_from_players(
                squad_data.to_dict('records'), formation
            )
        else:
            # Try all formations and pick best
            best_score = -1
            best_xi = []
            best_formation = None
            
            for candidate_formation in self.validator.get_valid_formations():
                xi_candidates, form_used = self.validator.optimize_formation_from_players(
                    squad_data.to_dict('records'), candidate_formation
                )
                
                if xi_candidates:
                    # Calculate formation score
                    formation_score = sum(
                        player.get('proj_points', 0) for player in xi_candidates
                    )
                    
                    if formation_score > best_score:
                        best_score = formation_score
                        best_xi = xi_candidates
                        best_formation = form_used
            
            selected_xi = best_xi
            actual_formation = best_formation
        
        if not selected_xi:
            logger.error("Could not form valid starting XI")
            return {}
        
        # Optimize captain and vice-captain
        captain_optimization = self._optimize_captain(selected_xi, predictions)
        
        result = {
            'starting_xi': selected_xi,
            'formation': actual_formation,
            'captain': captain_optimization.get('captain'),
            'vice_captain': captain_optimization.get('vice_captain'),
            'total_points': sum(p.get('proj_points', 0) for p in selected_xi),
            'captain_points': captain_optimization.get('captain_expected_points', 0),
            'bench': [p for p in squad if p not in selected_xi]
        }
        
        # Validate
        valid, error = self.validator.validate_starting_xi(selected_xi)
        result['validation'] = {'valid': valid, 'error': error}
        
        logger.info(f"Starting XI optimized: {actual_formation}, {result['total_points']:.1f} points")
        return result
    
    def _prepare_optimization_data(
        self,
        predictions: pd.DataFrame,
        budget: float,
        exclude_players: Optional[List[int]],
        include_players: Optional[List[int]]
    ) -> pd.DataFrame:
        """Prepare data for optimization."""
        opt_data = predictions.copy()
        
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
        """Optimize for maximum expected points."""
        logger.debug("Running mean points optimization")
        
        # Use greedy approach for speed, then local search
        best_squad = self._greedy_selection(opt_data, budget, formation_preference)
        
        if not best_squad:
            return {}
        
        # Try local improvements
        improved_squad = self._local_search(best_squad, opt_data, budget)
        
        return {
            'squad': improved_squad,
            'total_cost': sum(p['now_cost'] for p in improved_squad),
            'expected_points': sum(p['proj_points'] for p in improved_squad),
            'method': 'greedy_with_local_search'
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
            opt_data['estimated_risk'] = opt_data['position'].map(position_risks).fillna(2.5)
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
        """Greedy player selection respecting constraints."""
        # Group players by position
        by_position = {}
        for _, player in opt_data.iterrows():
            pos = player['position']
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(player.to_dict())
        
        # Sort each position by score
        for pos in by_position:
            by_position[pos].sort(key=lambda x: x.get(score_col, 0), reverse=True)
        
        # Position requirements
        position_limits = self.validator.position_limits
        
        # Try to build squad
        squad = []
        remaining_budget = budget
        
        # Fill each position to minimum requirements first
        for position, (min_count, max_count) in position_limits.items():
            if position not in by_position:
                continue
            
            # Add minimum required players
            for i in range(min(min_count, len(by_position[position]))):
                player = by_position[position][i]
                
                if player['now_cost'] <= remaining_budget:
                    squad.append(player)
                    remaining_budget -= player['now_cost']
                else:
                    # Can't afford minimum, optimization failed
                    return []
        
        if len(squad) != 15:
            logger.warning(f"Could not build complete squad: {len(squad)}/15 players")
            return []
        
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
    
    def _optimize_captain(
        self,
        starting_xi: List[Dict],
        predictions: pd.DataFrame
    ) -> Dict[str, Any]:
        """Optimize captain and vice-captain selection."""
        # Merge with predictions for latest data
        xi_df = pd.DataFrame(starting_xi)
        
        if 'element_id' in xi_df.columns:
            xi_with_pred = xi_df.merge(
                predictions[['element_id', 'proj_points', 'prediction_std']],
                on='element_id',
                how='left',
                suffixes=('', '_pred')
            )
        else:
            xi_with_pred = xi_df
        
        # Use prediction data if available, otherwise original
        for col in ['proj_points', 'prediction_std']:
            pred_col = f"{col}_pred"
            if pred_col in xi_with_pred.columns:
                xi_with_pred[col] = xi_with_pred[pred_col].fillna(xi_with_pred.get(col, 0))
        
        # Sort by expected points
        xi_with_pred = xi_with_pred.sort_values('proj_points', ascending=False)
        
        # Captain is highest expected points
        captain = xi_with_pred.iloc[0].to_dict()
        
        # Vice-captain considerations (not same team as captain ideally)
        vc_candidates = xi_with_pred.iloc[1:].copy()
        
        # Prefer different team for vice-captain
        captain_team = captain.get('team_id', captain.get('team_name'))
        different_team = vc_candidates[
            vc_candidates.get('team_id', vc_candidates.get('team_name', '')) != captain_team
        ]
        
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
