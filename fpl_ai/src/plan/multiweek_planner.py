"""
Multi-week transfer planner for FPL.

Implements 10-week horizon planning with GW1 baseline initialization,
greedy optimization with risk adjustment, and comprehensive transfer planning.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from itertools import combinations
from pathlib import Path
from ..common.config import get_config, get_logger
from ..common.cache import get_cache
from .utils_prices import (
    calculate_team_value, 
    can_afford_transfer, 
    calculate_transfer_budget,
    fpl_sell_value
)
from ..optimize.optimizer import TeamOptimizer
from ..optimize.formations import FormationValidator

logger = get_logger(__name__)


class MultiWeekPlanner:
    """
    Multi-week transfer planner for FPL with GW1 baseline.
    """
    
    def __init__(self):
        """Initialize multi-week planner."""
        self.config = get_config()
        self.cache = get_cache()
        
        # Planning configuration
        planner_config = self.config.get("planner", {})
        self.horizon_gws = planner_config.get("horizon_gws", 10)
        self.max_transfers_per_gw = planner_config.get("max_transfers_per_gw", 2)
        self.max_hits_per_gw = planner_config.get("max_hits_per_gw", 2)
        self.consider_roll = planner_config.get("consider_roll", True)
        self.risk_lambda = planner_config.get("risk_lambda", 0.2)
        
        # Monte Carlo settings
        mc_config = planner_config.get("mc", {})
        self.mc_scenarios = mc_config.get("scenarios", 1000)
        self.mc_seed = mc_config.get("seed", 42)
        self.minutes_uncertainty = mc_config.get("minutes_uncertainty", 0.20)
        
        # Future planning
        self.shortlist_per_pos = planner_config.get("shortlist", {}).get("per_pos", 20)
        self.future_fixtures_weight = planner_config.get("future_fixtures_weight", 1.0)
        self.dgw_bonus = planner_config.get("dgw_bonus", 0.5)
        
        # Initialize components
        self.optimizer = TeamOptimizer()
        self.validator = FormationValidator()
        
        logger.info(f"Multi-week planner initialized: {self.horizon_gws}GW horizon")
    
    def initialize_from_gw1_team(
        self,
        gw1_team: Optional[List[Dict]] = None,
        entry_id: Optional[int] = None,
        csv_path: Optional[str] = None,
        bank: float = 0.0
    ) -> Dict[str, Any]:
        """
        Initialize planning baseline from GW1 team.
        
        Args:
            gw1_team: Direct team specification
            entry_id: FPL entry ID to fetch team from
            csv_path: Path to CSV with team data
            bank: Bank balance at GW1
            
        Returns:
            Initialized team state
        """
        logger.info("Initializing GW1 baseline team")
        
        # Load team data
        if gw1_team:
            team_data = gw1_team
        elif entry_id:
            team_data = self._fetch_team_from_entry(entry_id, gameweek=1)
        elif csv_path:
            team_data = self._load_team_from_csv(csv_path)
        else:
            logger.error("No team data source provided")
            return {}
        
        if not team_data:
            logger.error("Failed to load team data")
            return {}
        
        # Initialize team state
        team_state = {
            'gameweek': 1,
            'squad': team_data,
            'bank': bank,
            'free_transfers': 1,
            'chips_available': ['triple_captain', 'bench_boost', 'free_hit', 'wildcard'],
            'total_value': self._calculate_total_value(team_data, bank),
            'purchase_prices': {p['element_id']: p.get('purchase_price', p.get('now_cost', 0)) 
                               for p in team_data}
        }
        
        # Save baseline
        self._save_team_state(team_state, filename="team_state_gw1.json")
        
        logger.info(f"GW1 baseline initialized: £{team_state['total_value']:.1f}M, {len(team_data)} players")
        return team_state
    
    def plan_transfers(
        self,
        start_gw: int,
        predictions_by_gw: Dict[int, pd.DataFrame],
        current_state: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Plan transfers over the horizon.
        
        Args:
            start_gw: Starting gameweek for planning
            predictions_by_gw: Predictions for each gameweek
            current_state: Current team state (or None to load from cache)
            overrides: State overrides (bank, FTs, etc.)
            
        Returns:
            Complete transfer plan
        """
        logger.info(f"Planning transfers from GW {start_gw} for {self.horizon_gws} gameweeks")
        
        # Load or initialize state
        if current_state is None:
            current_state = self._load_team_state(f"team_state_gw{start_gw}.json")
            if not current_state:
                # Fallback to GW1 baseline
                current_state = self._load_team_state("team_state_gw1.json")
                if current_state:
                    current_state['gameweek'] = start_gw
                    logger.info("Using GW1 baseline for planning")
        
        if not current_state:
            logger.error("No team state available for planning")
            return {}
        
        # Apply overrides
        if overrides:
            current_state.update(overrides)
        
        # Validate predictions
        required_gws = range(start_gw, start_gw + self.horizon_gws + 1)
        missing_gws = [gw for gw in required_gws if gw not in predictions_by_gw]
        if missing_gws:
            logger.warning(f"Missing predictions for GWs: {missing_gws}")
        
        # Run planning algorithm
        plan_result = self._run_greedy_horizon_planning(
            current_state, predictions_by_gw, start_gw
        )
        
        # Enhance with analysis
        plan_result['analysis'] = self._analyze_plan(plan_result, predictions_by_gw)
        plan_result['metadata'] = {
            'start_gw': start_gw,
            'horizon': self.horizon_gws,
            'planning_timestamp': pd.Timestamp.now().isoformat(),
            'config': {
                'max_transfers_per_gw': self.max_transfers_per_gw,
                'risk_lambda': self.risk_lambda,
                'mc_scenarios': self.mc_scenarios
            }
        }
        
        logger.info(f"Transfer plan completed: {len(plan_result.get('steps', []))} steps")
        return plan_result
    
    def _run_greedy_horizon_planning(
        self,
        initial_state: Dict[str, Any],
        predictions_by_gw: Dict[int, pd.DataFrame],
        start_gw: int
    ) -> Dict[str, Any]:
        """Run greedy horizon planning algorithm."""
        current_state = initial_state.copy()
        plan_steps = []
        total_value = 0
        
        for step, gw in enumerate(range(start_gw, start_gw + self.horizon_gws + 1)):
            logger.debug(f"Planning step {step + 1}: GW {gw}")
            
            if gw not in predictions_by_gw:
                logger.warning(f"No predictions for GW {gw}, skipping")
                continue
            
            gw_predictions = predictions_by_gw[gw]
            
            # Generate transfer options for this gameweek
            transfer_options = self._generate_transfer_options(
                current_state, gw_predictions, gw
            )
            
            # Score options using Monte Carlo + CVaR
            scored_options = []
            for option in transfer_options:
                score = self._score_transfer_option(
                    option, gw_predictions, current_state, gw
                )
                scored_options.append((score, option))
            
            # Select best option
            if scored_options:
                scored_options.sort(key=lambda x: x[0], reverse=True)
                best_score, best_option = scored_options[0]
                
                # Apply transfer
                new_state = self._apply_transfer_option(current_state, best_option, gw)
                
                # Calculate gameweek value
                gw_value = self._calculate_gameweek_value(new_state, gw_predictions)
                total_value += gw_value
                
                step_info = {
                    'gameweek': gw,
                    'action': best_option,
                    'score': best_score,
                    'gw_value': gw_value,
                    'state_after': new_state.copy(),
                    'alternatives': [{'score': s, 'action': opt['action']} 
                                   for s, opt in scored_options[:3]]
                }
                
                plan_steps.append(step_info)
                current_state = new_state
                
                logger.debug(f"GW {gw}: {best_option['action']}, score: {best_score:.2f}")
            else:
                # No transfers, just roll
                step_info = {
                    'gameweek': gw,
                    'action': {'action': 'roll', 'transfers': []},
                    'score': 0,
                    'gw_value': self._calculate_gameweek_value(current_state, gw_predictions),
                    'state_after': current_state.copy(),
                    'alternatives': []
                }
                plan_steps.append(step_info)
                
                # Update FTs for next week
                current_state['free_transfers'] = min(2, current_state.get('free_transfers', 1) + 1)
        
        return {
            'steps': plan_steps,
            'final_state': current_state,
            'total_value': total_value,
            'initial_state': initial_state
        }
    
    def _generate_transfer_options(
        self,
        state: Dict[str, Any],
        predictions: pd.DataFrame,
        gameweek: int
    ) -> List[Dict[str, Any]]:
        """Generate possible transfer options for a gameweek."""
        squad = state['squad']
        free_transfers = state.get('free_transfers', 1)
        bank = state.get('bank', 0)
        
        options = []
        
        # Option 1: Roll FT
        if self.consider_roll:
            options.append({
                'action': 'roll',
                'transfers': [],
                'cost': 0,
                'hits': 0
            })
        
        # Option 2: Single transfers
        single_transfers = self._generate_single_transfers(squad, predictions, bank, free_transfers)
        options.extend(single_transfers)
        
        # Option 3: Double transfers (if enough FTs or willing to take hits)
        if free_transfers >= 2 or self.max_hits_per_gw >= 1:
            double_transfers = self._generate_double_transfers(squad, predictions, bank, free_transfers)
            options.extend(double_transfers[:10])  # Limit combinations
        
        # Option 4: Chip plays (if available)
        chip_options = self._generate_chip_options(state, predictions, gameweek)
        options.extend(chip_options)
        
        return options
    
    def _generate_single_transfers(
        self,
        squad: List[Dict],
        predictions: pd.DataFrame,
        bank: float,
        free_transfers: int
    ) -> List[Dict[str, Any]]:
        """Generate single transfer options."""
        options = []
        squad_ids = {p['element_id'] for p in squad}
        
        # Group by position
        by_position = {}
        for player in squad:
            pos = player.get('position', '')
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(player)
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            if position not in by_position:
                continue
            
            # Get candidates for this position
            position_candidates = predictions[
                (predictions['position'] == position) &
                (~predictions['element_id'].isin(squad_ids))
            ].copy()
            
            if position_candidates.empty:
                continue
            
            # Sort by projected points and take top candidates
            position_candidates = position_candidates.nlargest(
                self.shortlist_per_pos, 'proj_points'
            )
            
            # Try swapping each current player with each candidate
            for current_player in by_position[position]:
                for _, candidate in position_candidates.iterrows():
                    candidate_cost = candidate.get('now_cost', 5.0)
                    if candidate_cost > 20:  # Convert from tenths
                        candidate_cost /= 10
                    
                    # Check affordability
                    affordable, transfer_details = can_afford_transfer(
                        squad,
                        current_player['element_id'],
                        candidate_cost,
                        free_transfers,
                        bank
                    )
                    
                    if affordable:
                        # Calculate transfer value
                        current_proj = self._get_player_projection(current_player, predictions)
                        candidate_proj = candidate['proj_points']
                        points_gain = candidate_proj - current_proj
                        
                        if points_gain > 0.5:  # Minimum meaningful gain
                            hits = 0 if free_transfers > 0 else 1
                            
                            options.append({
                                'action': 'single_transfer',
                                'transfers': [{
                                    'out': current_player,
                                    'in': candidate.to_dict(),
                                    'points_gain': points_gain,
                                    'cost': transfer_details['net_cost']
                                }],
                                'cost': transfer_details['net_cost'],
                                'hits': hits,
                                'total_points_gain': points_gain - (hits * 4)
                            })
        
        # Sort by total points gain
        options.sort(key=lambda x: x['total_points_gain'], reverse=True)
        return options[:20]  # Top 20 single transfers
    
    def _generate_double_transfers(
        self,
        squad: List[Dict],
        predictions: pd.DataFrame,
        bank: float,
        free_transfers: int
    ) -> List[Dict[str, Any]]:
        """Generate double transfer options."""
        options = []
        squad_df = pd.DataFrame(squad)
        squad_ids = set(squad_df['element_id'].tolist())
        
        # Get top candidates by position
        candidates_by_pos = {}
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_candidates = predictions[
                (predictions['position'] == position) &
                (~predictions['element_id'].isin(squad_ids))
            ].nlargest(10, 'proj_points')  # Top 10 per position
            
            candidates_by_pos[position] = pos_candidates
        
        # Generate pairs of transfers
        position_combinations = [
            ('DEF', 'DEF'), ('MID', 'MID'), ('DEF', 'MID'),
            ('MID', 'FWD'), ('DEF', 'FWD'), ('GK', 'DEF')
        ]
        
        for pos1, pos2 in position_combinations:
            # Current players in these positions
            current_pos1 = squad_df[squad_df['position'] == pos1]
            current_pos2 = squad_df[squad_df['position'] == pos2]
            
            if current_pos1.empty or current_pos2.empty:
                continue
            
            # Try combinations
            for _, curr1 in current_pos1.head(3).iterrows():  # Top 3 from each position
                for _, curr2 in current_pos2.head(3).iterrows():
                    if curr1['element_id'] == curr2['element_id']:
                        continue
                    
                    # Try candidate combinations
                    for _, cand1 in candidates_by_pos[pos1].head(5).iterrows():
                        for _, cand2 in candidates_by_pos[pos2].head(5).iterrows():
                            # Check if double transfer is feasible
                            total_in_cost = (cand1.get('now_cost', 5.0) + 
                                           cand2.get('now_cost', 5.0))
                            if total_in_cost > 40:  # Convert if in tenths
                                total_in_cost /= 10
                            
                            # Calculate selling prices
                            selling1 = self._get_selling_price(curr1)
                            selling2 = self._get_selling_price(curr2)
                            total_selling = selling1 + selling2
                            
                            net_cost = total_in_cost - total_selling
                            
                            if bank + total_selling >= total_in_cost:  # Affordable
                                # Calculate points gain
                                curr1_proj = self._get_player_projection(curr1, predictions)
                                curr2_proj = self._get_player_projection(curr2, predictions)
                                total_current = curr1_proj + curr2_proj
                                
                                total_new = cand1['proj_points'] + cand2['proj_points']
                                points_gain = total_new - total_current
                                
                                if points_gain > 1.0:  # Minimum gain for double transfer
                                    hits = max(0, 2 - free_transfers)
                                    
                                    if hits <= self.max_hits_per_gw:
                                        options.append({
                                            'action': 'double_transfer',
                                            'transfers': [
                                                {
                                                    'out': curr1.to_dict(),
                                                    'in': cand1.to_dict(),
                                                    'points_gain': cand1['proj_points'] - curr1_proj,
                                                    'cost': cand1.get('now_cost', 5.0) - selling1
                                                },
                                                {
                                                    'out': curr2.to_dict(),
                                                    'in': cand2.to_dict(),
                                                    'points_gain': cand2['proj_points'] - curr2_proj,
                                                    'cost': cand2.get('now_cost', 5.0) - selling2
                                                }
                                            ],
                                            'cost': net_cost,
                                            'hits': hits,
                                            'total_points_gain': points_gain - (hits * 4)
                                        })
        
        # Sort and return top options
        options.sort(key=lambda x: x['total_points_gain'], reverse=True)
        return options[:10]
    
    def _generate_chip_options(
        self,
        state: Dict[str, Any],
        predictions: pd.DataFrame,
        gameweek: int
    ) -> List[Dict[str, Any]]:
        """Generate chip play options."""
        options = []
        available_chips = state.get('chips_available', [])
        
        # Import chips optimizer
        from ..optimize.chips_forward import ChipsOptimizer
        chips_optimizer = ChipsOptimizer()
        
        for chip in available_chips:
            if chip == 'free_hit':
                # Analyze FH value
                fh_analysis = chips_optimizer.analyze_single_chip_value(
                    'free_hit', gameweek, predictions
                )
                
                if fh_analysis.get('fh_advantage', 0) > 10:  # Minimum FH value
                    options.append({
                        'action': 'free_hit',
                        'transfers': [],
                        'cost': 0,
                        'hits': 0,
                        'chip_value': fh_analysis.get('fh_advantage', 0),
                        'chip_details': fh_analysis
                    })
            
            elif chip == 'wildcard':
                # Analyze WC value
                wc_analysis = chips_optimizer.analyze_single_chip_value(
                    'wildcard', gameweek, predictions, state['squad']
                )
                
                if wc_analysis.get('wc_advantage', 0) > 15:  # Minimum WC value
                    options.append({
                        'action': 'wildcard',
                        'transfers': [],
                        'cost': 0,
                        'hits': 0,
                        'chip_value': wc_analysis.get('total_wc_value', 0),
                        'chip_details': wc_analysis
                    })
        
        return options
    
    def _score_transfer_option(
        self,
        option: Dict[str, Any],
        predictions: pd.DataFrame,
        state: Dict[str, Any],
        gameweek: int
    ) -> float:
        """Score a transfer option using Monte Carlo + CVaR."""
        # Base score from immediate points gain
        base_score = option.get('total_points_gain', option.get('chip_value', 0))
        
        # Penalty for hits
        hit_penalty = option.get('hits', 0) * 4
        
        # Risk adjustment using Monte Carlo simulation
        if self.mc_scenarios > 0:
            mc_score = self._monte_carlo_score(option, predictions, state)
            risk_adjusted_score = mc_score
        else:
            risk_adjusted_score = base_score - hit_penalty
        
        # Future value consideration (simplified)
        future_value = self._estimate_future_value(option, predictions, gameweek)
        
        # Combine scores
        total_score = risk_adjusted_score + future_value * 0.3
        
        return total_score
    
    def _monte_carlo_score(
        self,
        option: Dict[str, Any],
        predictions: pd.DataFrame,
        state: Dict[str, Any]
    ) -> float:
        """Calculate Monte Carlo score with CVaR."""
        np.random.seed(self.mc_seed)
        scenarios = []
        
        for _ in range(self.mc_scenarios):
            scenario_score = 0
            
            # Simulate each transfer
            for transfer in option.get('transfers', []):
                if 'in' in transfer:
                    # Simulate new player performance
                    player_in = transfer['in']
                    expected_points = player_in.get('proj_points', 3.0)
                    uncertainty = player_in.get('prediction_std', 2.0)
                    
                    simulated_points = np.random.normal(expected_points, uncertainty)
                    simulated_points = max(0, simulated_points)  # No negative points
                    
                    # Subtract expected points from player out
                    player_out = transfer['out']
                    out_expected = self._get_player_projection(player_out, predictions)
                    out_uncertainty = 2.0  # Default uncertainty
                    
                    simulated_out = np.random.normal(out_expected, out_uncertainty)
                    simulated_out = max(0, simulated_out)
                    
                    scenario_score += (simulated_points - simulated_out)
            
            # Subtract hit costs
            scenario_score -= option.get('hits', 0) * 4
            
            scenarios.append(scenario_score)
        
        scenarios = np.array(scenarios)
        
        # Calculate risk metrics
        mean_score = np.mean(scenarios)
        cvar_alpha = 0.2  # Use worst 20% scenarios
        cvar_cutoff = np.percentile(scenarios, cvar_alpha * 100)
        cvar_score = np.mean(scenarios[scenarios <= cvar_cutoff])
        
        # Risk-adjusted score: mean - λ * (mean - CVaR)
        risk_adjusted = mean_score - self.risk_lambda * (mean_score - cvar_score)
        
        return risk_adjusted
    
    def _estimate_future_value(
        self,
        option: Dict[str, Any],
        predictions: pd.DataFrame,
        gameweek: int
    ) -> float:
        """Estimate future value of transfer option."""
        future_value = 0
        
        # Consider fixture difficulty and DGWs in near future
        for transfer in option.get('transfers', []):
            if 'in' in transfer:
                player_in = transfer['in']
                
                # Future fixtures value (simplified)
                team_id = player_in.get('team_id')
                
                # Bonus for bringing in players with good upcoming fixtures
                # This would typically use fixture difficulty data
                # For now, use a simplified approach
                
                if player_in.get('proj_points', 0) > 6:  # High scoring player
                    future_value += 1.0  # Bonus for quality
                
                # DGW bonus (if applicable)
                if self._has_dgw_soon(team_id, gameweek):
                    future_value += self.dgw_bonus
        
        return future_value
    
    def _apply_transfer_option(
        self,
        state: Dict[str, Any],
        option: Dict[str, Any],
        gameweek: int
    ) -> Dict[str, Any]:
        """Apply a transfer option to the current state."""
        new_state = state.copy()
        new_squad = state['squad'].copy()
        
        # Apply transfers
        for transfer in option.get('transfers', []):
            if 'out' in transfer and 'in' in transfer:
                # Remove player out
                new_squad = [p for p in new_squad if p['element_id'] != transfer['out']['element_id']]
                
                # Add player in
                player_in = transfer['in'].copy()
                player_in['purchase_price'] = player_in.get('now_cost', 5.0)
                new_squad.append(player_in)
        
        # Update state
        new_state['squad'] = new_squad
        new_state['gameweek'] = gameweek
        
        # Update FTs
        transfers_made = len(option.get('transfers', []))
        current_fts = new_state.get('free_transfers', 1)
        hits = option.get('hits', 0)
        
        if transfers_made > 0:
            remaining_fts = max(0, current_fts - transfers_made)
            new_state['free_transfers'] = remaining_fts
        else:
            # Rolling FT
            new_state['free_transfers'] = min(2, current_fts + 1)
        
        # Update bank
        transfer_cost = option.get('cost', 0)
        new_state['bank'] = new_state.get('bank', 0) - transfer_cost
        
        # Remove used chip
        if option.get('action') in ['free_hit', 'wildcard', 'triple_captain', 'bench_boost']:
            chips_available = new_state.get('chips_available', [])
            if option['action'] in chips_available:
                chips_available.remove(option['action'])
                new_state['chips_available'] = chips_available
        
        return new_state
    
    def _calculate_gameweek_value(
        self,
        state: Dict[str, Any],
        predictions: pd.DataFrame
    ) -> float:
        """Calculate expected points for the gameweek."""
        squad = state.get('squad', [])
        total_value = 0
        
        # Get best XI
        xi_result = self.optimizer.optimize_starting_xi(
            squad, predictions, objective="mean"
        )
        
        if xi_result and xi_result.get('starting_xi'):
            # Sum expected points for starting XI
            for player in xi_result['starting_xi']:
                total_value += self._get_player_projection(player, predictions)
            
            # Add captain bonus
            captain = xi_result.get('captain')
            if captain:
                captain_points = self._get_player_projection(captain, predictions)
                total_value += captain_points  # Captain gets 2x, so add 1x more
        
        return total_value
    
    def _get_player_projection(
        self,
        player: Dict[str, Any],
        predictions: pd.DataFrame
    ) -> float:
        """Get projection for a player."""
        element_id = player.get('element_id')
        if element_id and not predictions.empty:
            player_pred = predictions[predictions['element_id'] == element_id]
            if not player_pred.empty:
                return player_pred.iloc[0].get('proj_points', 2.0)
        
        return 2.0  # Default projection
    
    def _get_selling_price(self, player: Dict[str, Any]) -> float:
        """Calculate selling price for a player."""
        purchase_price = player.get('purchase_price', player.get('now_cost', 5.0))
        current_price = player.get('now_cost', purchase_price)
        
        # Convert to tenths for calculation
        if purchase_price < 20:
            purchase_price *= 10
        if current_price < 20:
            current_price *= 10
        
        selling_price_tenths = fpl_sell_value(purchase_price, current_price)
        return selling_price_tenths / 10
    
    def _has_dgw_soon(self, team_id: Optional[int], gameweek: int) -> bool:
        """Check if team has DGW in near future (simplified)."""
        # This would typically check fixture data
        # For now, return False as placeholder
        return False
    
    def _analyze_plan(
        self,
        plan_result: Dict[str, Any],
        predictions_by_gw: Dict[int, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze the generated plan."""
        steps = plan_result.get('steps', [])
        
        # Count transfer types
        transfer_summary = {
            'total_steps': len(steps),
            'rolls': 0,
            'single_transfers': 0,
            'double_transfers': 0,
            'chips_used': [],
            'total_hits': 0,
            'total_transfer_cost': 0
        }
        
        for step in steps:
            action = step.get('action', {})
            action_type = action.get('action', 'unknown')
            
            if action_type == 'roll':
                transfer_summary['rolls'] += 1
            elif action_type == 'single_transfer':
                transfer_summary['single_transfers'] += 1
            elif action_type == 'double_transfer':
                transfer_summary['double_transfers'] += 1
            elif action_type in ['free_hit', 'wildcard', 'triple_captain', 'bench_boost']:
                transfer_summary['chips_used'].append(action_type)
            
            transfer_summary['total_hits'] += action.get('hits', 0)
            transfer_summary['total_transfer_cost'] += action.get('cost', 0)
        
        # Calculate total value and risk metrics
        total_expected_value = plan_result.get('total_value', 0)
        hit_cost = transfer_summary['total_hits'] * 4
        net_value = total_expected_value - hit_cost
        
        analysis = {
            'transfer_summary': transfer_summary,
            'value_analysis': {
                'total_expected_points': total_expected_value,
                'total_hit_cost': hit_cost,
                'net_expected_value': net_value,
                'average_gw_value': total_expected_value / len(steps) if steps else 0
            },
            'efficiency_metrics': {
                'transfers_per_gw': transfer_summary['single_transfers'] / len(steps) if steps else 0,
                'hits_per_gw': transfer_summary['total_hits'] / len(steps) if steps else 0,
                'value_per_transfer': (
                    net_value / (transfer_summary['single_transfers'] + transfer_summary['double_transfers'] * 2)
                    if (transfer_summary['single_transfers'] + transfer_summary['double_transfers']) > 0 else 0
                )
            }
        }
        
        return analysis
    
    def _fetch_team_from_entry(self, entry_id: int, gameweek: int = 1) -> List[Dict]:
        """Fetch team from FPL entry ID."""
        try:
            from ..providers.fpl_picks import get_user_picks
            team_data = get_user_picks(entry_id, gameweek)
            return team_data
        except Exception as e:
            logger.error(f"Failed to fetch team from entry {entry_id}: {e}")
            return []
    
    def _load_team_from_csv(self, csv_path: str) -> List[Dict]:
        """Load team from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Expected columns: element_id, purchase_price, selling_price, is_captain, bank
            required_cols = ['element_id']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"CSV missing required columns: {required_cols}")
                return []
            
            team_data = df.to_dict('records')
            logger.info(f"Loaded team from CSV: {len(team_data)} players")
            return team_data
        
        except Exception as e:
            logger.error(f"Failed to load team from CSV {csv_path}: {e}")
            return []
    
    def _calculate_total_value(self, team_data: List[Dict], bank: float) -> float:
        """Calculate total team value."""
        team_value = calculate_team_value(team_data, include_bank=False)
        return team_value.get('total_selling_value', 100.0) + bank
    
    def _save_team_state(self, state: Dict[str, Any], filename: str) -> None:
        """Save team state to cache."""
        try:
            cache_dir = Path(self.config.get("io", {}).get("cache_dir", "cache"))
            cache_dir.mkdir(exist_ok=True)
            
            filepath = cache_dir / filename
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.debug(f"Team state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save team state: {e}")
    
    def _load_team_state(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load team state from cache."""
        try:
            cache_dir = Path(self.config.get("io", {}).get("cache_dir", "cache"))
            filepath = cache_dir / filename
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                logger.debug(f"Team state loaded from {filepath}")
                return state
            else:
                logger.debug(f"Team state file not found: {filepath}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load team state: {e}")
            return None
