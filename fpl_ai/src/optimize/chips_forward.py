"""
Chip optimization for FPL planning.

Optimizes timing and usage of FPL chips (Triple Captain, Bench Boost, 
Free Hit, Wildcard) based on future projections and opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


class ChipsOptimizer:
    """
    Optimizer for FPL chip timing and usage.
    """
    
    def __init__(self):
        """Initialize chips optimizer."""
        self.config = get_config()
        
        # Chip configuration
        self.chips_config = self.config.get("chips", {})
        self.allow_tc = self.chips_config.get("allow_tc", True)
        self.allow_bb = self.chips_config.get("allow_bb", True)
        self.allow_fh = self.chips_config.get("allow_fh", True)
        self.allow_wc = self.chips_config.get("allow_wc", True)
        
        # Planning parameters
        self.future_gws = self.chips_config.get("simulate_future_gws", 10)
        self.hit_cost = self.chips_config.get("hit_cost", 4)
        
        logger.info("Chips optimizer initialized")
    
    def optimize_chip_timing(
        self,
        predictions_by_gw: Dict[int, pd.DataFrame],
        current_squad: List[Dict],
        available_chips: List[str],
        current_gw: int,
        planning_horizon: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize chip timing over planning horizon.
        
        Args:
            predictions_by_gw: Predictions for each gameweek
            current_squad: Current 15-player squad
            available_chips: List of available chips
            current_gw: Current gameweek
            planning_horizon: Number of gameweeks to plan ahead
            
        Returns:
            Chip timing optimization results
        """
        logger.info(f"Optimizing chip timing for GW {current_gw}-{current_gw + planning_horizon}")
        
        if not predictions_by_gw or not available_chips:
            return {}
        
        # Analyze each chip opportunity
        chip_opportunities = {}
        
        if "triple_captain" in available_chips and self.allow_tc:
            chip_opportunities["triple_captain"] = self._analyze_triple_captain_opportunities(
                predictions_by_gw, current_gw, planning_horizon
            )
        
        if "bench_boost" in available_chips and self.allow_bb:
            chip_opportunities["bench_boost"] = self._analyze_bench_boost_opportunities(
                predictions_by_gw, current_squad, current_gw, planning_horizon
            )
        
        if "free_hit" in available_chips and self.allow_fh:
            chip_opportunities["free_hit"] = self._analyze_free_hit_opportunities(
                predictions_by_gw, current_gw, planning_horizon
            )
        
        if "wildcard" in available_chips and self.allow_wc:
            chip_opportunities["wildcard"] = self._analyze_wildcard_opportunities(
                predictions_by_gw, current_squad, current_gw, planning_horizon
            )
        
        # Find optimal timing
        optimal_timing = self._find_optimal_chip_timing(chip_opportunities)
        
        result = {
            'chip_opportunities': chip_opportunities,
            'optimal_timing': optimal_timing,
            'recommendations': self._generate_chip_recommendations(optimal_timing),
            'analysis_period': f"GW {current_gw}-{current_gw + planning_horizon}"
        }
        
        logger.info(f"Chip optimization completed: {len(optimal_timing)} recommendations")
        return result
    
    def analyze_single_chip_value(
        self,
        chip_type: str,
        gameweek: int,
        predictions: pd.DataFrame,
        current_squad: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze value of using a specific chip in a specific gameweek.
        
        Args:
            chip_type: Type of chip to analyze
            gameweek: Gameweek to analyze
            predictions: Player predictions for the gameweek
            current_squad: Current squad (needed for some chips)
            
        Returns:
            Chip value analysis
        """
        if chip_type == "triple_captain":
            return self._analyze_tc_value(predictions, gameweek)
        elif chip_type == "bench_boost":
            return self._analyze_bb_value(predictions, current_squad, gameweek)
        elif chip_type == "free_hit":
            return self._analyze_fh_value(predictions, gameweek)
        elif chip_type == "wildcard":
            return self._analyze_wc_value(predictions, current_squad, gameweek)
        else:
            logger.warning(f"Unknown chip type: {chip_type}")
            return {}
    
    def _analyze_triple_captain_opportunities(
        self,
        predictions_by_gw: Dict[int, pd.DataFrame],
        current_gw: int,
        horizon: int
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze Triple Captain opportunities."""
        tc_opportunities = {}
        
        for gw in range(current_gw, current_gw + horizon + 1):
            if gw not in predictions_by_gw:
                continue
            
            gw_predictions = predictions_by_gw[gw]
            
            # Find best captain candidate
            if not gw_predictions.empty:
                best_captain = gw_predictions.nlargest(1, 'proj_points').iloc[0]
                
                # TC value = extra points from doubling captain
                tc_value = best_captain['proj_points']  # Extra points (captain already gets 2x)
                
                # Consider uncertainty
                uncertainty = best_captain.get('prediction_std', 2.0)
                confidence = 1 / (1 + uncertainty)
                
                # Check for double gameweeks (higher TC value)
                is_dgw = self._is_double_gameweek(gw, gw_predictions)
                
                tc_opportunities[gw] = {
                    'captain_candidate': {
                        'element_id': best_captain['element_id'],
                        'name': best_captain.get('web_name', ''),
                        'team': best_captain.get('team_name', ''),
                        'projected_points': best_captain['proj_points']
                    },
                    'tc_value': tc_value,
                    'confidence': confidence,
                    'is_double_gameweek': is_dgw,
                    'adjusted_value': tc_value * confidence * (1.5 if is_dgw else 1.0)
                }
        
        return tc_opportunities
    
    def _analyze_bench_boost_opportunities(
        self,
        predictions_by_gw: Dict[int, pd.DataFrame],
        current_squad: List[Dict],
        current_gw: int,
        horizon: int
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze Bench Boost opportunities."""
        bb_opportunities = {}
        
        if not current_squad or len(current_squad) < 15:
            logger.warning("Insufficient squad data for BB analysis")
            return bb_opportunities
        
        # Get bench players (typically last 4 players)
        squad_df = pd.DataFrame(current_squad)
        # Assume bench is determined by some ordering (or could be passed in)
        bench_players = squad_df.tail(4)['element_id'].tolist()
        
        for gw in range(current_gw, current_gw + horizon + 1):
            if gw not in predictions_by_gw:
                continue
            
            gw_predictions = predictions_by_gw[gw]
            
            # Get bench predictions
            bench_predictions = gw_predictions[
                gw_predictions['element_id'].isin(bench_players)
            ]
            
            if not bench_predictions.empty:
                # BB value = total bench points
                bb_value = bench_predictions['proj_points'].sum()
                
                # Consider double gameweeks
                is_dgw = self._is_double_gameweek(gw, gw_predictions)
                
                # Average uncertainty
                avg_uncertainty = bench_predictions.get('prediction_std', 2.0).mean()
                confidence = 1 / (1 + avg_uncertainty)
                
                bb_opportunities[gw] = {
                    'bench_players': bench_predictions[['element_id', 'web_name', 'proj_points']].to_dict('records'),
                    'bb_value': bb_value,
                    'confidence': confidence,
                    'is_double_gameweek': is_dgw,
                    'adjusted_value': bb_value * confidence * (1.3 if is_dgw else 1.0)
                }
        
        return bb_opportunities
    
    def _analyze_free_hit_opportunities(
        self,
        predictions_by_gw: Dict[int, pd.DataFrame],
        current_gw: int,
        horizon: int
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze Free Hit opportunities."""
        fh_opportunities = {}
        
        for gw in range(current_gw, current_gw + horizon + 1):
            if gw not in predictions_by_gw:
                continue
            
            gw_predictions = predictions_by_gw[gw]
            
            if gw_predictions.empty:
                continue
            
            # Optimal FH team (top 11 players within budget)
            budget = self.chips_config.get("fh_budget_buffer", 0.5) + 100
            
            # Simple greedy selection for FH team
            fh_team = self._select_optimal_fh_team(gw_predictions, budget)
            
            if fh_team:
                fh_value = sum(p['proj_points'] for p in fh_team)
                
                # Compare against typical team value
                typical_team_value = gw_predictions['proj_points'].nlargest(11).sum() * 0.85
                fh_advantage = fh_value - typical_team_value
                
                # Consider special circumstances
                is_dgw = self._is_double_gameweek(gw, gw_predictions)
                is_bgw = self._is_blank_gameweek(gw, gw_predictions)
                
                # FH is most valuable during BGWs or very favorable DGWs
                special_multiplier = 1.0
                if is_bgw:
                    special_multiplier = 2.0
                elif is_dgw:
                    special_multiplier = 1.4
                
                fh_opportunities[gw] = {
                    'optimal_team': fh_team,
                    'fh_value': fh_value,
                    'fh_advantage': fh_advantage,
                    'is_double_gameweek': is_dgw,
                    'is_blank_gameweek': is_bgw,
                    'adjusted_value': fh_advantage * special_multiplier
                }
        
        return fh_opportunities
    
    def _analyze_wildcard_opportunities(
        self,
        predictions_by_gw: Dict[int, pd.DataFrame],
        current_squad: List[Dict],
        current_gw: int,
        horizon: int
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze Wildcard opportunities."""
        wc_opportunities = {}
        
        if not current_squad:
            return wc_opportunities
        
        # Current squad value baseline
        current_squad_ids = [p['element_id'] for p in current_squad]
        
        for gw in range(current_gw, current_gw + horizon + 1):
            if gw not in predictions_by_gw:
                continue
            
            gw_predictions = predictions_by_gw[gw]
            
            # Calculate current squad value
            current_value = 0
            current_squad_data = gw_predictions[
                gw_predictions['element_id'].isin(current_squad_ids)
            ]
            
            if not current_squad_data.empty:
                current_value = current_squad_data['proj_points'].nlargest(11).sum()
            
            # Calculate optimal squad value
            optimal_squad = self._select_optimal_wc_team(gw_predictions)
            optimal_value = sum(p['proj_points'] for p in optimal_squad[:11])
            
            wc_advantage = optimal_value - current_value
            
            # Consider future value (WC allows better long-term planning)
            future_value_boost = self._estimate_future_wc_value(
                predictions_by_gw, gw, horizon
            )
            
            wc_opportunities[gw] = {
                'current_squad_value': current_value,
                'optimal_squad_value': optimal_value,
                'immediate_advantage': wc_advantage,
                'future_value_boost': future_value_boost,
                'total_wc_value': wc_advantage + future_value_boost,
                'optimal_squad': optimal_squad[:15]  # Full squad
            }
        
        return wc_opportunities
    
    def _find_optimal_chip_timing(
        self,
        chip_opportunities: Dict[str, Dict[int, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Find optimal timing for each chip."""
        optimal_timing = {}
        
        for chip_type, opportunities in chip_opportunities.items():
            if not opportunities:
                continue
            
            # Find best gameweek for this chip
            best_gw = None
            best_value = -1
            
            for gw, analysis in opportunities.items():
                value = analysis.get('adjusted_value', analysis.get('total_wc_value', 0))
                
                if value > best_value:
                    best_value = value
                    best_gw = gw
            
            if best_gw is not None:
                optimal_timing[chip_type] = {
                    'recommended_gw': best_gw,
                    'expected_value': best_value,
                    'analysis': opportunities[best_gw]
                }
        
        return optimal_timing
    
    def _generate_chip_recommendations(
        self,
        optimal_timing: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate human-readable chip recommendations."""
        recommendations = []
        
        # Sort by gameweek
        timing_by_gw = {}
        for chip_type, timing_info in optimal_timing.items():
            gw = timing_info['recommended_gw']
            if gw not in timing_by_gw:
                timing_by_gw[gw] = []
            timing_by_gw[gw].append((chip_type, timing_info))
        
        for gw in sorted(timing_by_gw.keys()):
            for chip_type, timing_info in timing_by_gw[gw]:
                expected_value = timing_info['expected_value']
                
                # Generate recommendation text
                if chip_type == "triple_captain":
                    candidate = timing_info['analysis']['captain_candidate']
                    rec_text = f"Use Triple Captain on {candidate['name']} (projected: {candidate['projected_points']:.1f} pts)"
                elif chip_type == "bench_boost":
                    bb_value = timing_info['analysis']['bb_value']
                    rec_text = f"Use Bench Boost (projected bench value: {bb_value:.1f} pts)"
                elif chip_type == "free_hit":
                    fh_advantage = timing_info['analysis']['fh_advantage']
                    rec_text = f"Use Free Hit (projected advantage: {fh_advantage:.1f} pts)"
                elif chip_type == "wildcard":
                    total_value = timing_info['analysis']['total_wc_value']
                    rec_text = f"Use Wildcard (projected total value: {total_value:.1f} pts)"
                else:
                    rec_text = f"Use {chip_type.replace('_', ' ').title()}"
                
                recommendations.append({
                    'gameweek': gw,
                    'chip': chip_type,
                    'recommendation': rec_text,
                    'expected_value': expected_value,
                    'priority': 'high' if expected_value > 10 else 'medium' if expected_value > 5 else 'low'
                })
        
        return recommendations
    
    def _is_double_gameweek(self, gw: int, predictions: pd.DataFrame) -> bool:
        """Check if gameweek has double fixtures."""
        # Simple heuristic: if many players have unusually high projections
        if predictions.empty:
            return False
        
        avg_projection = predictions['proj_points'].mean()
        # DGW players typically have 1.5-2x normal projections
        return avg_projection > 4.5  # Rough threshold
    
    def _is_blank_gameweek(self, gw: int, predictions: pd.DataFrame) -> bool:
        """Check if gameweek has limited fixtures."""
        if predictions.empty:
            return True
        
        # BGW typically has fewer than 10 teams playing
        num_teams = predictions.get('team_id', predictions.get('team_name', pd.Series())).nunique()
        return num_teams < 10
    
    def _select_optimal_fh_team(
        self,
        predictions: pd.DataFrame,
        budget: float
    ) -> List[Dict]:
        """Select optimal Free Hit team."""
        # Simple greedy selection within budget
        # This is a simplified version - could be more sophisticated
        
        if predictions.empty:
            return []
        
        # Convert cost to millions if needed
        cost_col = 'now_cost'
        if predictions[cost_col].max() > 20:
            predictions = predictions.copy()
            predictions[cost_col] = predictions[cost_col] / 10
        
        # Filter by budget
        affordable = predictions[predictions[cost_col] <= budget].copy()
        affordable = affordable.sort_values('proj_points', ascending=False)
        
        # Simple selection: top players within budget constraint
        selected = []
        remaining_budget = budget
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        position_limits = {'GK': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}  # Max for XI
        
        for _, player in affordable.iterrows():
            position = player['position']
            cost = player[cost_col]
            
            if (position_counts[position] < position_limits[position] and
                cost <= remaining_budget and
                sum(position_counts.values()) < 11):
                
                selected.append(player.to_dict())
                remaining_budget -= cost
                position_counts[position] += 1
        
        return selected
    
    def _select_optimal_wc_team(self, predictions: pd.DataFrame) -> List[Dict]:
        """Select optimal Wildcard team (15 players)."""
        # Full squad selection with position constraints
        from .optimizer import TeamOptimizer
        
        optimizer = TeamOptimizer()
        
        # Use optimizer to build squad
        result = optimizer.optimize_team(
            predictions, 
            budget=100.0, 
            objective="mean"
        )
        
        return result.get('squad', [])
    
    def _estimate_future_wc_value(
        self,
        predictions_by_gw: Dict[int, pd.DataFrame],
        wc_gw: int,
        horizon: int
    ) -> float:
        """Estimate future value boost from using Wildcard."""
        # Simplified: WC allows better team for future weeks
        future_value = 0
        
        for gw in range(wc_gw + 1, min(wc_gw + 6, max(predictions_by_gw.keys()) + 1)):
            if gw in predictions_by_gw:
                # Assume WC gives 2-3 point advantage per week for next 5 weeks
                future_value += 2.5
        
        return future_value
    
    def _analyze_tc_value(self, predictions: pd.DataFrame, gameweek: int) -> Dict[str, Any]:
        """Analyze Triple Captain value for a specific gameweek."""
        if predictions.empty:
            return {}
        
        best_captain = predictions.nlargest(1, 'proj_points').iloc[0]
        tc_value = best_captain['proj_points']  # Extra points from TC
        
        return {
            'gameweek': gameweek,
            'captain_candidate': best_captain.to_dict(),
            'tc_value': tc_value,
            'is_double_gameweek': self._is_double_gameweek(gameweek, predictions)
        }
    
    def _analyze_bb_value(
        self,
        predictions: pd.DataFrame,
        current_squad: Optional[List[Dict]],
        gameweek: int
    ) -> Dict[str, Any]:
        """Analyze Bench Boost value for a specific gameweek."""
        if predictions.empty or not current_squad:
            return {}
        
        # Get bench players (simplified)
        squad_ids = [p['element_id'] for p in current_squad[-4:]]  # Last 4 as bench
        bench_predictions = predictions[predictions['element_id'].isin(squad_ids)]
        
        bb_value = bench_predictions['proj_points'].sum() if not bench_predictions.empty else 0
        
        return {
            'gameweek': gameweek,
            'bb_value': bb_value,
            'bench_players': bench_predictions.to_dict('records') if not bench_predictions.empty else [],
            'is_double_gameweek': self._is_double_gameweek(gameweek, predictions)
        }
    
    def _analyze_fh_value(self, predictions: pd.DataFrame, gameweek: int) -> Dict[str, Any]:
        """Analyze Free Hit value for a specific gameweek."""
        if predictions.empty:
            return {}
        
        optimal_team = self._select_optimal_fh_team(predictions, 100.5)
        fh_value = sum(p['proj_points'] for p in optimal_team)
        
        return {
            'gameweek': gameweek,
            'fh_value': fh_value,
            'optimal_team': optimal_team,
            'is_blank_gameweek': self._is_blank_gameweek(gameweek, predictions)
        }
    
    def _analyze_wc_value(
        self,
        predictions: pd.DataFrame,
        current_squad: Optional[List[Dict]],
        gameweek: int
    ) -> Dict[str, Any]:
        """Analyze Wildcard value for a specific gameweek."""
        if predictions.empty:
            return {}
        
        optimal_squad = self._select_optimal_wc_team(predictions)
        optimal_value = sum(p.get('proj_points', 0) for p in optimal_squad[:11])
        
        # Current squad value
        current_value = 0
        if current_squad:
            current_ids = [p['element_id'] for p in current_squad]
            current_predictions = predictions[predictions['element_id'].isin(current_ids)]
            current_value = current_predictions['proj_points'].nlargest(11).sum()
        
        wc_advantage = optimal_value - current_value
        
        return {
            'gameweek': gameweek,
            'wc_advantage': wc_advantage,
            'optimal_value': optimal_value,
            'current_value': current_value,
            'optimal_squad': optimal_squad
        }
