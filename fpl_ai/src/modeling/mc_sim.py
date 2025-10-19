"""
Monte Carlo simulation for uncertainty quantification and risk assessment.

Provides probabilistic predictions with proper correlation modeling
for portfolio optimization and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import norm, multivariate_normal
from sklearn.covariance import LedoitWolf
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulator for FPL predictions with correlation modeling.
    """
    
    def __init__(self):
        """Initialize Monte Carlo simulator."""
        self.config = get_config()
        
        # MC configuration
        self.num_scenarios = self.config.get("mc.num_scenarios", 2000)
        self.seed = self.config.get("mc.seed", 42)
        self.minutes_uncertainty = self.config.get("mc.minutes_uncertainty", 0.20)
        
        # Correlation configuration
        self.team_correlation = self.config.get("mc.correlation.team_level", 0.25)
        self.opponent_correlation = self.config.get("mc.correlation.opponent_level", 0.15)
        
        # Risk parameters
        self.risk_measure = self.config.get("mc.risk_measure", "cvar")
        self.cvar_alpha = self.config.get("mc.cvar_alpha", 0.2)
        
        # Default uncertainties by position
        self.position_uncertainties = {
            'GK': 1.8,
            'DEF': 2.0,
            'MID': 2.8,
            'FWD': 3.0
        }
        
        # Set random seed
        np.random.seed(self.seed)
        
        logger.info(f"Monte Carlo simulator initialized with {self.num_scenarios} scenarios")
    
    def simulate_predictions(
        self,
        predictions: pd.DataFrame,
        residuals_data: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on predictions.
        
        Args:
            predictions: DataFrame with point predictions and uncertainties
            residuals_data: Optional residual statistics by position
            correlation_matrix: Optional pre-computed correlation matrix
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running Monte Carlo simulation for {len(predictions)} players")
        
        if predictions.empty:
            return {}
        
        # Prepare simulation inputs
        sim_inputs = self._prepare_simulation_inputs(predictions, residuals_data)
        
        if not sim_inputs:
            logger.error("Failed to prepare simulation inputs")
            return {}
        
        # Build correlation matrix
        if correlation_matrix is None:
            correlation_matrix = self._build_correlation_matrix(predictions)
        
        # Run simulation
        scenarios = self._run_simulation(sim_inputs, correlation_matrix)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(scenarios, predictions)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(scenarios)
        
        results = {
            'scenarios': scenarios,
            'summary_stats': summary_stats,
            'risk_metrics': risk_metrics,
            'correlation_matrix': correlation_matrix,
            'num_scenarios': self.num_scenarios,
            'config': {
                'team_correlation': self.team_correlation,
                'opponent_correlation': self.opponent_correlation,
                'minutes_uncertainty': self.minutes_uncertainty
            }
        }
        
        logger.info("Monte Carlo simulation completed")
        return results
    
    def _prepare_simulation_inputs(
        self,
        predictions: pd.DataFrame,
        residuals_data: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Prepare inputs for Monte Carlo simulation."""
        try:
            sim_inputs = {
                'player_ids': predictions['element_id'].values,
                'mean_points': predictions['proj_points'].values,
                'positions': predictions['position'].values,
                'teams': predictions.get('team_id', predictions.get('team_name', '')).values,
                'expected_minutes': predictions.get('expected_minutes', np.full(len(predictions), 75)).values
            }
            
            # Get uncertainties
            uncertainties = []
            
            for _, player in predictions.iterrows():
                position = player.get('position', 'MID')
                
                # Use residuals data if available, otherwise defaults
                if residuals_data and position in residuals_data:
                    uncertainty = residuals_data[position]
                elif 'prediction_std' in predictions.columns:
                    uncertainty = player['prediction_std']
                else:
                    uncertainty = self.position_uncertainties.get(position, 2.5)
                
                # Apply uncertainty bump for low-tier league priors
                if 'prior_league_uncertainty' in predictions.columns:
                    league_uncertainty_bump = player.get('prior_league_uncertainty', 0.0)
                    if league_uncertainty_bump > 0:
                        uncertainty = uncertainty * (1.0 + league_uncertainty_bump)
                        logger.debug(f"Applied uncertainty bump {league_uncertainty_bump:.2f} to player {player.get('element_id', '')}")
                
                uncertainties.append(uncertainty)
            
            sim_inputs['uncertainties'] = np.array(uncertainties)
            
            return sim_inputs
            
        except Exception as e:
            logger.error(f"Error preparing simulation inputs: {e}")
            return {}
    
    def _build_correlation_matrix(self, predictions: pd.DataFrame) -> np.ndarray:
        """Build correlation matrix for players."""
        n_players = len(predictions)
        correlation_matrix = np.eye(n_players)
        
        try:
            # Add team-level correlations
            if 'team_id' in predictions.columns or 'team_name' in predictions.columns:
                team_col = 'team_id' if 'team_id' in predictions.columns else 'team_name'
                
                for i in range(n_players):
                    for j in range(i + 1, n_players):
                        # Same team correlation
                        if predictions.iloc[i][team_col] == predictions.iloc[j][team_col]:
                            correlation_matrix[i, j] = self.team_correlation
                            correlation_matrix[j, i] = self.team_correlation
            
            # Add opponent-level correlations (if opponent data available)
            if 'opponent_id' in predictions.columns:
                for i in range(n_players):
                    for j in range(i + 1, n_players):
                        # Same opponent correlation (negative)
                        if (predictions.iloc[i].get('opponent_id') == predictions.iloc[j].get('opponent_id') and
                            predictions.iloc[i].get('team_id') != predictions.iloc[j].get('team_id')):
                            correlation_matrix[i, j] = -self.opponent_correlation
                            correlation_matrix[j, i] = -self.opponent_correlation
            
            # Ensure positive definite matrix
            correlation_matrix = self._make_positive_definite(correlation_matrix)
            
            logger.debug(f"Built correlation matrix: {n_players}x{n_players}")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error building correlation matrix: {e}")
            return np.eye(n_players)
    
    def _make_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive definite."""
        try:
            # Use Ledoit-Wolf shrinkage to make positive definite
            shrinkage = LedoitWolf()
            
            # Create dummy data for shrinkage estimation
            n = matrix.shape[0]
            dummy_data = np.random.multivariate_normal(np.zeros(n), matrix, size=100)
            
            # Fit and get shrunk covariance
            shrunk_cov = shrinkage.fit(dummy_data).covariance_
            
            # Convert back to correlation matrix
            D = np.sqrt(np.diag(shrunk_cov))
            correlation = shrunk_cov / np.outer(D, D)
            
            return correlation
            
        except Exception as e:
            logger.warning(f"Failed to make matrix positive definite: {e}, using identity")
            return np.eye(matrix.shape[0])
    
    def _run_simulation(
        self,
        sim_inputs: Dict[str, Any],
        correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """Run the actual Monte Carlo simulation."""
        n_players = len(sim_inputs['player_ids'])
        scenarios = np.zeros((self.num_scenarios, n_players))
        
        try:
            # Draw correlated random variables
            if correlation_matrix.shape[0] == n_players:
                # Use multivariate normal for correlated samples
                random_draws = multivariate_normal.rvs(
                    mean=np.zeros(n_players),
                    cov=correlation_matrix,
                    size=self.num_scenarios
                )
            else:
                # Fallback to independent samples
                random_draws = np.random.standard_normal((self.num_scenarios, n_players))
            
            # Convert to scenarios
            for i in range(n_players):
                mean_points = sim_inputs['mean_points'][i]
                uncertainty = sim_inputs['uncertainties'][i]
                expected_minutes = sim_inputs['expected_minutes'][i]
                
                # Base point scenarios
                point_scenarios = mean_points + uncertainty * random_draws[:, i]
                
                # Minutes uncertainty
                minutes_scenarios = self._simulate_minutes_uncertainty(
                    expected_minutes, self.num_scenarios
                )
                
                # Adjust points for minutes (per-90 basis)
                adjusted_scenarios = point_scenarios * (minutes_scenarios / 90)
                
                # Ensure non-negative points
                adjusted_scenarios = np.maximum(0, adjusted_scenarios)
                
                scenarios[:, i] = adjusted_scenarios
            
            logger.debug(f"Generated {self.num_scenarios} scenarios for {n_players} players")
            return scenarios
            
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            return np.zeros((self.num_scenarios, n_players))
    
    def _simulate_minutes_uncertainty(
        self,
        expected_minutes: float,
        num_scenarios: int
    ) -> np.ndarray:
        """Simulate minutes with uncertainty."""
        # Minutes follow a truncated normal distribution
        std_minutes = expected_minutes * self.minutes_uncertainty
        
        # Generate scenarios
        minutes_scenarios = np.random.normal(expected_minutes, std_minutes, num_scenarios)
        
        # Truncate to [0, 90] range
        minutes_scenarios = np.clip(minutes_scenarios, 0, 90)
        
        return minutes_scenarios
    
    def _calculate_summary_statistics(
        self,
        scenarios: np.ndarray,
        predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate summary statistics from scenarios."""
        try:
            summary_stats = predictions[['element_id', 'web_name', 'position', 'team_name']].copy()
            
            # Calculate percentiles and statistics
            summary_stats['mean'] = np.mean(scenarios, axis=0)
            summary_stats['std'] = np.std(scenarios, axis=0)
            summary_stats['p10'] = np.percentile(scenarios, 10, axis=0)
            summary_stats['p25'] = np.percentile(scenarios, 25, axis=0)
            summary_stats['p50'] = np.percentile(scenarios, 50, axis=0)
            summary_stats['p75'] = np.percentile(scenarios, 75, axis=0)
            summary_stats['p90'] = np.percentile(scenarios, 90, axis=0)
            
            # Calculate probability metrics
            summary_stats['prob_over_6'] = np.mean(scenarios >= 6, axis=0)
            summary_stats['prob_over_10'] = np.mean(scenarios >= 10, axis=0)
            summary_stats['prob_blank'] = np.mean(scenarios <= 1, axis=0)
            
            # Risk-adjusted scores
            if self.risk_measure == "cvar":
                summary_stats['cvar'] = self._calculate_cvar(scenarios, self.cvar_alpha)
            
            summary_stats['sharpe_ratio'] = summary_stats['mean'] / (summary_stats['std'] + 0.1)
            
            logger.debug(f"Calculated summary statistics for {len(summary_stats)} players")
            return summary_stats
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
            return pd.DataFrame()
    
    def _calculate_cvar(self, scenarios: np.ndarray, alpha: float) -> np.ndarray:
        """Calculate Conditional Value at Risk (CVaR)."""
        try:
            # Calculate VaR threshold
            var_threshold = np.percentile(scenarios, alpha * 100, axis=0)
            
            # Calculate CVaR (mean of scenarios below VaR threshold)
            cvar_values = np.zeros(scenarios.shape[1])
            
            for i in range(scenarios.shape[1]):
                below_var = scenarios[:, i] <= var_threshold[i]
                if np.any(below_var):
                    cvar_values[i] = np.mean(scenarios[below_var, i])
                else:
                    cvar_values[i] = var_threshold[i]
            
            return cvar_values
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return np.zeros(scenarios.shape[1])
    
    def _calculate_risk_metrics(self, scenarios: np.ndarray) -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics."""
        try:
            risk_metrics = {}
            
            # Portfolio statistics (sum across players for each scenario)
            portfolio_scenarios = np.sum(scenarios, axis=1)
            
            risk_metrics['portfolio_mean'] = np.mean(portfolio_scenarios)
            risk_metrics['portfolio_std'] = np.std(portfolio_scenarios)
            risk_metrics['portfolio_p10'] = np.percentile(portfolio_scenarios, 10)
            risk_metrics['portfolio_p90'] = np.percentile(portfolio_scenarios, 90)
            
            # Risk measures
            if self.risk_measure == "cvar":
                risk_metrics['portfolio_cvar'] = np.mean(
                    portfolio_scenarios[portfolio_scenarios <= np.percentile(portfolio_scenarios, self.cvar_alpha * 100)]
                )
            
            # Concentration risk
            player_contributions = np.mean(scenarios, axis=0)
            total_contribution = np.sum(player_contributions)
            
            if total_contribution > 0:
                contribution_shares = player_contributions / total_contribution
                risk_metrics['concentration_hhi'] = np.sum(contribution_shares ** 2)
                risk_metrics['max_player_share'] = np.max(contribution_shares)
            
            logger.debug("Calculated portfolio risk metrics")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def optimize_portfolio(
        self,
        simulation_results: Dict[str, Any],
        budget_constraints: Dict[str, Any],
        risk_lambda: float = 0.2
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using simulation results.
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            budget_constraints: Budget and position constraints
            risk_lambda: Risk penalty parameter
            
        Returns:
            Optimized portfolio results
        """
        logger.info("Optimizing portfolio with Monte Carlo results")
        
        try:
            scenarios = simulation_results['scenarios']
            summary_stats = simulation_results['summary_stats']
            
            # Mean-CVaR optimization
            expected_returns = summary_stats['mean'].values
            
            if self.risk_measure == "cvar" and 'cvar' in summary_stats.columns:
                risk_measures = summary_stats['cvar'].values
            else:
                risk_measures = summary_stats['std'].values
            
            # Risk-adjusted objective: E[R] - λ * Risk
            objective_scores = expected_returns - risk_lambda * risk_measures
            
            # Add to summary stats
            summary_stats['risk_adjusted_score'] = objective_scores
            summary_stats['risk_rank'] = summary_stats['risk_adjusted_score'].rank(ascending=False)
            
            optimization_results = {
                'risk_adjusted_scores': objective_scores,
                'top_players': summary_stats.nlargest(20, 'risk_adjusted_score'),
                'risk_lambda': risk_lambda,
                'optimization_method': f'mean_{self.risk_measure}'
            }
            
            logger.info("Portfolio optimization completed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {}
    
    def simulate_player_matrix(
        self,
        players_df: pd.DataFrame,
        S: int,
        seed: int,
        minutes_uncertainty: float,
        settings: Optional[Dict] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Return per-player scenarios matrix for captain optimization.
        
        Args:
            players_df: DataFrame with player data and projections
            S: Number of scenarios to generate
            seed: Random seed
            minutes_uncertainty: Minutes uncertainty multiplier
            settings: Optional settings dictionary
            
        Returns:
            Tuple of (scenarios_matrix, player_ids)
            scenarios_matrix: np.ndarray shape (S, n_players) where column j is scenarios for player j
            player_ids: list of player_id in column order
        """
        try:
            if players_df.empty:
                return np.array([]), []
            
            # Extract player info
            player_ids = players_df["element_id"].astype(int).tolist()
            n_players = len(player_ids)
            
            # Set random seed
            rng = np.random.default_rng(seed)
            
            # Get mean projections and uncertainties
            proj_data = players_df.get("proj_points", players_df.get("proj", np.zeros(n_players)))
            if isinstance(proj_data, pd.Series):
                mean_points = proj_data.to_numpy(dtype=float)
            else:
                mean_points = np.array(proj_data, dtype=float)
            
            minutes_data = players_df.get("expected_minutes", np.full(n_players, 75.0))
            if isinstance(minutes_data, pd.Series):
                expected_minutes = minutes_data.to_numpy(dtype=float)
            else:
                expected_minutes = np.array(minutes_data, dtype=float)
            
            # Calculate base uncertainties (use prediction_std if available, else position defaults)
            base_uncertainties = []
            for _, player in players_df.iterrows():
                position = player.get("position", "MID")
                
                if "prediction_std" in players_df.columns:
                    uncertainty = player["prediction_std"]
                else:
                    # Position-based defaults
                    position_uncertainties = {
                        'GK': 1.8, 'DEF': 2.0, 'MID': 2.8, 'FWD': 3.0
                    }
                    uncertainty = position_uncertainties.get(position, 2.5)
                
                base_uncertainties.append(uncertainty)
            
            base_uncertainties = np.array(base_uncertainties, dtype=float)
            
            # Apply league strength and international window adjustments if available
            if settings:
                # International window uncertainty bump
                if "is_intl_window" in players_df.columns:
                    intl_series = players_df["is_intl_window"].fillna(False).astype(bool)
                    if isinstance(intl_series, pd.Series):
                        intl_mask = intl_series.to_numpy()
                    else:
                        intl_mask = np.array(intl_series, dtype=bool)
                    intl_bump = float(settings.get("mc", {}).get("intl_minutes_uncertainty_add", 0.10))
                    base_uncertainties = base_uncertainties * (1.0 + intl_mask * intl_bump)
                
                # Manager rotation uncertainty
                if "manager_rotation_prior" in players_df.columns:
                    rotation_data = players_df["manager_rotation_prior"].fillna(0.0)
                    if isinstance(rotation_data, pd.Series):
                        rotation_prior = rotation_data.to_numpy(dtype=float)
                    else:
                        rotation_prior = np.array(rotation_data, dtype=float)
                    rotation_mult = float(settings.get("mc", {}).get("manager_rotation_sigma_mult", 0.30))
                    base_uncertainties = base_uncertainties * (1.0 + rotation_prior * rotation_mult)
                
                # Low-tier league uncertainty bump
                if "prior_league_uncertainty" in players_df.columns:
                    league_data = players_df["prior_league_uncertainty"].fillna(0.0)
                    if isinstance(league_data, pd.Series):
                        league_uncertainty = league_data.to_numpy(dtype=float)
                    else:
                        league_uncertainty = np.array(league_data, dtype=float)
                    base_uncertainties = base_uncertainties * (1.0 + league_uncertainty)
            
            # Generate correlated scenarios if we have team information
            if "team_id" in players_df.columns or "team_name" in players_df.columns:
                correlation_matrix = self._build_correlation_matrix(players_df)
                
                # Use multivariate normal if correlation matrix is valid
                if correlation_matrix.shape[0] == n_players:
                    try:
                        # Draw correlated random variables
                        random_draws = rng.multivariate_normal(
                            mean=np.zeros(n_players),
                            cov=correlation_matrix,
                            size=S
                        )
                    except np.linalg.LinAlgError:
                        # Fallback to independent draws if correlation matrix is invalid
                        random_draws = rng.standard_normal((S, n_players))
                else:
                    random_draws = rng.standard_normal((S, n_players))
            else:
                # Independent scenarios
                random_draws = rng.standard_normal((S, n_players))
            
            # Generate point scenarios
            scenarios_matrix = np.zeros((S, n_players))
            
            for i in range(n_players):
                # Base point scenarios
                point_scenarios = mean_points[i] + base_uncertainties[i] * random_draws[:, i]
                
                # Apply minutes uncertainty
                minutes_scenarios = rng.normal(
                    expected_minutes[i], 
                    expected_minutes[i] * minutes_uncertainty, 
                    size=S
                )
                minutes_scenarios = np.clip(minutes_scenarios, 0, 90)
                
                # Adjust points for minutes (per-90 basis)
                adjusted_scenarios = point_scenarios * (minutes_scenarios / 90)
                
                # Ensure non-negative points
                scenarios_matrix[:, i] = np.maximum(0, adjusted_scenarios)
            
            logger.debug(f"Generated {S} scenarios for {n_players} players")
            return scenarios_matrix, player_ids
            
        except Exception as e:
            logger.error(f"Error in simulate_player_matrix: {e}")
            # Return empty array with proper shape to avoid indexing errors
            return np.array([]).reshape(0, 0), []

    def simulate_captain_scenarios(
        self,
        simulation_results: Dict[str, Any],
        captain_candidates: List[int]
    ) -> pd.DataFrame:
        """
        Simulate captain choice scenarios.
        
        Args:
            simulation_results: Monte Carlo simulation results
            captain_candidates: List of player IDs to consider for captain
            
        Returns:
            DataFrame with captain analysis
        """
        try:
            scenarios = simulation_results['scenarios']
            summary_stats = simulation_results['summary_stats']
            
            # Get indices of captain candidates
            candidate_indices = []
            for player_id in captain_candidates:
                idx = summary_stats[summary_stats['element_id'] == player_id].index
                if len(idx) > 0:
                    candidate_indices.append(idx[0])
            
            if not candidate_indices:
                logger.warning("No valid captain candidates found")
                return pd.DataFrame()
            
            captain_results = []
            
            for idx in candidate_indices:
                player_info = summary_stats.iloc[idx]
                player_scenarios = scenarios[:, idx]
                
                # Captain gets 2x points
                captain_scenarios = player_scenarios * 2
                
                captain_analysis = {
                    'element_id': player_info['element_id'],
                    'web_name': player_info['web_name'],
                    'position': player_info['position'],
                    'captain_mean': np.mean(captain_scenarios),
                    'captain_std': np.std(captain_scenarios),
                    'captain_p10': np.percentile(captain_scenarios, 10),
                    'captain_p50': np.percentile(captain_scenarios, 50),
                    'captain_p90': np.percentile(captain_scenarios, 90),
                    'prob_top_captain': 0,  # Will be calculated below
                    'prob_over_12': np.mean(captain_scenarios >= 12),
                    'prob_over_20': np.mean(captain_scenarios >= 20)
                }
                
                captain_results.append(captain_analysis)
            
            captain_df = pd.DataFrame(captain_results)
            
            # Calculate probability of being top captain in each scenario
            if len(candidate_indices) > 1:
                captain_scenarios_matrix = scenarios[:, candidate_indices] * 2
                
                for i, idx in enumerate(candidate_indices):
                    # Count scenarios where this player is the highest scoring captain
                    is_top = np.argmax(captain_scenarios_matrix, axis=1) == i
                    captain_df.loc[i, 'prob_top_captain'] = np.mean(is_top)
            else:
                captain_df['prob_top_captain'] = 1.0
            
            # Sort by expected captain points
            captain_df = captain_df.sort_values('captain_mean', ascending=False)
            
            logger.info(f"Analyzed {len(captain_df)} captain candidates")
            return captain_df
            
        except Exception as e:
            logger.error(f"Error in captain simulation: {e}")
            return pd.DataFrame()


# Standalone functions for easy importing
def simulate_player_matrix(
    players_df: pd.DataFrame, 
    S: int, 
    seed: int, 
    minutes_uncertainty: float, 
    settings: Optional[Dict] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Standalone function to generate per-player scenarios matrix.
    
    Args:
        players_df: DataFrame with player data and projections
        S: Number of scenarios to generate
        seed: Random seed
        minutes_uncertainty: Minutes uncertainty multiplier
        settings: Optional settings dictionary
        
    Returns:
        Tuple of (scenarios_matrix, player_ids)
    """
    simulator = MonteCarloSimulator()
    return simulator.simulate_player_matrix(players_df, S, seed, minutes_uncertainty, settings)


def simulate_points(team_df: pd.DataFrame, S: int, seed: int, minutes_uncertainty: float, settings: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Backward-compatible simulate_points function.
    
    Args:
        team_df: DataFrame with team players
        S: Number of scenarios
        seed: Random seed
        minutes_uncertainty: Minutes uncertainty
        settings: Optional settings
        
    Returns:
        Dictionary with 'scenarios' key containing team sum scenarios
    """
    scenarios_matrix, player_ids = simulate_player_matrix(team_df, S, seed, minutes_uncertainty, settings)
    
    if scenarios_matrix.size == 0 or not isinstance(scenarios_matrix, np.ndarray):
        return {"scenarios": np.array([])}
    
    # Sum across players for team total
    team_scenarios = np.sum(scenarios_matrix, axis=1)
    
    return {"scenarios": team_scenarios}


def run_mc_simulation(prediction_data: pd.DataFrame, settings: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for player predictions.
    
    Args:
        prediction_data: DataFrame with player predictions
        settings: Optional simulation settings
        
    Returns:
        Simulation results
    """
    try:
        if prediction_data.empty:
            logger.warning("No prediction data provided for MC simulation")
            return {"error": "No prediction data"}
        
        # Default settings
        default_settings = {
            "num_scenarios": 2000,
            "seed": 42,
            "minutes_uncertainty": 0.10
        }
        
        if settings:
            default_settings.update(settings)
        
        # Apply a conservative minutes floor for likely starters
        try:
            if {'expected_minutes', 'avail_prob', 'rotation_risk'}.issubset(set(prediction_data.columns)):
                starters_mask = (prediction_data['avail_prob'] >= 0.8) | (prediction_data['rotation_risk'] <= 0.2)
                prediction_data.loc[starters_mask, 'expected_minutes'] = prediction_data.loc[starters_mask, 'expected_minutes'].clip(lower=70)
        except Exception:
            pass

        # Run simulation
        scenarios_matrix, player_ids = simulate_player_matrix(
            prediction_data,
            default_settings["num_scenarios"],
            default_settings["seed"],
            default_settings["minutes_uncertainty"],
            default_settings
        )
        
        if scenarios_matrix.size == 0:
            return {"error": "Failed to generate scenarios"}
        
        # Calculate summary statistics
        player_summaries = []
        for i, player_id in enumerate(player_ids):
            player_scenarios = scenarios_matrix[:, i]
            
            # Check if player exists in prediction_data and handle missing players safely
            player_filtered = prediction_data[prediction_data['element_id'] == player_id]
            if player_filtered.empty:
                # Create safe defaults for missing player
                player_data = {
                    'web_name': 'Unknown',
                    'position': 'Unknown'
                }
                logger.warning(f"Player {player_id} not found in prediction data, using defaults")
            else:
                player_data = player_filtered.iloc[0]
            
            summary = {
                'element_id': player_id,
                'web_name': player_data.get('web_name', 'Unknown'),
                'position': player_data.get('position', 'Unknown'),
                'mean_points': np.mean(player_scenarios),
                'std_points': np.std(player_scenarios),
                'p10': np.percentile(player_scenarios, 10),
                'p50': np.percentile(player_scenarios, 50),
                'p90': np.percentile(player_scenarios, 90),
                'prob_double_digits': np.mean(player_scenarios >= 10),
                'prob_15_plus': np.mean(player_scenarios >= 15),
                # Add essential fields for team optimization
                'now_cost': player_data.get('now_cost', 0),
                'proj_points': np.mean(player_scenarios),  # Same as mean_points for optimizer
                'team_id': player_data.get('team_id', 0),
                'team_name': player_data.get('team_name', 'Unknown'),
                # Add rolling features for form analysis
                'points_r3': player_data.get('points_r3', 0),
                'points_r5': player_data.get('points_r5', 0),
                'points_r8': player_data.get('points_r8', 0),
                'goals_r3': player_data.get('goals_r3', 0),
                'goals_r5': player_data.get('goals_r5', 0),
                'goals_r8': player_data.get('goals_r8', 0),
                'assists_r3': player_data.get('assists_r3', 0),
                'assists_r5': player_data.get('assists_r5', 0),
                'assists_r8': player_data.get('assists_r8', 0),
                'minutes_r3': player_data.get('minutes_r3', 0),
                'minutes_r5': player_data.get('minutes_r5', 0),
                'minutes_r8': player_data.get('minutes_r8', 0)
            }
            player_summaries.append(summary)
        
        results_df = pd.DataFrame(player_summaries)
        
        logger.info(f"MC simulation completed: {len(player_summaries)} players, {default_settings['num_scenarios']} scenarios")
        
        return {
            "success": True,
            "scenarios_matrix": scenarios_matrix,
            "player_ids": player_ids,
            "player_summaries": results_df,
            "settings": default_settings
        }
        
    except Exception as e:
        logger.error(f"MC simulation failed: {e}")
        return {"error": str(e)}
