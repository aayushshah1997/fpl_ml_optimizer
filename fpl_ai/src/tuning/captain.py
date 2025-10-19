"""
Captain selection policy optimization.

Implements different captain selection strategies:
- Mean-based: Select captain with highest expected points
- CVaR-based: Select captain with best downside protection (tail risk)
- Mixed: Weighted combination of mean and CVaR considerations
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from ..common.config import get_logger

logger = get_logger(__name__)


def cvar(samples: np.ndarray, alpha: float = 0.2) -> float:
    """
    Calculate Conditional Value at Risk (CVaR).
    
    Args:
        samples: Array of scenario values
        alpha: Confidence level (e.g., 0.2 for 20th percentile)
        
    Returns:
        CVaR value (mean of worst alpha% of outcomes)
    """
    if samples.size == 0:
        return 0.0
    
    # Find the alpha quantile threshold
    threshold = np.quantile(samples, alpha)
    
    # Get tail samples (worst alpha% of outcomes)
    tail_samples = samples[samples <= threshold]
    
    # Return mean of tail, or overall mean if no tail samples
    return float(np.mean(tail_samples)) if tail_samples.size > 0 else float(np.mean(samples))


def choose_captain(
    xi_df: pd.DataFrame, 
    scenarios_matrix: np.ndarray, 
    player_ids: List[int], 
    policy: str = "mix", 
    alpha: float = 0.2, 
    mix_lambda: float = 0.6, 
    topN: int = 5
) -> Tuple[int, int]:
    """
    Choose captain and vice-captain using specified policy.
    
    Args:
        xi_df: DataFrame with starting XI player data
        scenarios_matrix: Monte Carlo scenarios matrix (S x n_players)
        player_ids: List of player IDs corresponding to scenario columns
        policy: Captain selection policy ("mean", "cvar", "mix")
        alpha: CVaR confidence level for tail risk assessment
        mix_lambda: Weight on mean vs CVaR in mixed policy (0..1)
        topN: Number of top candidates to consider for speed/stability
        
    Returns:
        Tuple of (captain_idx, vice_captain_idx) w.r.t xi_df row order
    """
    try:
        if xi_df.empty or scenarios_matrix.size == 0:
            logger.warning("Empty data provided to captain selection")
            return 0, 0
        
        n_players = len(xi_df)
        
        if scenarios_matrix.shape[1] != n_players:
            logger.error(f"Scenarios matrix columns ({scenarios_matrix.shape[1]}) != players ({n_players})")
            return 0, 0
        
        # Get mean projections for initial filtering
        proj_data = xi_df.get("proj_points", xi_df.get("proj", np.zeros(n_players)))
        if isinstance(proj_data, pd.Series):
            mean_projections = proj_data.to_numpy(dtype=float)
        else:
            # If it's a scalar or array, convert to numpy array
            mean_projections = np.array(proj_data, dtype=float)
            if mean_projections.size == 1:
                mean_projections = np.full(n_players, mean_projections.item())
        
        # Restrict to top N candidates by mean for computational efficiency
        top_indices = np.argsort(-mean_projections)[:min(topN, n_players)]
        
        if len(top_indices) == 0:
            return 0, 0
        
        # Calculate selection criteria for top candidates
        candidate_scores = []
        
        for idx in top_indices:
            # Ensure scenarios_matrix is a numpy array before indexing
            if not isinstance(scenarios_matrix, np.ndarray):
                logger.error(f"scenarios_matrix is not a numpy array: {type(scenarios_matrix)}")
                return 0, 0
            
            if scenarios_matrix.size == 0:
                logger.error("scenarios_matrix is empty")
                return 0, 0
                
            player_scenarios = scenarios_matrix[:, idx]
            # Ensure player_scenarios is a numpy array
            if not isinstance(player_scenarios, np.ndarray):
                player_scenarios = np.array(player_scenarios, dtype=float)
            mean_score = float(np.mean(player_scenarios))
            cvar_score = cvar(player_scenarios, alpha)
            
            if policy == "mean":
                final_score = mean_score
            elif policy == "cvar":
                final_score = cvar_score
            elif policy == "mix":
                final_score = mix_lambda * mean_score + (1.0 - mix_lambda) * cvar_score
            else:
                logger.warning(f"Unknown captain policy '{policy}', using 'mix'")
                final_score = 0.6 * mean_score + 0.4 * cvar_score
            
            candidate_scores.append({
                'idx': idx,
                'score': final_score,
                'mean': mean_score,
                'cvar': cvar_score
            })
        
        # Sort by score (descending)
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select captain (best score)
        captain_idx = candidate_scores[0]['idx']
        
        # Select vice-captain (second best, but not same as captain)
        vice_captain_idx = captain_idx  # Fallback
        
        for candidate in candidate_scores[1:]:
            if candidate['idx'] != captain_idx:
                vice_captain_idx = candidate['idx']
                break
        
        logger.debug(f"Captain selection ({policy}): Captain idx={captain_idx}, Vice idx={vice_captain_idx}")
        
        return int(captain_idx), int(vice_captain_idx)
        
    except Exception as e:
        logger.error(f"Error in captain selection: {e}")
        return 0, min(1, len(xi_df) - 1)


def evaluate_captain_policies(
    xi_df: pd.DataFrame,
    scenarios_matrix: np.ndarray,
    player_ids: List[int],
    policies: Optional[List[str]] = None,
    alpha: float = 0.2,
    mix_lambdas: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Evaluate multiple captain selection policies for comparison.
    
    Args:
        xi_df: DataFrame with starting XI player data
        scenarios_matrix: Monte Carlo scenarios matrix
        player_ids: List of player IDs
        policies: List of policies to evaluate
        alpha: CVaR confidence level
        mix_lambdas: List of mix_lambda values to test for mixed policy
        
    Returns:
        DataFrame with policy evaluation results
    """
    if policies is None:
        policies = ["mean", "cvar", "mix"]
    
    if mix_lambdas is None:
        mix_lambdas = [0.4, 0.5, 0.6, 0.7, 0.8]
    
    results = []
    
    try:
        # Evaluate basic policies
        for policy in policies:
            if policy == "mix":
                # Test different mix_lambda values
                for mix_lambda in mix_lambdas:
                    captain_idx, vice_idx = choose_captain(
                        xi_df, scenarios_matrix, player_ids, 
                        policy="mix", alpha=alpha, mix_lambda=mix_lambda
                    )
                    
                    # Calculate performance metrics
                    captain_scenarios = scenarios_matrix[:, captain_idx] * 2  # Captain gets 2x points
                    vice_scenarios = scenarios_matrix[:, vice_idx] * 2       # Vice gets 2x if captain blanks
                    
                    results.append({
                        'policy': f"mix_{mix_lambda:.1f}",
                        'captain_idx': captain_idx,
                        'vice_idx': vice_idx,
                        'captain_mean': float(np.mean(captain_scenarios)),
                        'captain_cvar': cvar(captain_scenarios, alpha),
                        'captain_std': float(np.std(captain_scenarios)),
                        'mix_lambda': mix_lambda
                    })
            else:
                captain_idx, vice_idx = choose_captain(
                    xi_df, scenarios_matrix, player_ids, 
                    policy=policy, alpha=alpha
                )
                
                captain_scenarios = scenarios_matrix[:, captain_idx] * 2
                
                results.append({
                    'policy': policy,
                    'captain_idx': captain_idx,
                    'vice_idx': vice_idx,
                    'captain_mean': float(np.mean(captain_scenarios)),
                    'captain_cvar': cvar(captain_scenarios, alpha),
                    'captain_std': float(np.std(captain_scenarios)),
                    'mix_lambda': None
                })
        
        return pd.DataFrame(results).sort_values('captain_mean', ascending=False)
        
    except Exception as e:
        logger.error(f"Error evaluating captain policies: {e}")
        return pd.DataFrame()


def captain_policy_analysis(
    xi_df: pd.DataFrame,
    scenarios_matrix: np.ndarray,
    player_ids: List[int],
    alpha: float = 0.2
) -> dict:
    """
    Comprehensive analysis of captain selection for different policies.
    
    Args:
        xi_df: DataFrame with starting XI player data
        scenarios_matrix: Monte Carlo scenarios matrix
        player_ids: List of player IDs
        alpha: CVaR confidence level
        
    Returns:
        Dictionary with comprehensive captain analysis
    """
    try:
        analysis = {}
        
        # Basic player statistics
        n_players = len(xi_df)
        n_scenarios = scenarios_matrix.shape[0]
        
        # Calculate per-player captain statistics
        player_stats = []
        
        for i, (_, player) in enumerate(xi_df.iterrows()):
            player_scenarios = scenarios_matrix[:, i]
            captain_scenarios = player_scenarios * 2  # Captain bonus
            
            player_stats.append({
                'player_idx': i,
                'player_name': player.get('web_name', f'Player_{i}'),
                'position': player.get('position', 'UNKNOWN'),
                'team': player.get('team_name', 'UNKNOWN'),
                'proj_points': float(player.get('proj_points', 0)),
                'captain_mean': float(np.mean(captain_scenarios)),
                'captain_cvar': cvar(captain_scenarios, alpha),
                'captain_std': float(np.std(captain_scenarios)),
                'prob_best_captain': 0.0,  # Will calculate below
                'prob_top3_captain': 0.0
            })
        
        player_stats_df = pd.DataFrame(player_stats)
        
        # Calculate probability of being best captain in each scenario
        captain_scenarios_matrix = scenarios_matrix * 2  # All players as captain
        
        for scenario in range(n_scenarios):
            best_captain_idx = np.argmax(captain_scenarios_matrix[scenario, :])
            player_stats_df.loc[best_captain_idx, 'prob_best_captain'] += 1
            
            # Top 3 captain performances in this scenario
            top3_indices = np.argsort(-captain_scenarios_matrix[scenario, :])[:3]
            for idx in top3_indices:
                player_stats_df.loc[idx, 'prob_top3_captain'] += 1
        
        # Convert to probabilities
        player_stats_df['prob_best_captain'] /= n_scenarios
        player_stats_df['prob_top3_captain'] /= n_scenarios
        
        # Policy comparison
        policy_comparison = evaluate_captain_policies(
            xi_df, scenarios_matrix, player_ids, alpha=alpha
        )
        
        # Risk-return analysis
        captain_means = player_stats_df['captain_mean'].values
        captain_cvars = player_stats_df['captain_cvar'].values
        captain_stds = player_stats_df['captain_std'].values
        
        # Correlation between mean and risk measures
        mean_cvar_corr = np.corrcoef(captain_means, captain_cvars)[0, 1] if len(captain_means) > 1 else 0
        mean_std_corr = np.corrcoef(captain_means, captain_stds)[0, 1] if len(captain_means) > 1 else 0
        
        analysis = {
            'summary': {
                'n_players': n_players,
                'n_scenarios': n_scenarios,
                'alpha': alpha,
                'mean_cvar_correlation': float(mean_cvar_corr),
                'mean_std_correlation': float(mean_std_corr)
            },
            'player_stats': player_stats_df,
            'policy_comparison': policy_comparison,
            'recommendations': {
                'best_mean_captain': player_stats_df.loc[player_stats_df['captain_mean'].idxmax()]['player_name'],
                'best_cvar_captain': player_stats_df.loc[player_stats_df['captain_cvar'].idxmax()]['player_name'],
                'most_consistent': player_stats_df.loc[player_stats_df['captain_std'].idxmin()]['player_name'],
                'most_likely_best': player_stats_df.loc[player_stats_df['prob_best_captain'].idxmax()]['player_name']
            }
        }
        
        logger.info(f"Captain analysis completed for {n_players} players, {n_scenarios} scenarios")
        return analysis
        
    except Exception as e:
        logger.error(f"Error in captain policy analysis: {e}")
        return {'error': str(e)}


def optimal_mix_lambda(
    xi_df: pd.DataFrame,
    scenarios_matrix: np.ndarray,
    player_ids: List[int],
    alpha: float = 0.2,
    objective: str = "expected_utility"
) -> float:
    """
    Find optimal mix_lambda value for mixed captain policy.
    
    Args:
        xi_df: DataFrame with starting XI player data
        scenarios_matrix: Monte Carlo scenarios matrix
        player_ids: List of player IDs
        alpha: CVaR confidence level
        objective: Optimization objective ("expected_utility", "sharpe", "cvar")
        
    Returns:
        Optimal mix_lambda value
    """
    try:
        lambda_values = np.linspace(0.0, 1.0, 21)  # Test 21 values from 0 to 1
        best_lambda = 0.6  # Default
        best_score = -np.inf
        
        for mix_lambda in lambda_values:
            captain_idx, _ = choose_captain(
                xi_df, scenarios_matrix, player_ids,
                policy="mix", alpha=alpha, mix_lambda=mix_lambda
            )
            
            captain_scenarios = scenarios_matrix[:, captain_idx] * 2
            
            if objective == "expected_utility":
                # Simple utility: mean - risk_penalty * CVaR
                mean_score = np.mean(captain_scenarios)
                cvar_score = cvar(captain_scenarios, alpha)
                score = mean_score - 0.2 * (mean_score - cvar_score)  # Risk penalty
            elif objective == "sharpe":
                # Sharpe-like ratio
                mean_score = np.mean(captain_scenarios)
                std_score = np.std(captain_scenarios)
                score = mean_score / (std_score + 1e-9)
            elif objective == "cvar":
                # Pure CVaR maximization
                score = cvar(captain_scenarios, alpha)
            else:
                # Default to mean
                score = np.mean(captain_scenarios)
            
            if score > best_score:
                best_score = score
                best_lambda = mix_lambda
        
        logger.debug(f"Optimal mix_lambda: {best_lambda:.2f} (objective: {objective}, score: {best_score:.2f})")
        return float(best_lambda)
        
    except Exception as e:
        logger.error(f"Error finding optimal mix_lambda: {e}")
        return 0.6  # Fallback
