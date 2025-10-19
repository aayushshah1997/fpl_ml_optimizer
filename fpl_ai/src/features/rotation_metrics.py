"""
Rotation metrics computation module.

Computes empirical rotation metrics from historical match data across all competitions
to generate data-driven rotation priors for managers.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Set
from ..common.config import get_config, get_logger
from ..providers.fbrapi_client import FBRAPIClient

logger = get_logger(__name__)


def _starts_variance(starts_series: pd.Series) -> float:
    """
    Variance of starts across squad players, normalized to [0,1] by dividing by max(starts)^2.
    
    Args:
        starts_series: Series of start counts per player
        
    Returns:
        Normalized variance between 0 and 1
    """
    if starts_series.empty or len(starts_series) <= 1:
        return 0.0
    
    var = float(np.var(starts_series.values))
    max_starts = max(float(starts_series.max()), 1.0)
    
    # Normalize by theoretical maximum variance
    normalized_var = var / (max_starts ** 2)
    return float(min(1.0, normalized_var))


def _median_minutes_shortfall(minutes_series: pd.Series) -> float:
    """
    Compute (90 - median_minutes)/90 clipped to [0,1].
    
    Args:
        minutes_series: Series of minutes played per match
        
    Returns:
        Shortfall ratio between 0 and 1
    """
    if minutes_series.empty:
        return 0.0
    
    # Filter out zero minutes for median calculation
    non_zero_minutes = minutes_series[minutes_series > 0]
    if non_zero_minutes.empty:
        return 1.0  # Maximum shortfall if no one plays
    
    median_mins = float(np.median(non_zero_minutes.values))
    shortfall = max(0.0, min(1.0, (90.0 - median_mins) / 90.0))
    return float(shortfall)


def _bench_rate(lineups_df: pd.DataFrame) -> float:
    """
    Share of attackers (MID/FWD) with 0 minutes among those who appeared in previous matches.
    This is a proxy for rotation-induced DNPs among fit players.
    
    Args:
        lineups_df: DataFrame with player lineups across matches
        
    Returns:
        Bench rate between 0 and 1
    """
    if lineups_df.empty or 'match_seq' not in lineups_df.columns:
        return 0.0
    
    df = lineups_df.sort_values('match_seq')
    rates = []
    prev_match_players = None
    
    for match_seq, match_group in df.groupby('match_seq'):
        if prev_match_players is None:
            prev_match_players = set(match_group['player_id'].astype(str))
            continue
        
        # Current match data
        current_match = match_group.copy()
        # Ensure we get a Series before calling fillna
        minutes_val = current_match.get('minutes', 0)
        numeric_series = pd.to_numeric(minutes_val, errors='coerce')
        if isinstance(numeric_series, pd.Series):
            current_match['minutes'] = numeric_series.fillna(0)
        else:
            # If it's a scalar, use it directly
            current_match['minutes'] = numeric_series
        
        # Focus on attackers (MID/FWD)
        attackers = current_match[
            current_match.get('position', '').astype(str).str.contains('MID|FWD', na=False)
        ]
        
        if attackers.empty:
            prev_match_players = set(current_match['player_id'].astype(str))
            continue
        
        # Players who played in previous match and are attackers this match
        prev_attackers = attackers[
            attackers['player_id'].astype(str).isin(prev_match_players)
        ]
        
        if len(prev_attackers) == 0:
            prev_match_players = set(current_match['player_id'].astype(str))
            continue
        
        # Count those with 0 minutes (benched)
        benched_count = len(prev_attackers[prev_attackers['minutes'] == 0])
        bench_rate = benched_count / len(prev_attackers)
        rates.append(bench_rate)
        
        prev_match_players = set(current_match['player_id'].astype(str))
    
    return float(np.mean(rates)) if rates else 0.0


def _xi_change_pct(lineups_per_match: List[Set]) -> float:
    """
    Compute percentage of consecutive matches where starting XI changed by >=3 players.
    
    Args:
        lineups_per_match: List of sets, each containing player IDs who started
        
    Returns:
        Percentage of matches with significant XI changes
    """
    if len(lineups_per_match) < 2:
        return 0.0
    
    changes = []
    for i in range(1, len(lineups_per_match)):
        # Count symmetric difference (players who started in one match but not the other)
        change_count = len(lineups_per_match[i].symmetric_difference(lineups_per_match[i-1]))
        changes.append(change_count)
    
    if not changes:
        return 0.0
    
    # Percentage of matches with 3+ changes
    significant_changes = sum(1 for change in changes if change >= 3)
    return float(significant_changes / len(changes))


def _team_lineup_table(team_id: str, season: str, client: FBRAPIClient) -> Tuple[pd.DataFrame, List[Set]]:
    """
    Return wide table with per-match lineup data and a list of starting XI sets.
    
    Args:
        team_id: Team ID
        season: Season string
        client: FBR API client instance
        
    Returns:
        Tuple of (lineup_dataframe, list_of_starting_xi_sets)
    """
    try:
        matches = client.get_team_matches_all_comps(team_id, season)
        if matches.empty:
            logger.debug(f"No matches found for team {team_id} in season {season}")
            return pd.DataFrame(), []
        
        # Sort matches by date
        if 'date' in matches.columns:
            matches = matches.sort_values('date').reset_index(drop=True)
        
        rows = []
        start_sets = []
        
        for i, match_row in matches.iterrows():
            match_id = str(match_row.get('match_id', match_row.get('id', '')))
            
            try:
                lineup = client.get_match_lineups(match_id)
                if lineup.empty:
                    continue
                
                # Standardize column names
                lineup['started'] = lineup.get('started', lineup.get('is_start', False)).astype(bool)
                # Ensure we get a Series before calling fillna
                minutes_val = lineup.get('minutes', 0)
                numeric_series = pd.to_numeric(minutes_val, errors='coerce')
                if isinstance(numeric_series, pd.Series):
                    lineup['minutes'] = numeric_series.fillna(0)
                else:
                    # If it's a scalar, use it directly
                    lineup['minutes'] = numeric_series
                lineup['player_id'] = lineup.get('player_id', lineup.get('id', '')).astype(str)
                lineup['position'] = lineup.get('position', '').astype(str)
                lineup['match_seq'] = i
                lineup['match_id'] = match_id
                
                rows.append(lineup)
                
                # Extract starting XI player IDs
                starters = lineup[lineup['started']]['player_id'].astype(str).tolist()
                start_sets.append(set(starters))
                
                logger.debug(f"Processed lineup for match {match_id}: {len(starters)} starters")
                
            except Exception as e:
                logger.warning(f"Failed to get lineup for match {match_id}: {e}")
                continue
        
        if not rows:
            logger.warning(f"No lineup data found for team {team_id} in season {season}")
            return pd.DataFrame(), []
        
        lineups_df = pd.concat(rows, ignore_index=True)
        logger.info(f"Retrieved lineup data for {len(start_sets)} matches for team {team_id}")
        return lineups_df, start_sets
        
    except Exception as e:
        logger.error(f"Failed to get lineup table for team {team_id}, season {season}: {e}")
        return pd.DataFrame(), []


def compute_rotation_metrics_for_team(team_id: str, season: str) -> Dict:
    """
    Compute comprehensive rotation metrics for a team in a given season.
    
    Args:
        team_id: Team ID
        season: Season string
        
    Returns:
        Dictionary with rotation metrics:
        - xi_change_pct: Percentage of matches with 3+ XI changes
        - starts_variance: Normalized variance in player starts
        - median_minutes_shortfall: Minutes shortfall metric
        - bench_rate: Rate of attackers being benched
        - n_matches: Number of matches analyzed
    """
    from ..common.config import get_config
    config = get_config()
    
    # Return default metrics if FBR API is disabled
    if not config.get("fbrapi.enabled", False):
        logger.info(f"FBR API disabled, returning default rotation metrics for team {team_id}")
        return {
            "xi_change_pct": 0.05,
            "starts_variance": 0.05,
            "median_minutes_shortfall": 0.05,
            "bench_rate": 0.05,
            "n_matches": 0
        }
    
    client = FBRAPIClient()
    
    lineups_df, start_sets = _team_lineup_table(team_id, season, client)
    
    if lineups_df.empty:
        return {
            "xi_change_pct": 0.0,
            "starts_variance": 0.0,
            "median_minutes_shortfall": 0.0,
            "bench_rate": 0.0,
            "n_matches": 0
        }
    
    # Compute individual metrics
    xi_pct = _xi_change_pct(start_sets)
    
    # Starts per player
    starts_per_player = lineups_df.groupby('player_id')['started'].sum()
    starts_var = _starts_variance(starts_per_player)
    
    # Minutes distribution among players who played
    minutes_series = lineups_df[lineups_df['minutes'] > 0]['minutes']
    med_shortfall = _median_minutes_shortfall(minutes_series)
    
    # Bench rate for attackers
    bench = _bench_rate(lineups_df)
    
    n_matches = int(lineups_df['match_id'].nunique())
    
    metrics = {
        "xi_change_pct": float(xi_pct),
        "starts_variance": float(starts_var),
        "median_minutes_shortfall": float(med_shortfall),
        "bench_rate": float(bench),
        "n_matches": n_matches
    }
    
    logger.info(f"Computed rotation metrics for team {team_id}, season {season}: {metrics}")
    return metrics


def map_metrics_to_prior(metrics: Dict) -> float:
    """
    Map empirical rotation metrics to a rotation prior using configured weights.
    
    Args:
        metrics: Dictionary of rotation metrics
        
    Returns:
        Rotation prior between floor and cap values
    """
    config = get_config()
    weights = config.get("rotation_engine.weights", {})
    floor = float(config.get("rotation_engine.floor", 0.03))
    cap = float(config.get("rotation_engine.cap", 0.30))
    
    # Weighted sum of normalized metrics
    prior = (
        weights.get("xi_change_pct", 0.20) * metrics.get("xi_change_pct", 0.0) +
        weights.get("starts_variance", 0.10) * metrics.get("starts_variance", 0.0) +
        weights.get("median_minutes_shortfall", 0.05) * metrics.get("median_minutes_shortfall", 0.0) +
        weights.get("bench_rate", 0.05) * metrics.get("bench_rate", 0.0)
    )
    
    # Apply floor and cap
    prior = max(floor, min(cap, prior))
    
    logger.debug(f"Mapped metrics to prior: {prior:.3f} (floor: {floor}, cap: {cap})")
    return float(prior)

