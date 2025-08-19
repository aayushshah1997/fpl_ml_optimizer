"""
Head-to-head (H2H) features for fixture-specific performance.

Calculates how players perform against specific opponents,
with Bayesian shrinkage for small sample sizes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


def calculate_h2h_features(
    match_data: pd.DataFrame,
    min_h2h_matches: int = 3,
    shrinkage_factor: float = 0.3
) -> pd.DataFrame:
    """
    Calculate head-to-head performance features with Bayesian shrinkage.
    
    Args:
        match_data: DataFrame with player match data
        min_h2h_matches: Minimum matches required for H2H stats
        shrinkage_factor: Shrinkage towards global mean (0-1)
        
    Returns:
        DataFrame with H2H features
    """
    if match_data.empty:
        return pd.DataFrame()
    
    # Calculate raw H2H statistics
    h2h_stats = _calculate_raw_h2h_stats(match_data, min_h2h_matches)
    
    # Apply Bayesian shrinkage
    h2h_shrunk = _apply_bayesian_shrinkage(h2h_stats, match_data, shrinkage_factor)
    
    # Calculate opponent-specific features
    h2h_features = _calculate_opponent_features(h2h_shrunk, match_data)
    
    logger.debug(f"Calculated H2H features for {len(h2h_features)} player-opponent pairs")
    return h2h_features


def _calculate_raw_h2h_stats(df: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Calculate raw head-to-head statistics."""
    if 'element_id' not in df.columns or 'opponent_id' not in df.columns:
        logger.warning("Missing required columns for H2H calculation")
        return pd.DataFrame()
    
    # Group by player and opponent
    h2h_groups = df.groupby(['element_id', 'opponent_id'])
    
    h2h_stats = []
    
    for (player_id, opponent_id), group in h2h_groups:
        if len(group) < min_matches:
            continue
        
        # Calculate statistics against this opponent
        stats = {
            'element_id': player_id,
            'opponent_id': opponent_id,
            'h2h_matches': len(group),
            'h2h_points_avg': group['points'].mean(),
            'h2h_points_std': group['points'].std(),
            'h2h_goals_avg': group.get('goals_scored', group.get('goals', 0)).mean(),
            'h2h_assists_avg': group.get('assists', 0).mean(),
            'h2h_minutes_avg': group.get('minutes', 0).mean(),
            'h2h_bonus_avg': group.get('bonus', 0).mean(),
            'h2h_bps_avg': group.get('bps', 0).mean(),
            'h2h_clean_sheets_rate': group.get('clean_sheets', 0).mean(),
            'h2h_home_points': group[group.get('home_away') == 'H']['points'].mean() if 'home_away' in group.columns else np.nan,
            'h2h_away_points': group[group.get('home_away') == 'A']['points'].mean() if 'home_away' in group.columns else np.nan
        }
        
        # Handle NaN values
        for key, value in stats.items():
            if pd.isna(value):
                stats[key] = 0.0
        
        h2h_stats.append(stats)
    
    return pd.DataFrame(h2h_stats)


def _apply_bayesian_shrinkage(
    h2h_stats: pd.DataFrame,
    all_data: pd.DataFrame,
    shrinkage_factor: float
) -> pd.DataFrame:
    """Apply Bayesian shrinkage to H2H statistics."""
    if h2h_stats.empty:
        return h2h_stats
    
    shrunk_stats = h2h_stats.copy()
    
    # Calculate global means for shrinkage targets
    global_means = {
        'points': all_data.get('points', 0).mean(),
        'goals': all_data.get('goals_scored', all_data.get('goals', 0)).mean(),
        'assists': all_data.get('assists', 0).mean(),
        'minutes': all_data.get('minutes', 0).mean(),
        'bonus': all_data.get('bonus', 0).mean(),
        'bps': all_data.get('bps', 0).mean(),
        'clean_sheets': all_data.get('clean_sheets', 0).mean()
    }
    
    # Apply shrinkage to key statistics
    shrinkage_columns = [
        ('h2h_points_avg', global_means['points']),
        ('h2h_goals_avg', global_means['goals']),
        ('h2h_assists_avg', global_means['assists']),
        ('h2h_minutes_avg', global_means['minutes']),
        ('h2h_bonus_avg', global_means['bonus']),
        ('h2h_bps_avg', global_means['bps']),
        ('h2h_clean_sheets_rate', global_means['clean_sheets'])
    ]
    
    for h2h_col, global_mean in shrinkage_columns:
        if h2h_col in shrunk_stats.columns:
            # Shrinkage: h2h_shrunk = (1-λ) * h2h_raw + λ * global_mean
            # Shrinkage factor increases with fewer matches
            effective_shrinkage = shrinkage_factor * (10 / (shrunk_stats['h2h_matches'] + 10))
            
            shrunk_stats[h2h_col + '_shrunk'] = (
                (1 - effective_shrinkage) * shrunk_stats[h2h_col] +
                effective_shrinkage * global_mean
            )
    
    return shrunk_stats


def _calculate_opponent_features(h2h_stats: pd.DataFrame, all_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional opponent-specific features."""
    if h2h_stats.empty:
        return h2h_stats
    
    enhanced_stats = h2h_stats.copy()
    
    # Calculate opponent defensive strength (affects player scoring)
    if 'team_goals_against' in all_data.columns:
        opponent_defense = all_data.groupby(['team_id', 'gameweek'])['team_goals_against'].mean().groupby('team_id').mean()
        enhanced_stats['opponent_defense_strength'] = enhanced_stats['opponent_id'].map(opponent_defense).fillna(1.0)
    
    # Player's improvement/decline against this opponent over time
    for (player_id, opponent_id), group_data in all_data.groupby(['element_id', 'opponent_id']):
        if len(group_data) >= 3:
            # Calculate trend over time
            group_data = group_data.sort_values('kickoff_time')
            x = np.arange(len(group_data))
            y = group_data['points'].values
            
            if np.std(y) > 0:
                trend = np.corrcoef(x, y)[0, 1] * np.std(y)
            else:
                trend = 0
            
            # Update H2H stats with trend
            mask = (enhanced_stats['element_id'] == player_id) & (enhanced_stats['opponent_id'] == opponent_id)
            enhanced_stats.loc[mask, 'h2h_trend'] = trend
    
    # Fill missing trend values
    enhanced_stats['h2h_trend'] = enhanced_stats.get('h2h_trend', 0).fillna(0)
    
    return enhanced_stats


def calculate_fixture_specific_features(
    player_data: pd.DataFrame,
    upcoming_fixtures: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate fixture-specific features for upcoming matches.
    
    Args:
        player_data: Player performance data
        upcoming_fixtures: Upcoming fixture information
        
    Returns:
        DataFrame with fixture-specific features
    """
    if player_data.empty or upcoming_fixtures.empty:
        return player_data
    
    fixture_features = player_data.copy()
    
    # Merge upcoming fixture information
    fixture_lookup = upcoming_fixtures.set_index(['team_id', 'gameweek']).to_dict('index')
    
    for idx, row in fixture_features.iterrows():
        team_id = row.get('team_id')
        gameweek = row.get('gameweek')
        
        fixture_info = fixture_lookup.get((team_id, gameweek), {})
        
        if fixture_info:
            fixture_features.loc[idx, 'next_opponent_id'] = fixture_info.get('opponent_id', 0)
            fixture_features.loc[idx, 'next_home_away'] = fixture_info.get('home_away', 'H')
            fixture_features.loc[idx, 'next_difficulty'] = fixture_info.get('difficulty', 3)
    
    return fixture_features


def calculate_positional_matchups(
    player_data: pd.DataFrame,
    h2h_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate position-specific matchup features.
    
    Args:
        player_data: Player data with positions
        h2h_data: Head-to-head data
        
    Returns:
        DataFrame with positional matchup features
    """
    if player_data.empty or h2h_data.empty:
        return player_data
    
    matchup_features = player_data.copy()
    
    # Position-specific performance against different team styles
    position_groups = player_data.groupby('position')
    
    for position, position_group in position_groups:
        if position == 'GK':
            # Goalkeepers vs high-scoring teams
            matchup_features = _calculate_gk_matchups(matchup_features, position_group, h2h_data)
        elif position == 'DEF':
            # Defenders vs attacking teams
            matchup_features = _calculate_def_matchups(matchup_features, position_group, h2h_data)
        elif position in ['MID', 'FWD']:
            # Attackers vs defensive teams
            matchup_features = _calculate_att_matchups(matchup_features, position_group, h2h_data)
    
    return matchup_features


def _calculate_gk_matchups(df: pd.DataFrame, gk_data: pd.DataFrame, h2h_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate goalkeeper-specific matchup features."""
    # GK performance vs high/low scoring opponents
    if 'opponent_id' in gk_data.columns:
        # Calculate opponent scoring rates
        opponent_scoring = gk_data.groupby('opponent_id')['team_goals_against'].mean()
        
        # High scoring opponents (>1.5 goals per game)
        high_scoring_opponents = opponent_scoring[opponent_scoring > 1.5].index
        
        for idx, row in df.iterrows():
            if row.get('position') == 'GK':
                opponent = row.get('opponent_id', 0)
                df.loc[idx, 'gk_vs_high_scoring'] = 1 if opponent in high_scoring_opponents else 0
    
    return df


def _calculate_def_matchups(df: pd.DataFrame, def_data: pd.DataFrame, h2h_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate defender-specific matchup features."""
    # Defender performance vs attacking teams
    if 'opponent_id' in def_data.columns:
        # Calculate opponent attacking strength
        opponent_attacks = def_data.groupby('opponent_id')['team_goals_for'].mean()
        
        # Strong attacking opponents
        strong_attacks = opponent_attacks[opponent_attacks > opponent_attacks.median()].index
        
        for idx, row in df.iterrows():
            if row.get('position') == 'DEF':
                opponent = row.get('opponent_id', 0)
                df.loc[idx, 'def_vs_strong_attack'] = 1 if opponent in strong_attacks else 0
    
    return df


def _calculate_att_matchups(df: pd.DataFrame, att_data: pd.DataFrame, h2h_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate attacker-specific matchup features."""
    # Attacker performance vs defensive teams
    if 'opponent_id' in att_data.columns:
        # Calculate opponent defensive strength (goals conceded)
        opponent_defense = att_data.groupby('opponent_id')['team_goals_against'].mean()
        
        # Weak defensive opponents
        weak_defenses = opponent_defense[opponent_defense > opponent_defense.median()].index
        
        for idx, row in df.iterrows():
            if row.get('position') in ['MID', 'FWD']:
                opponent = row.get('opponent_id', 0)
                df.loc[idx, 'att_vs_weak_defense'] = 1 if opponent in weak_defenses else 0
    
    return df


def calculate_historical_fixture_performance(
    player_data: pd.DataFrame,
    years_back: int = 3
) -> pd.DataFrame:
    """
    Calculate player performance in similar fixtures historically.
    
    Args:
        player_data: Player match data
        years_back: Number of years of history to consider
        
    Returns:
        DataFrame with historical fixture features
    """
    if player_data.empty:
        return player_data
    
    historical_features = player_data.copy()
    
    # Get current date for filtering
    current_date = pd.Timestamp.now()
    cutoff_date = current_date - pd.DateOffset(years=years_back)
    
    if 'kickoff_time' in historical_features.columns:
        historical_features['kickoff_time'] = pd.to_datetime(historical_features['kickoff_time'])
        recent_data = historical_features[historical_features['kickoff_time'] >= cutoff_date]
        
        # Performance in similar gameweeks (e.g., GW1 vs GW1)
        if 'gameweek' in recent_data.columns:
            gw_performance = recent_data.groupby(['element_id', 'gameweek'])['points'].agg(['mean', 'count'])
            gw_performance.columns = ['historical_gw_points', 'historical_gw_count']
            
            # Merge back to main data
            historical_features = historical_features.merge(
                gw_performance,
                left_on=['element_id', 'gameweek'],
                right_index=True,
                how='left'
            )
        
        # Performance in similar months (seasonal patterns)
        if 'kickoff_time' in recent_data.columns:
            recent_data['month'] = recent_data['kickoff_time'].dt.month
            month_performance = recent_data.groupby(['element_id', 'month'])['points'].agg(['mean', 'count'])
            month_performance.columns = ['historical_month_points', 'historical_month_count']
            
            # Add month to main data and merge
            historical_features['month'] = pd.to_datetime(historical_features.get('kickoff_time', pd.Timestamp.now())).dt.month
            historical_features = historical_features.merge(
                month_performance,
                left_on=['element_id', 'month'],
                right_index=True,
                how='left'
            )
    
    # Fill missing values
    fill_columns = [
        'historical_gw_points', 'historical_gw_count',
        'historical_month_points', 'historical_month_count'
    ]
    
    for col in fill_columns:
        if col in historical_features.columns:
            if 'points' in col:
                historical_features[col] = historical_features[col].fillna(historical_features.get('points', 0).mean())
            else:
                historical_features[col] = historical_features[col].fillna(0)
    
    return historical_features
