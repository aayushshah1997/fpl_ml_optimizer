"""
Team form and strength calculation features.

Calculates team-level metrics including form, attacking/defensive strength,
and contextual factors that affect player performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


def calculate_team_form(
    match_data: pd.DataFrame,
    rolling_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Calculate comprehensive team form metrics.
    
    Args:
        match_data: DataFrame with match-level data
        rolling_windows: Rolling window sizes
        
    Returns:
        DataFrame with team form metrics
    """
    if match_data.empty:
        return pd.DataFrame()
    
    config = get_config()
    if rolling_windows is None:
        rolling_windows = config.get("training.rolling_windows", [3, 5, 8])
    
    # Aggregate to team-gameweek level
    team_data = _aggregate_to_team_level(match_data)
    
    # Calculate rolling form metrics
    team_form = _calculate_rolling_team_metrics(team_data, rolling_windows)
    
    # Calculate strength ratings
    team_form = _calculate_team_strength(team_form, rolling_windows)
    
    # Calculate opponent-adjusted metrics
    team_form = _calculate_opponent_adjusted_metrics(team_form, rolling_windows)
    
    logger.debug(f"Calculated team form for {len(team_form)} team-gameweek combinations")
    return team_form


def _aggregate_to_team_level(match_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate match data to team level for form calculations."""
    if 'team_id' not in match_data.columns or 'gameweek' not in match_data.columns:
        logger.warning("Missing required columns for team aggregation")
        return pd.DataFrame()
    
    # Check what columns are actually available
    available_columns = match_data.columns.tolist()
    logger.info(f"Available columns for team aggregation: {available_columns}")
    
    # Team-level aggregations - only use columns that exist
    agg_dict = {
        'points': 'sum'  # Use 'points' instead of 'total_points'
    }
    
    # Add optional columns if they exist
    if 'goals' in available_columns:
        agg_dict['goals'] = 'sum'
    if 'assists' in available_columns:
        agg_dict['assists'] = 'sum'
    if 'clean_sheets' in available_columns:
        agg_dict['clean_sheets'] = 'max'
    if 'goals_conceded' in available_columns:
        agg_dict['goals_conceded'] = 'max'
    if 'yellow_cards' in available_columns:
        agg_dict['yellow_cards'] = 'sum'
    if 'red_cards' in available_columns:
        agg_dict['red_cards'] = 'sum'
    if 'saves' in available_columns:
        agg_dict['saves'] = 'sum'
    if 'bonus' in available_columns:
        agg_dict['bonus'] = 'sum'
    if 'bps' in available_columns:
        agg_dict['bps'] = 'sum'
    if 'minutes' in available_columns:
        agg_dict['minutes'] = 'sum'
    if 'kickoff_time' in available_columns:
        agg_dict['kickoff_time'] = 'first'
    if 'home_away' in available_columns:
        agg_dict['home_away'] = 'first'
    if 'opponent_id' in available_columns:
        agg_dict['opponent_id'] = 'first'
    if 'fixture_difficulty' in available_columns:
        agg_dict['fixture_difficulty'] = 'first'
    
    # Perform aggregation
    team_agg = match_data.groupby(['team_id', 'gameweek']).agg(agg_dict).reset_index()
    
    # Handle potential multi-level column names from aggregation
    if isinstance(team_agg.columns, pd.MultiIndex):
        team_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in team_agg.columns.values]
    
    # Rename columns to sensible names
    rename_map = {
        'points': 'team_points'
    }
    
    # Add optional column renames - handle both direct names and aggregated names
    available_cols = team_agg.columns.tolist()
    if 'goals' in available_cols:
        rename_map['goals'] = 'team_goals_for'
    elif 'goals_sum' in available_cols:
        rename_map['goals_sum'] = 'team_goals_for'
    
    if 'assists' in available_cols:
        rename_map['assists'] = 'team_assists'
    elif 'assists_sum' in available_cols:
        rename_map['assists_sum'] = 'team_assists'
    
    if 'clean_sheets' in available_cols:
        rename_map['clean_sheets'] = 'team_clean_sheet'
    elif 'clean_sheets_max' in available_cols:
        rename_map['clean_sheets_max'] = 'team_clean_sheet'
    if 'goals_conceded_max' in team_agg.columns:
        rename_map['goals_conceded_max'] = 'team_goals_against'
    if 'yellow_cards_sum' in team_agg.columns:
        rename_map['yellow_cards_sum'] = 'team_yellow_cards'
    if 'red_cards_sum' in team_agg.columns:
        rename_map['red_cards_sum'] = 'team_red_cards'
    if 'saves_sum' in team_agg.columns:
        rename_map['saves_sum'] = 'team_saves'
    if 'bonus_sum' in team_agg.columns:
        rename_map['bonus_sum'] = 'team_bonus'
    if 'bps_sum' in team_agg.columns:
        rename_map['bps_sum'] = 'team_bps'
    if 'minutes_sum' in team_agg.columns:
        rename_map['minutes_sum'] = 'team_minutes'
    if 'kickoff_time_first' in team_agg.columns:
        rename_map['kickoff_time_first'] = 'kickoff_time'
    if 'home_away_first' in team_agg.columns:
        rename_map['home_away_first'] = 'home_away'
    if 'opponent_id_first' in team_agg.columns:
        rename_map['opponent_id_first'] = 'opponent_id'
    if 'fixture_difficulty_first' in team_agg.columns:
        rename_map['fixture_difficulty_first'] = 'fixture_difficulty'
    
    team_agg = team_agg.rename(columns=rename_map)
    
    # Calculate derived metrics based on available columns
    if 'team_goals_for' in team_agg.columns and 'team_goals_against' in team_agg.columns:
        team_agg['team_goal_difference'] = team_agg['team_goals_for'] - team_agg['team_goals_against']
        # Team result (3 points win, 1 point draw, 0 points loss)
        team_agg['team_result_points'] = np.where(
            team_agg['team_goal_difference'] > 0, 3,
            np.where(team_agg['team_goal_difference'] == 0, 1, 0)
        )
    else:
        # If we don't have goals data, use points as a proxy
        team_agg['team_result_points'] = team_agg['team_points']
    
    return team_agg


def _calculate_rolling_team_metrics(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Calculate rolling team performance metrics."""
    if 'team_id' not in df.columns:
        return df
    
    # Sort by team and time
    # Handle different column names for kickoff_time
    kickoff_col = 'kickoff_time' if 'kickoff_time' in df.columns else 'kickoff_time_x' if 'kickoff_time_x' in df.columns else 'kickoff_time_y'
    df = df.sort_values(['team_id', kickoff_col])
    
    # Metrics to calculate rolling averages for
    rolling_metrics = [
        'team_points', 'team_goals_for', 'team_goals_against',
        'team_clean_sheet', 'team_goal_difference', 'team_result_points'
    ]
    
    for window in windows:
        for metric in rolling_metrics:
            if metric in df.columns:
                rolling_col = f"{metric}_r{window}"
                df[rolling_col] = df.groupby('team_id')[metric].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(level=0, drop=True)
    
    return df


def _calculate_team_strength(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Calculate team strength ratings."""
    # Calculate attacking and defensive strength
    for window in windows:
        goals_for_col = f"team_goals_for_r{window}"
        goals_against_col = f"team_goals_against_r{window}"
        
        if goals_for_col in df.columns:
            # Attacking strength (goals scored relative to league average)
            league_avg_goals_for = df[goals_for_col].mean()
            df[f"attack_strength_r{window}"] = df[goals_for_col] / (league_avg_goals_for + 0.1)
        
        if goals_against_col in df.columns:
            # Defensive strength (goals conceded relative to league average)
            league_avg_goals_against = df[goals_against_col].mean()
            df[f"defense_strength_r{window}"] = league_avg_goals_against / (df[goals_against_col] + 0.1)
    
    # Overall team strength
    for window in windows:
        attack_col = f"attack_strength_r{window}"
        defense_col = f"defense_strength_r{window}"
        
        if attack_col in df.columns and defense_col in df.columns:
            df[f"team_strength_r{window}"] = (
                df[attack_col] * 0.6 + df[defense_col] * 0.4
            )
    
    return df


def _calculate_opponent_adjusted_metrics(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Calculate opponent-adjusted team metrics."""
    if 'opponent_id' not in df.columns:
        return df
    
    # For each team, calculate their performance against different opponent strengths
    for window in windows:
        # Get opponent defensive strength (affects team's attacking output)
        opponent_defense_col = f"opponent_defense_r{window}"
        if f"defense_strength_r{window}" in df.columns:
            # Create a mapping DataFrame for more efficient merging
            defense_map = df[['team_id', 'gameweek', f"defense_strength_r{window}"]].copy()
            defense_map = defense_map.rename(columns={
                'team_id': 'opponent_id',
                f"defense_strength_r{window}": opponent_defense_col
            })
            
            # Merge to get opponent defensive strength
            df = df.merge(defense_map, on=['opponent_id', 'gameweek'], how='left')
            df[opponent_defense_col] = df[opponent_defense_col].fillna(1.0)
        
        # Get opponent attacking strength (affects team's defensive performance)
        opponent_attack_col = f"opponent_attack_r{window}"
        if f"attack_strength_r{window}" in df.columns:
            # Create a mapping DataFrame for more efficient merging
            attack_map = df[['team_id', 'gameweek', f"attack_strength_r{window}"]].copy()
            attack_map = attack_map.rename(columns={
                'team_id': 'opponent_id',
                f"attack_strength_r{window}": opponent_attack_col
            })
            
            # Merge to get opponent attacking strength
            df = df.merge(attack_map, on=['opponent_id', 'gameweek'], how='left')
            df[opponent_attack_col] = df[opponent_attack_col].fillna(1.0)
        
        # Calculate opponent-adjusted performance
        goals_for_col = f"team_goals_for_r{window}"
        goals_against_col = f"team_goals_against_r{window}"
        
        if goals_for_col in df.columns and opponent_defense_col in df.columns:
            # Adjust goals for by opponent defensive strength
            df[f"adj_attack_r{window}"] = df[goals_for_col] / (df[opponent_defense_col] + 0.1)
        
        if goals_against_col in df.columns and opponent_attack_col in df.columns:
            # Adjust goals against by opponent attacking strength
            df[f"adj_defense_r{window}"] = (df[opponent_attack_col] + 0.1) / (df[goals_against_col] + 0.1)
    
    return df


def calculate_fixture_context(
    team_data: pd.DataFrame,
    fixtures_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate fixture context features.
    
    Args:
        team_data: Team performance data
        fixtures_data: Fixture information
        
    Returns:
        DataFrame with fixture context features
    """
    if team_data.empty:
        return team_data
    
    context_data = team_data.copy()
    
    # Home/away form split
    home_mask = context_data.get('home_away') == 'H'
    away_mask = context_data.get('home_away') == 'A'
    
    if home_mask.any() and away_mask.any():
        for window in [3, 5, 8]:
            goals_col = f"team_goals_for_r{window}"
            points_col = f"team_points_r{window}"
            
            if goals_col in context_data.columns:
                # Calculate home/away splits
                home_goals = context_data[home_mask].groupby('team_id')[goals_col].mean()
                away_goals = context_data[away_mask].groupby('team_id')[goals_col].mean()
                
                context_data[f"home_advantage_goals_r{window}"] = context_data['team_id'].map(
                    (home_goals - away_goals).to_dict()
                ).fillna(0)
            
            if points_col in context_data.columns:
                home_points = context_data[home_mask].groupby('team_id')[points_col].mean()
                away_points = context_data[away_mask].groupby('team_id')[points_col].mean()
                
                context_data[f"home_advantage_points_r{window}"] = context_data['team_id'].map(
                    (home_points - away_points).to_dict()
                ).fillna(0)
    
    # Fixture congestion (games in short period)
    if 'kickoff_time' in context_data.columns:
        # Handle timezone-aware datetime conversion
        context_data['kickoff_time'] = pd.to_datetime(context_data['kickoff_time'], utc=True, errors='coerce')

        # Calculate days since last game
        # Handle different column names for kickoff_time
        kickoff_col = 'kickoff_time' if 'kickoff_time' in context_data.columns else 'kickoff_time_x' if 'kickoff_time_x' in context_data.columns else 'kickoff_time_y'
        context_data = context_data.sort_values(['team_id', kickoff_col])

        # Ensure consistent timezone handling for date differences
        kickoff_series = context_data[kickoff_col]
        if kickoff_series.dt.tz is not None:
            # Convert to timezone-naive for date calculations
            kickoff_series = kickoff_series.dt.tz_convert(None)

        context_data['days_since_last_game'] = context_data.groupby('team_id')[kickoff_col].diff().dt.days
        
        # Congestion indicator (games within 3 days)
        context_data['fixture_congestion'] = (context_data['days_since_last_game'] <= 3).astype(int)
        
        # Rest advantage (more than 7 days rest)
        context_data['extra_rest'] = (context_data['days_since_last_game'] > 7).astype(int)
    
    return context_data


def calculate_momentum_indicators(team_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team momentum and streaks.
    
    Args:
        team_data: Team performance data
        
    Returns:
        DataFrame with momentum indicators
    """
    if team_data.empty or 'team_id' not in team_data.columns:
        return team_data
    
    momentum_data = team_data.copy()
    
    # Sort by team and time
    # Handle different column names for kickoff_time
    kickoff_col = 'kickoff_time' if 'kickoff_time' in momentum_data.columns else 'kickoff_time_x' if 'kickoff_time_x' in momentum_data.columns else 'kickoff_time_y'
    momentum_data = momentum_data.sort_values(['team_id', kickoff_col])
    
    # Win/loss streaks
    if 'team_result_points' in momentum_data.columns:
        # Convert to win/draw/loss
        momentum_data['result'] = momentum_data['team_result_points'].map({3: 'W', 1: 'D', 0: 'L'})
        
        # Calculate current streak
        def calculate_streak(series):
            if len(series) == 0:
                return 0
            
            current_result = series.iloc[-1]
            streak = 1
            
            for i in range(len(series) - 2, -1, -1):
                if series.iloc[i] == current_result:
                    streak += 1
                else:
                    break
            
            return streak if current_result == 'W' else (-streak if current_result == 'L' else 0)
        
        momentum_data['current_streak'] = momentum_data.groupby('team_id')['result'].apply(
            lambda x: x.rolling(window=len(x), min_periods=1).apply(
                lambda y: calculate_streak(y), raw=False
            )
        ).reset_index(level=0, drop=True)
    
    # Points trajectory (improving/declining)
    for window in [3, 5]:
        points_col = f"team_points_r{window}"
        if points_col in momentum_data.columns:
            # Calculate trend using linear regression slope
            def calculate_trend(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                y = series.values
                if np.std(y) == 0:
                    return 0
                return np.corrcoef(x, y)[0, 1] * np.std(y)
            
            momentum_data[f"points_trend_r{window}"] = momentum_data.groupby('team_id')[points_col].rolling(
                window=window, min_periods=2
            ).apply(calculate_trend, raw=False).reset_index(level=0, drop=True)
    
    # Recent form vs season average
    if 'team_points_r3' in momentum_data.columns and 'team_points_r8' in momentum_data.columns:
        momentum_data['recent_vs_season'] = (
            momentum_data['team_points_r3'] - momentum_data['team_points_r8']
        )
    
    return momentum_data


def calculate_team_player_synergy(
    player_data: pd.DataFrame,
    team_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate how well players perform relative to team performance.
    
    Args:
        player_data: Individual player data
        team_data: Team performance data
        
    Returns:
        DataFrame with synergy metrics
    """
    if player_data.empty or team_data.empty:
        return player_data
    
    synergy_data = player_data.copy()
    
    # Merge team performance data
    team_metrics = team_data[['team_id', 'gameweek'] + [
        col for col in team_data.columns if col.startswith(('team_', 'attack_', 'defense_'))
    ]]
    
    synergy_data = synergy_data.merge(
        team_metrics,
        on=['team_id', 'gameweek'],
        how='left'
    )
    
    # Calculate player contribution to team performance
    if 'points' in synergy_data.columns and 'team_points' in synergy_data.columns:
        synergy_data['player_team_contribution'] = (
            synergy_data['points'] / (synergy_data['team_points'] + 1)
        )
    
    # Player performance relative to team form
    for window in [3, 5, 8]:
        player_points_col = f"points_r{window}"
        team_points_col = f"team_points_r{window}"
        
        if player_points_col in synergy_data.columns and team_points_col in synergy_data.columns:
            synergy_data[f"player_vs_team_r{window}"] = (
                synergy_data[player_points_col] / (synergy_data[team_points_col] / 11 + 1)
            )
    
    return synergy_data
