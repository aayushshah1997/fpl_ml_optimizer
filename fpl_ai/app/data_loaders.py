"""
Shared data loading utilities for dashboard pages.

Consolidates common data loading patterns to eliminate duplication across
all dashboard pages and improve caching efficiency.
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Optional
from pathlib import Path

from fpl_ai.src.common.config import get_config, get_logger
from fpl_ai.src.providers.fpl_api import FPLAPIClient

logger = get_logger(__name__)


def validate_actual_results(df: pd.DataFrame) -> bool:
    """Validate that actual results look realistic."""
    if df.empty:
        logger.warning("Empty actual results DataFrame")
        return False
    
    # Check for unrealistic values
    if df['actual_points'].max() > 30:  # FPL max is ~25-30
        logger.warning("Suspiciously high points detected")
        return False
    
    # Check for too many negative points (should be rare)
    negative_points = (df['actual_points'] < 0).sum()
    if negative_points > len(df) * 0.1:  # More than 10% negative
        logger.warning(f"Too many negative points: {negative_points}")
        return False
    
    # Check that minutes are realistic
    unrealistic_minutes = ((df['actual_minutes'] > 90) | (df['actual_minutes'] < 0)).sum()
    if unrealistic_minutes > 0:
        logger.warning(f"Unrealistic minutes detected: {unrealistic_minutes}")
    
    logger.info(f"Actual results validation passed - {len(df)} players, max points: {df['actual_points'].max()}")
    return True


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions_cached(gameweek: int) -> pd.DataFrame:
    """
    Load predictions for a specific gameweek with caching.
    
    Args:
        gameweek: Gameweek number
        
    Returns:
        DataFrame with player predictions
    """
    try:
        config = get_config()
        artifacts_dir = Path(config.get("io", {}).get("artifacts_dir", "fpl_ai/artifacts"))
        
        predictions_file = artifacts_dir / f"predictions_gw{gameweek}.csv"
        
        if not predictions_file.exists():
            logger.warning(f"Predictions file not found: {predictions_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(predictions_file)
        logger.info(f"Loaded {len(df)} predictions for GW{gameweek}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading predictions for GW{gameweek}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7200)  # Cache for 2 hours (longer as it changes less frequently)
def load_fpl_bootstrap_cached() -> Optional[Dict]:
    """
    Load FPL bootstrap data with caching.
    
    Returns:
        FPL bootstrap data dictionary or None if failed
    """
    try:
        fpl_api = FPLAPIClient()
        bootstrap_data = fpl_api.get_bootstrap_data()
        
        if bootstrap_data:
            logger.info("Successfully loaded FPL bootstrap data")
            return bootstrap_data
        else:
            logger.warning("FPL bootstrap data is empty")
            return None
            
    except Exception as e:
        logger.error(f"Error loading FPL bootstrap data: {e}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_fpl_fixtures_cached() -> Optional[List[Dict]]:
    """
    Load FPL fixtures data with caching.
    
    Returns:
        List of fixture dictionaries or None if failed
    """
    try:
        fpl_api = FPLAPIClient()
        fixtures = fpl_api.get_fixtures()
        
        if fixtures:
            logger.info(f"Successfully loaded {len(fixtures)} FPL fixtures")
            return fixtures
        else:
            logger.warning("FPL fixtures data is empty")
            return None
            
    except Exception as e:
        logger.error(f"Error loading FPL fixtures: {e}")
        return None


def enrich_with_current_fpl_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich predictions DataFrame with current FPL data (costs, team names, etc.).
    
    Args:
        df: Predictions DataFrame
        
    Returns:
        Enriched DataFrame with FPL data
    """
    if df.empty:
        return df
    
    try:
        # Load bootstrap data
        bootstrap_data = load_fpl_bootstrap_cached()
        if not bootstrap_data:
            logger.warning("Cannot enrich data - bootstrap data not available")
            return df
        
        # Create lookup dictionaries
        elements = {elem['id']: elem for elem in bootstrap_data.get('elements', [])}
        teams = {team['id']: team for team in bootstrap_data.get('teams', [])}
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        # Enrich the DataFrame
        enriched_df = df.copy()
        
        # Add player information
        enriched_df['web_name'] = enriched_df['element_id'].map(
            lambda x: elements.get(x, {}).get('web_name', 'Unknown')
        )
        enriched_df['now_cost'] = enriched_df['element_id'].map(
            lambda x: elements.get(x, {}).get('now_cost', 5.0) / 10.0  # Convert to millions
        )
        enriched_df['team_name'] = enriched_df['element_id'].map(
            lambda x: teams.get(elements.get(x, {}).get('team', 0), {}).get('short_name', 'Unknown')
        )
        enriched_df['position'] = enriched_df['element_id'].map(
            lambda x: position_map.get(elements.get(x, {}).get('element_type', 0), 'UNKNOWN')
        )
        
        # Handle missing values
        enriched_df['web_name'] = enriched_df['web_name'].fillna('Unknown')
        enriched_df['now_cost'] = enriched_df['now_cost'].fillna(5.0)
        enriched_df['team_name'] = enriched_df['team_name'].fillna('Unknown')
        enriched_df['position'] = enriched_df['position'].fillna('UNKNOWN')
        
        logger.info(f"Enriched {len(enriched_df)} players with FPL data")
        return enriched_df
        
    except Exception as e:
        logger.error(f"Error enriching data with FPL information: {e}")
        return df


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_current_gameweek() -> int:
    """
    Load current gameweek number.
    
    Returns:
        Current gameweek number or 1 as fallback
    """
    try:
        fpl_api = FPLAPIClient()
        events = fpl_api.get_events()
        
        if events:
            for event in events:
                if event.get('is_current'):
                    return event.get('id', 1)
        
        # Fallback: try to infer from fixtures
        fixtures = load_fpl_fixtures_cached()
        if fixtures:
            completed_gws = set()
            for fixture in fixtures:
                if fixture.get('finished'):
                    completed_gws.add(fixture.get('event'))
            
            if completed_gws:
                return max(completed_gws) + 1
        
        logger.warning("Could not determine current gameweek, defaulting to 1")
        return 1
        
    except Exception as e:
        logger.error(f"Error loading current gameweek: {e}")
        return 1


@st.cache_data(ttl=7200)  # Cache for 2 hours
def load_team_fixtures(team_id: int, current_gw: int, num_fixtures: int = 5) -> List[Dict]:
    """
    Load upcoming fixtures for a specific team.
    
    Args:
        team_id: Team ID
        current_gw: Current gameweek
        num_fixtures: Number of upcoming fixtures to return
        
    Returns:
        List of upcoming fixture dictionaries
    """
    try:
        fixtures = load_fpl_fixtures_cached()
        if not fixtures:
            return []
        
        bootstrap_data = load_fpl_bootstrap_cached()
        if not bootstrap_data:
            return []
        
        # Create team name lookup
        teams = {team['id']: team for team in bootstrap_data.get('teams', [])}
        
        # Filter upcoming fixtures for the team
        upcoming_fixtures = []
        for fixture in fixtures:
            if (fixture.get('event', 0) >= current_gw and 
                (fixture.get('team_h') == team_id or fixture.get('team_a') == team_id)):
                
                opponent_id = fixture.get('team_a') if fixture.get('team_h') == team_id else fixture.get('team_h')
                is_home = fixture.get('team_h') == team_id
                
                fixture_data = {
                    'gameweek': fixture.get('event'),
                    'opponent': teams.get(opponent_id, {}).get('short_name', f'Team {opponent_id}'),
                    'is_home': is_home,
                    'difficulty': fixture.get('team_h_difficulty' if is_home else 'team_a_difficulty', 3),
                    'kickoff_time': fixture.get('kickoff_time')
                }
                upcoming_fixtures.append(fixture_data)
        
        # Sort by gameweek and return requested number
        upcoming_fixtures.sort(key=lambda x: x['gameweek'])
        return upcoming_fixtures[:num_fixtures]
        
    except Exception as e:
        logger.error(f"Error loading fixtures for team {team_id}: {e}")
        return []


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_player_performance_data() -> pd.DataFrame:
    """
    Load player performance tracking data.
    
    Returns:
        DataFrame with player performance data
    """
    try:
        config = get_config()
        artifacts_dir = Path(config.get("io", {}).get("artifacts_dir", "fpl_ai/artifacts"))
        
        performance_file = artifacts_dir / "performance" / "player_performance.csv"
        
        if not performance_file.exists():
            logger.warning(f"Performance file not found: {performance_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(performance_file)
        logger.info(f"Loaded {len(df)} player performance records")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading player performance data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_team_performance_data() -> pd.DataFrame:
    """
    Load team performance tracking data.
    
    Returns:
        DataFrame with team performance data
    """
    try:
        config = get_config()
        artifacts_dir = Path(config.get("io", {}).get("artifacts_dir", "fpl_ai/artifacts"))
        
        performance_file = artifacts_dir / "performance" / "team_performance.csv"
        
        if not performance_file.exists():
            logger.warning(f"Team performance file not found: {performance_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(performance_file)
        logger.info(f"Loaded {len(df)} team performance records")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading team performance data: {e}")
        return pd.DataFrame()


def get_difficulty_color(difficulty: int) -> str:
    """
    Get color emoji for difficulty rating.
    
    Args:
        difficulty: Difficulty rating (1-5)
        
    Returns:
        Color emoji string
    """
    if difficulty <= 2:
        return "🟢"  # Green for easy
    elif difficulty <= 3:
        return "🟡"  # Yellow for medium
    else:
        return "🔴"  # Red for hard


def format_player_cost(cost: float) -> str:
    """
    Format player cost for display.
    
    Args:
        cost: Player cost in millions
        
    Returns:
        Formatted cost string
    """
    return f"£{cost:.1f}M"


def format_expected_points(points: float) -> str:
    """
    Format expected points for display.
    
    Args:
        points: Expected points value
        
    Returns:
        Formatted points string
    """
    return f"{points:.1f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default
