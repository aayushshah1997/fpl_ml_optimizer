"""
Time and gameweek utilities for FPL AI system.

Provides functionality for working with FPL gameweeks, seasons, and dates.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
import pandas as pd
from .config import get_config, get_logger

logger = get_logger(__name__)


def get_current_season() -> str:
    """
    Get current FPL season string (e.g., '2024-25').
    
    Returns:
        Season string in format 'YYYY-YY'
    """
    now = datetime.now()
    
    # FPL season typically starts in August
    if now.month >= 8:
        # Current calendar year to next year
        return f"{now.year}-{str(now.year + 1)[2:]}"
    else:
        # Previous calendar year to current year
        return f"{now.year - 1}-{str(now.year)[2:]}"


def get_season_dates(season: Optional[str] = None) -> Tuple[datetime, datetime]:
    """
    Get start and end dates for a season.
    
    Args:
        season: Season string (e.g., '2024-25'). Uses current season if None.
        
    Returns:
        Tuple of (season_start, season_end)
    """
    if season is None:
        season = get_current_season()
    
    # Parse season string
    match = re.match(r'(\d{4})-(\d{2})', season)
    if not match:
        raise ValueError(f"Invalid season format: {season}")
    
    start_year = int(match.group(1))
    end_year = int(f"20{match.group(2)}")
    
    # FPL season typically starts mid-August, ends late May
    season_start = datetime(start_year, 8, 1)
    season_end = datetime(end_year, 5, 31)
    
    return season_start, season_end


def get_current_gw(api_data: Optional[Dict] = None) -> int:
    """
    Get current gameweek.
    
    Args:
        api_data: Optional FPL bootstrap data to determine GW
        
    Returns:
        Current gameweek number
    """
    if api_data and 'events' in api_data:
        # Find current or next gameweek from API data
        now = datetime.now()
        
        for event in api_data['events']:
            if event['is_current']:
                return event['id']
        
        # If no current event, find next upcoming
        for event in api_data['events']:
            if event['deadline_time']:
                deadline = datetime.fromisoformat(
                    event['deadline_time'].replace('Z', '+00:00')
                )
                if deadline > now:
                    return event['id']
    
    # Fallback: estimate based on date
    season_start, _ = get_season_dates()
    weeks_since_start = (datetime.now() - season_start).days // 7
    
    # Rough estimate (adjust for breaks, postponements)
    estimated_gw = max(1, min(38, weeks_since_start - 1))
    
    logger.warning(f"Using estimated gameweek: {estimated_gw}")
    return estimated_gw


def gw_to_date(gw: int, season: Optional[str] = None) -> datetime:
    """
    Convert gameweek to approximate date.
    
    Args:
        gw: Gameweek number
        season: Season string
        
    Returns:
        Approximate date for gameweek
    """
    season_start, _ = get_season_dates(season)
    
    # Rough estimate: GW1 starts ~2 weeks after season start
    # Each GW is ~1 week apart (adjust for breaks)
    gw_date = season_start + timedelta(days=14 + (gw - 1) * 7)
    
    return gw_date


def date_to_gw(date: datetime, season: Optional[str] = None) -> int:
    """
    Convert date to approximate gameweek.
    
    Args:
        date: Date to convert
        season: Season string
        
    Returns:
        Approximate gameweek number
    """
    season_start, _ = get_season_dates(season)
    
    if date < season_start:
        return 1
    
    days_since_start = (date - season_start).days
    gw = max(1, min(38, (days_since_start - 14) // 7 + 1))
    
    return gw


def get_fixture_difficulty_period(
    gw_start: int, 
    gw_end: int, 
    team_fixtures: Dict[str, List[Dict]]
) -> Dict[str, float]:
    """
    Calculate fixture difficulty for teams over a period.
    
    Args:
        gw_start: Start gameweek
        gw_end: End gameweek
        team_fixtures: Team fixture data
        
    Returns:
        Dictionary mapping team to average fixture difficulty
    """
    config = get_config()
    fdr_weights = config.get("fixtures.fdr_weights", {
        "1": 1.2, "2": 1.1, "3": 1.0, "4": 0.9, "5": 0.8
    })
    
    team_difficulties = {}
    
    for team, fixtures in team_fixtures.items():
        difficulties = []
        
        for fixture in fixtures:
            if gw_start <= fixture.get('event', 0) <= gw_end:
                fdr = fixture.get('difficulty', 3)
                weight = fdr_weights.get(str(fdr), 1.0)
                difficulties.append(weight)
        
        if difficulties:
            team_difficulties[team] = sum(difficulties) / len(difficulties)
        else:
            team_difficulties[team] = 1.0  # Neutral if no fixtures
    
    return team_difficulties


def get_training_window_gws(
    current_gw: int, 
    seasons_back: int = 4,
    include_current: bool = False
) -> List[Tuple[str, int, int]]:
    """
    Get gameweek ranges for training window.
    
    Args:
        current_gw: Current gameweek
        seasons_back: Number of seasons to include
        include_current: Whether to include current season data
        
    Returns:
        List of (season, start_gw, end_gw) tuples
    """
    current_season = get_current_season()
    
    # Parse current season
    match = re.match(r'(\d{4})-(\d{2})', current_season)
    if not match:
        raise ValueError(f"Invalid season format: {current_season}")
    
    start_year = int(match.group(1))
    
    windows = []
    
    # Add previous complete seasons
    for i in range(1, seasons_back):
        season_year = start_year - i
        season_str = f"{season_year}-{str(season_year + 1)[2:]}"
        windows.append((season_str, 1, 38))
    
    # Add current season up to previous GW
    if include_current and current_gw > 1:
        windows.append((current_season, 1, current_gw - 1))
    
    return windows


def get_recency_weights(
    dates: pd.Series,
    current_date: Optional[datetime] = None,
    lambda_games: float = 0.08,
    current_season_boost: float = 1.3,
    last_season_boost: float = 1.1,
    older_seasons_boost: float = 0.7
) -> pd.Series:
    """
    Calculate recency weights for training samples.
    
    Args:
        dates: Series of match dates
        current_date: Reference date (defaults to now)
        lambda_games: Exponential decay parameter
        current_season_boost: Boost for current season
        last_season_boost: Boost for last season
        older_seasons_boost: Boost for older seasons
        
    Returns:
        Series of sample weights
    """
    if current_date is None:
        current_date = datetime.now()
    
    current_season = get_current_season()
    
    # Calculate days ago
    days_ago = (current_date - dates).dt.days
    
    # Base exponential decay
    weights = pd.Series(np.exp(-lambda_games * days_ago / 7), index=dates.index)
    
    # Season-based boosts
    for idx, date in dates.items():
        season = get_current_season() if date >= get_season_dates()[0] else None
        
        if season == current_season:
            weights.loc[idx] *= current_season_boost
        elif season == get_previous_season(current_season):
            weights.loc[idx] *= last_season_boost
        else:
            weights.loc[idx] *= older_seasons_boost
    
    return weights


def get_previous_season(season: str) -> str:
    """Get previous season string."""
    match = re.match(r'(\d{4})-(\d{2})', season)
    if not match:
        raise ValueError(f"Invalid season format: {season}")
    
    start_year = int(match.group(1)) - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def is_gameweek_complete(gw: int, api_data: Optional[Dict] = None) -> bool:
    """
    Check if a gameweek is complete.
    
    Args:
        gw: Gameweek number
        api_data: Optional FPL bootstrap data
        
    Returns:
        True if gameweek is complete
    """
    if api_data and 'events' in api_data:
        for event in api_data['events']:
            if event['id'] == gw:
                return event['finished']
    
    # Fallback: assume GWs before current are complete
    current_gw = get_current_gw(api_data)
    return gw < current_gw


def get_double_gameweeks(api_data: Optional[Dict] = None) -> List[int]:
    """
    Get list of double gameweeks.
    
    Args:
        api_data: Optional FPL bootstrap data
        
    Returns:
        List of gameweek numbers with double fixtures
    """
    if not api_data or 'fixtures' not in api_data:
        return []
    
    # Count fixtures per team per gameweek
    gw_team_counts = {}
    
    for fixture in api_data['fixtures']:
        gw = fixture['event']
        if gw is None:
            continue
            
        if gw not in gw_team_counts:
            gw_team_counts[gw] = {}
        
        for team_id in [fixture['team_h'], fixture['team_a']]:
            gw_team_counts[gw][team_id] = gw_team_counts[gw].get(team_id, 0) + 1
    
    # Find GWs where any team has > 1 fixture
    double_gws = []
    for gw, team_counts in gw_team_counts.items():
        if any(count > 1 for count in team_counts.values()):
            double_gws.append(gw)
    
    return sorted(double_gws)


# Add numpy import at top
import numpy as np
