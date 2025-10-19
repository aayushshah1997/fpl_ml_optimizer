"""
FBR API client for fetching comprehensive football statistics.

Provides access to FBRef data through their API with proper rate limiting,
caching, and league-specific data retrieval.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests
import pandas as pd
from ..common.config import get_config, get_logger
from ..common.cache import get_cache

logger = get_logger(__name__)


class FBRAPIClient:
    """
    Client for FBRef API with rate limiting and caching.
    """
    
    def __init__(self):
        """Initialize FBR API client."""
        self.config = get_config()
        self.cache = get_cache()
        
        # API configuration
        try:
            self.api_key = self.config.get_fbr_api_key()
        except ValueError as e:
            logger.error(f"FBR API key not found: {e}")
            self.api_key = None
        
        self.base_url = self.config.get("fbrapi.base_url", "https://fbrapi.com")
        self.session = requests.Session()
        
        # Rate limiting (FBR API requires 3 seconds between requests)
        self.rate_limit = self.config.get("fbrapi.rate_limit_sec", 3.0)
        self.last_request_time = 0
        
        # Cache settings
        self.cache_dir = self.config.get("fbrapi.cache_dir", "cache/fbrapi")
        self.default_cache_ttl = 86400  # 24 hours
        
        logger.info("FBR API client initialized")
    
    def _rate_limit_wait(self):
        """Implement rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        cache_ttl: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Make API request with caching and error handling.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            cache_ttl: Cache time-to-live in seconds
            
        Returns:
            API response data or None if failed
        """
        if not self.api_key:
            logger.error("FBR API key not available")
            return None
        
        # Prepare request parameters
        params = params or {}
        
        # Generate cache key (include API key in cache key for security)
        # Canonicalize params with stable JSON representation
        canonical_params = json.dumps(sorted(params.items()), sort_keys=True)
        params_digest = hashlib.sha256(canonical_params.encode()).hexdigest()[:16]
        
        # Create separate hash for API key to avoid leaking secrets
        api_key_digest = hashlib.sha256(self.api_key.encode()).hexdigest()[:16]
        
        cache_key = f"fbr_{endpoint}_{params_digest}_{api_key_digest}"
        ttl = cache_ttl or self.default_cache_ttl
        
        # Check cache first
        cached_data = self.cache.get(cache_key, "fbrapi", ttl)
        if cached_data is not None:
            logger.debug(f"Cache hit for FBR endpoint: {endpoint}")
            return cached_data
        
        # Rate limiting
        self._rate_limit_wait()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Set headers with API key (correct authentication method)
        headers = {"X-API-Key": self.api_key}
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.get("fbrapi.timeout", 30)
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response
            self.cache.set(cache_key, data, "fbrapi", ttl)
            
            logger.debug(f"FBR API call successful: {endpoint}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FBR API request failed for {endpoint}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Invalid JSON response from FBR {endpoint}: {e}")
            return None
    
    def get_leagues(self) -> Optional[List[Dict]]:
        """
        Get available leagues.
        
        Returns:
            List of league information
        """
        data = self._make_request("leagues")
        return data.get('leagues', []) if data else None
    
    def get_seasons(self, league_id: int) -> Optional[List[Dict]]:
        """
        Get available seasons for a league.
        
        Args:
            league_id: League ID
            
        Returns:
            List of season information
        """
        data = self._make_request(f"leagues/{league_id}/seasons")
        return data.get('seasons', []) if data else None
    
    def get_teams(self, league_id: int, season: str) -> Optional[List[Dict]]:
        """
        Get teams for a league and season.
        
        Args:
            league_id: League ID
            season: Season string (e.g., '2024-25')
            
        Returns:
            List of team information
        """
        data = self._make_request(f"leagues/{league_id}/seasons/{season}/teams")
        return data.get('teams', []) if data else None
    
    def get_players(
        self, 
        league_id: int, 
        season: str, 
        team_id: Optional[int] = None
    ) -> Optional[List[Dict]]:
        """
        Get players for a league, season, and optionally team.
        
        Args:
            league_id: League ID
            season: Season string
            team_id: Optional team ID filter
            
        Returns:
            List of player information
        """
        endpoint = f"leagues/{league_id}/seasons/{season}/players"
        params = {}
        
        if team_id:
            params['team_id'] = team_id
        
        data = self._make_request(endpoint, params)
        return data.get('players', []) if data else None
    
    def get_player_stats(
        self,
        player_id: int,
        league_id: int,
        season: str,
        stats_type: str = "standard"
    ) -> Optional[Dict]:
        """
        Get detailed player statistics.
        
        Args:
            player_id: Player ID
            league_id: League ID
            season: Season string
            stats_type: Type of stats ('standard', 'shooting', 'passing', etc.)
            
        Returns:
            Player statistics data
        """
        endpoint = f"players/{player_id}/stats"
        params = {
            'league_id': league_id,
            'season': season,
            'stats_type': stats_type
        }
        
        return self._make_request(endpoint, params)
    
    def get_player_matches(
        self,
        player_id: int,
        league_id: int,
        season: str,
        limit: int = 100
    ) -> Optional[List[Dict]]:
        """
        Get match-by-match data for a player.
        
        Args:
            player_id: Player ID
            league_id: League ID
            season: Season string
            limit: Maximum matches to return
            
        Returns:
            List of match data
        """
        endpoint = f"players/{player_id}/matches"
        params = {
            'league_id': league_id,
            'season': season,
            'limit': limit
        }
        
        data = self._make_request(endpoint, params)
        return data.get('matches', []) if data else None
    
    def search_player(self, name: str, league_id: Optional[int] = None) -> Optional[List[Dict]]:
        """
        Search for players by name.
        
        Args:
            name: Player name to search
            league_id: Optional league filter
            
        Returns:
            List of matching players
        """
        params = {'name': name}
        if league_id:
            params['league_id'] = league_id
        
        data = self._make_request("players/search", params)
        return data.get('players', []) if data else None
    
    def get_multi_league_player_history(
        self,
        player_name: str,
        leagues: List[int],
        seasons: List[str],
        max_matches_per_season: int = 50
    ) -> Dict[int, List[Dict]]:
        """
        Get player history across multiple leagues and seasons.
        
        Args:
            player_name: Player name
            leagues: List of league IDs to search
            seasons: List of season strings
            max_matches_per_season: Limit matches per season
            
        Returns:
            Dictionary mapping league_id to match data
        """
        all_data = {}
        
        for league_id in leagues:
            league_data = []
            
            # Search for player in this league
            players = self.search_player(player_name, league_id)
            if not players:
                continue
            
            # Take best match (first result)
            player = players[0]
            player_id = player['id']
            
            # Get matches for each season
            for season in seasons:
                matches = self.get_player_matches(
                    player_id, 
                    league_id, 
                    season, 
                    max_matches_per_season
                )
                
                if matches:
                    # Add league and season info to each match
                    for match in matches:
                        match['league_id'] = league_id
                        match['season'] = season
                        match['player_id'] = player_id
                    
                    league_data.extend(matches)
            
            if league_data:
                all_data[league_id] = league_data
                logger.info(f"Retrieved {len(league_data)} matches for {player_name} in league {league_id}")
        
        return all_data
    
    def get_mapped_player_data(
        self,
        fbr_player_mappings: pd.DataFrame,
        leagues: Optional[List[int]] = None,
        seasons_back: int = 2
    ) -> pd.DataFrame:
        """
        Get data for mapped FPL players from FBRef.
        
        Args:
            fbr_player_mappings: DataFrame with fpl_player_id, fbr_player_id, player_name
            leagues: League IDs to search (uses config default if None)
            seasons_back: Number of seasons to retrieve
            
        Returns:
            DataFrame with combined player match data
        """
        if leagues is None:
            leagues = self.config.get("training.extra_leagues", [12, 11, 20, 13, 10])
        
        # Generate recent seasons
        from ..common.timeutil import get_current_season, get_previous_season
        current_season = get_current_season()
        seasons = [current_season]
        
        season = current_season
        for _ in range(seasons_back - 1):
            season = get_previous_season(season)
            seasons.append(season)
        
        all_matches = []
        
        for _, mapping in fbr_player_mappings.iterrows():
            if pd.isna(mapping.get('fbr_player_id')) or pd.isna(mapping.get('player_name')):
                continue
            
            fpl_id = mapping['fpl_player_id']
            player_name = mapping['player_name']
            
            # Get multi-league history
            league_data = self.get_multi_league_player_history(
                player_name, leagues, seasons
            )
            
            # Combine all matches for this player
            player_matches = []
            for league_matches in league_data.values():
                player_matches.extend(league_matches)
            
            # Add FPL mapping info
            for match in player_matches:
                match['fpl_player_id'] = fpl_id
                match['mapped_name'] = player_name
            
            all_matches.extend(player_matches)
            
            if player_matches:
                logger.info(f"Retrieved {len(player_matches)} total matches for {player_name} (FPL ID: {fpl_id})")
        
        if not all_matches:
            logger.warning("No FBR match data retrieved for any mapped players")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_matches)
        
        # Apply league strength adjustments
        league_strength = self.config.get("training.league_strength", {})
        # Ensure we get a Series before calling fillna
        league_mapped = df['league_id'].astype(str).map(league_strength)
        if isinstance(league_mapped, pd.Series):
            df['league_strength_factor'] = league_mapped.fillna(0.85)
        else:
            # If it's a scalar, create a Series with the same length
            df['league_strength_factor'] = pd.Series([league_mapped] * len(df), index=df.index)
        
        logger.info(f"Retrieved {len(df)} total matches from FBR API for {df['fpl_player_id'].nunique()} players")
        return df
    
    def normalize_to_fpl_format(self, fbr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize FBR data to FPL-like column format.
        
        Args:
            fbr_df: DataFrame with FBR match data
            
        Returns:
            DataFrame with FPL-compatible columns
        """
        if fbr_df.empty:
            return fbr_df
        
        # Column mapping from FBR to FPL format
        column_mapping = {
            'goals': 'goals_scored',
            'assists': 'assists',
            'minutes': 'minutes',
            'shots': 'shots',
            'shots_on_target': 'shots_on_target',
            'key_passes': 'key_passes',
            'yellow_cards': 'yellow_cards',
            'red_cards': 'red_cards',
            'saves': 'saves',
            'clean_sheets': 'clean_sheets',
            'goals_conceded': 'goals_conceded',
            'expected_goals': 'expected_goals',
            'expected_assists': 'expected_assists',
            'big_chances_created': 'big_chances_created',
            'big_chances_missed': 'big_chances_missed'
        }
        
        # Create normalized DataFrame
        normalized_df = fbr_df.copy()
        
        # Rename columns
        for fbr_col, fpl_col in column_mapping.items():
            if fbr_col in normalized_df.columns:
                normalized_df = normalized_df.rename(columns={fbr_col: fpl_col})
        
        # Calculate FPL-like points (simplified)
        if 'goals_scored' in normalized_df.columns and 'assists' in normalized_df.columns:
            # Basic points calculation (can be refined)
            normalized_df['basic_points'] = (
                normalized_df.get('goals_scored', 0) * 4 +  # Goals
                normalized_df.get('assists', 0) * 3 +       # Assists
                normalized_df.get('clean_sheets', 0) * 4 +  # Clean sheets
                (normalized_df.get('minutes', 0) >= 60) * 2 # Appearance
            )
        
        # Apply league strength factor to per-90 metrics
        if 'league_strength_factor' in normalized_df.columns:
            per90_cols = ['goals_scored', 'assists', 'shots', 'shots_on_target', 'key_passes']
            for col in per90_cols:
                if col in normalized_df.columns:
                    normalized_df[f'{col}_per90'] = (
                        normalized_df[col] / (normalized_df.get('minutes', 90) / 90) *
                        normalized_df['league_strength_factor']
                    )
        
        return normalized_df
    
    def get_player_season_logs_any_league(self, player_id: str, seasons_back: int = 2) -> pd.DataFrame:
        """
        Return per-match logs for the last N seasons across all leagues the API returns.
        
        This method automatically discovers all leagues where the player has appeared
        and aggregates their match logs with normalized columns.
        
        Args:
            player_id: FBR player ID (string)
            seasons_back: Number of seasons to look back (default 2)
            
        Returns:
            DataFrame with normalized columns: date, league_id, minutes, goals, assists, 
            xg, xa, shots, sot, kp, bc, touches_box, team_id, opp_id
        """
        # Get current and previous seasons
        from ..common.timeutil import get_current_season, get_previous_season
        current_season = get_current_season()
        seasons = [current_season]
        
        season = current_season
        for _ in range(seasons_back - 1):
            season = get_previous_season(season)
            seasons.append(season)
        
        # Get all available leagues
        available_leagues = self.get_leagues()
        if not available_leagues:
            logger.warning("No leagues available from FBR API")
            return pd.DataFrame()
        
        league_ids = [league['id'] for league in available_leagues]
        all_matches = []
        
        # Search across all leagues for this player
        for league_id in league_ids:
            for season in seasons:
                try:
                    matches = self.get_player_matches(
                        int(player_id), 
                        league_id, 
                        season, 
                        limit=100  # Reasonable limit per season
                    )
                    
                    if matches:
                        # Add metadata to each match
                        for match in matches:
                            match['league_id'] = league_id
                            match['season'] = season
                            match['player_id'] = player_id
                        
                        all_matches.extend(matches)
                        logger.debug(f"Found {len(matches)} matches for player {player_id} in league {league_id}, season {season}")
                        
                except Exception as e:
                    logger.debug(f"No data for player {player_id} in league {league_id}, season {season}: {e}")
                    continue
        
        if not all_matches:
            logger.debug(f"No match data found for player {player_id} across any league")
            return pd.DataFrame()
        
        # Convert to DataFrame and normalize columns
        df = pd.DataFrame(all_matches)
        df = self._normalize_player_logs(df)
        
        logger.info(f"Retrieved {len(df)} matches for player {player_id} across {df['league_id'].nunique()} leagues")
        return df
    
    def _normalize_player_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize player match logs to standard column format.
        
        Args:
            df: Raw match logs DataFrame
            
        Returns:
            DataFrame with normalized columns
        """
        if df.empty:
            return df
        
        # Standard column mapping from FBR to normalized format
        column_mapping = {
            'match_date': 'date',
            'minutes_played': 'minutes',
            'goals_scored': 'goals',
            'assists': 'assists',
            'expected_goals': 'xg',
            'expected_assists': 'xa',
            'shots_total': 'shots',
            'shots_on_target': 'sot',
            'key_passes': 'kp',
            'big_chances_created': 'bc',
            'touches_in_box': 'touches_box',
            'team_id': 'team_id',
            'opponent_id': 'opp_id'
        }
        
        # Create normalized DataFrame
        normalized = df.copy()
        
        # Apply column mapping where columns exist
        for old_col, new_col in column_mapping.items():
            if old_col in normalized.columns:
                normalized = normalized.rename(columns={old_col: new_col})
        
        # Ensure required columns exist with defaults
        required_columns = {
            'date': pd.NaT,
            'league_id': 0,
            'minutes': 0,
            'goals': 0,
            'assists': 0,
            'xg': 0.0,
            'xa': 0.0,
            'shots': 0,
            'sot': 0,
            'kp': 0,
            'bc': 0,
            'touches_box': 0,
            'team_id': 0,
            'opp_id': 0
        }
        
        for col, default_val in required_columns.items():
            if col not in normalized.columns:
                normalized[col] = default_val
        
        # Convert data types
        numeric_cols = ['minutes', 'goals', 'assists', 'xg', 'xa', 'shots', 'sot', 'kp', 'bc', 'touches_box']
        for col in numeric_cols:
            if col in normalized.columns:
                # Ensure we get a Series before calling fillna
                numeric_series = pd.to_numeric(normalized[col], errors='coerce')
                if isinstance(numeric_series, pd.Series):
                    normalized[col] = numeric_series.fillna(0)
                else:
                    # If it's a scalar, create a Series with the same length
                    normalized[col] = pd.Series([numeric_series] * len(normalized), index=normalized.index)
        
        # Convert date column
        if 'date' in normalized.columns:
            normalized['date'] = pd.to_datetime(normalized['date'], utc=True, errors='coerce')
        
        # Sort by date
        if 'date' in normalized.columns:
            normalized = normalized.sort_values('date', ascending=False)
        
        return normalized
    
    def health_check(self) -> bool:
        """
        Check if FBR API is accessible.
        
        Returns:
            True if API is healthy
        """
        if not self.api_key:
            return False
        
        try:
            data = self.get_leagues()
            return data is not None and len(data) > 0
        except:
            return False
    
    # ======================================================
    # MANAGER & ROTATION ENGINE HELPERS
    # ======================================================
    
    def get_pl_teams(self) -> pd.DataFrame:
        """Return PL teams with ids and names."""
        comp_id = self.config.get("competitions.premier_league_id", "9")
        data = self._make_request(f"leagues/{comp_id}/teams")
        if data and 'teams' in data:
            return pd.DataFrame(data['teams'])
        return pd.DataFrame()
    
    def get_team_staff(self, team_id: str, season: str) -> pd.DataFrame:
        """Return staff for a team/season; expect manager/head coach if exposed by API."""
        data = self._make_request(f"teams/{team_id}/staff", {'season': season})
        if data and 'staff' in data:
            return pd.DataFrame(data['staff'])
        return pd.DataFrame()
    
    def get_team_matches_all_comps(self, team_id: str, season: str) -> pd.DataFrame:
        """Return all matches for a team in a season across competitions."""
        data = self._make_request(f"teams/{team_id}/matches", {'season': season, 'scope': 'all'})
        if data and 'matches' in data:
            return pd.DataFrame(data['matches'])
        return pd.DataFrame()
    
    def get_match_lineups(self, match_id: str) -> pd.DataFrame:
        """Return starting XI (and subs) for a match."""
        data = self._make_request(f"matches/{match_id}/lineups")
        if data and 'lineups' in data:
            return pd.DataFrame(data['lineups'])
        return pd.DataFrame()