"""
Core feature builder for training and prediction frames.

This is the central component for feature engineering, handling:
- Multi-season, multi-league training data
- Current season prediction frames  
- Feature normalization and rolling windows
- Sample weighting and target creation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from ..common.config import get_config, get_logger
from ..common.cache import get_cache
from ..common.timeutil import (
    get_current_season, get_season_dates, get_training_window_gws,
    get_recency_weights
)
from ..providers.fpl_api import FPLAPIClient
from ..providers.fbrapi_client import FBRAPIClient
from ..providers.fpl_map import FPLMapper
from ..providers.fixtures import FixturesProvider
from ..providers.injuries import InjuryProvider
from ..providers.setpiece_roles import SetPieceRolesManager
from ..providers.odds_input import OddsProvider
from ..providers.league_strength import (
    strength_and_weight_mult, log_seen_leagues, get_uncertainty_bump, is_lowtier_league
)
from .touches import calculate_touches_features
from .team_form import calculate_team_form
from .h2h import calculate_h2h_features

logger = get_logger(__name__)


class FeatureBuilder:
    """
    Main feature builder for training and prediction data.
    """
    
    def __init__(self):
        """Initialize feature builder."""
        self.config = get_config()
        self.cache = get_cache()
        
        # Initialize providers
        self.fpl_api = FPLAPIClient()
        self.fbr_api = FBRAPIClient()
        self.mapper = FPLMapper()
        self.fixtures = FixturesProvider()
        self.injuries = InjuryProvider()
        self.setpieces = SetPieceRolesManager()
        self.odds = OddsProvider()
        
        # Feature configuration
        self.rolling_windows = self.config.get("training.rolling_windows", [3, 5, 8])
        self.per90_features = self.config.get("features.per90_features", [
            "goals_scored", "assists", "clean_sheets", "goals_conceded",
            "yellow_cards", "red_cards", "saves", "bonus", "bps", "ict_index",
            "creativity", "influence", "threat", "shots", "shots_on_target",
            "key_passes", "big_chances_created", "big_chances_missed",
            "expected_goals", "expected_assists"
        ])
        
        logger.info("Feature builder initialized")
    
    def build_training_table(self, next_gw: int) -> pd.DataFrame:
        """
        Build comprehensive training table with multi-league data.
        
        Args:
            next_gw: Next gameweek to predict (training uses data before this)
            
        Returns:
            Training DataFrame with features, targets, and sample weights
        """
        logger.info(f"Building training table for prediction of GW {next_gw}")
        
        # Get training configuration
        seasons_back = self.config.get("training.seasons_back", 4)
        extra_leagues = self.config.get("training.extra_leagues", [12, 11, 20, 13, 10])
        extra_seasons_back = self.config.get("training.extra_leagues_seasons_back", 2)
        
        # Initialize set to track seen leagues
        self.seen_leagues = set()
        
        # Step 1: Get FPL training data (primary source)
        fpl_training_data = self._get_fpl_training_data(next_gw, seasons_back)
        
        # Step 2: Get multi-league data for mapped players (now supports "auto" discovery)
        multi_league_data = self._get_multi_league_data(extra_leagues, extra_seasons_back)
        
        # Step 3: Combine and normalize data
        combined_data = self._combine_and_normalize_data(fpl_training_data, multi_league_data)
        
        if combined_data.empty:
            logger.error("No training data available")
            return pd.DataFrame()
        
        # Step 4: Engineer features
        feature_data = self._engineer_training_features(combined_data, next_gw)
        
        # Step 5: Create targets and sample weights
        final_data = self._create_targets_and_weights(feature_data, next_gw)
        
        # Step 6: Log seen leagues for audit
        if hasattr(self, 'seen_leagues') and self.seen_leagues:
            log_seen_leagues(self.seen_leagues, self.config.get_settings())
        
        logger.info(f"Built training table: {len(final_data)} samples, {len(final_data.columns)} features")
        return final_data
    
    def build_prediction_frame(self, next_gw: int) -> pd.DataFrame:
        """
        Build prediction frame for current players.
        
        Args:
            next_gw: Gameweek to predict
            
        Returns:
            Prediction DataFrame with current player features
        """
        logger.info(f"Building prediction frame for GW {next_gw}")
        
        # Get current FPL data
        bootstrap_data = self.fpl_api.get_bootstrap_data()
        if not bootstrap_data:
            logger.error("Could not retrieve FPL bootstrap data")
            return pd.DataFrame()
        
        # Get current player data
        current_players = self._get_current_player_data(bootstrap_data)
        
        # Get recent performance data for features
        recent_data = self._get_recent_performance_data(current_players, next_gw)
        
        # Engineer prediction features
        prediction_features = self._engineer_prediction_features(
            current_players, recent_data, next_gw, bootstrap_data
        )
        
        logger.info(f"Built prediction frame: {len(prediction_features)} players, {len(prediction_features.columns)} features")
        return prediction_features
    
    def _get_fpl_training_data(self, next_gw: int, seasons_back: int) -> pd.DataFrame:
        """Get FPL training data for specified seasons."""
        cache_key = f"fpl_training_data_gw{next_gw}_seasons{seasons_back}"
        cached_data = self.cache.get(cache_key, "training", ttl=3600)  # 1 hour cache
        
        if cached_data is not None:
            logger.info("Using cached FPL training data")
            return cached_data
        
        # Get training window
        training_windows = get_training_window_gws(next_gw, seasons_back, include_current=True)
        
        all_match_data = []
        
        for season, start_gw, end_gw in training_windows:
            logger.info(f"Processing {season} GW {start_gw}-{end_gw}")
            
            # Get player data for this season
            season_data = self._get_season_player_data(season, start_gw, end_gw)
            if not season_data.empty:
                season_data['season'] = season
                all_match_data.append(season_data)
        
        if not all_match_data:
            logger.warning("No FPL training data found")
            return pd.DataFrame()
        
        # Combine all seasons
        combined_data = pd.concat(all_match_data, ignore_index=True)
        
        # Cache result
        self.cache.set(cache_key, combined_data, "training")
        
        logger.info(f"Retrieved FPL training data: {len(combined_data)} match records")
        return combined_data
    
    def _get_season_player_data(self, season: str, start_gw: int, end_gw: int) -> pd.DataFrame:
        """Get player data for a specific season and GW range."""
        # This would typically load from cached historical data or API
        # For current season, use element-summary endpoints
        # For past seasons, use cached/stored data
        
        if season == get_current_season():
            return self._get_current_season_data(start_gw, end_gw)
        else:
            return self._get_historical_season_data(season, start_gw, end_gw)
    
    def _get_current_season_data(self, start_gw: int, end_gw: int) -> pd.DataFrame:
        """Get current season data from FPL API."""
        bootstrap_data = self.fpl_api.get_bootstrap_data()
        if not bootstrap_data:
            return pd.DataFrame()
        
        all_data = []
        
        # Get all players
        for player in bootstrap_data['elements']:
            player_id = player['id']
            
            # Get player summary
            summary_data = self.fpl_api.get_player_summary(player_id)
            if not summary_data or 'history' not in summary_data:
                continue
            
            # Process match history
            for match in summary_data['history']:
                gw = match.get('round')
                if gw is None or not (start_gw <= gw <= end_gw):
                    continue
                
                # Add player info to match data
                match_record = match.copy()
                match_record.update({
                    'element_id': player_id,
                    'web_name': player['web_name'],
                    'team_id': player['team'],
                    'element_type': player['element_type'],
                    'now_cost': player['now_cost']
                })
                
                all_data.append(match_record)
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # Add team and position names
        teams = {team['id']: team for team in bootstrap_data['teams']}
        positions = {pos['id']: pos for pos in bootstrap_data['element_types']}
        
        df['team_name'] = df['team_id'].map(lambda x: teams.get(x, {}).get('name', ''))
        df['position'] = df['element_type'].map(lambda x: positions.get(x, {}).get('singular_name', ''))
        
        # Convert kickoff_time to datetime
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
        
        return df
    
    def _get_historical_season_data(self, season: str, start_gw: int, end_gw: int) -> pd.DataFrame:
        """Get historical season data from cache/storage."""
        # This would load from cached historical data files
        # For now, return empty DataFrame (implement historical data storage separately)
        logger.warning(f"Historical data not implemented for {season}")
        return pd.DataFrame()
    
    def _get_multi_league_data(self, leagues, seasons_back: int) -> pd.DataFrame:
        """
        Get multi-league data for mapped players with dynamic league discovery.
        
        Args:
            leagues: Either a list of league IDs or "auto" for dynamic discovery
            seasons_back: Number of seasons to look back
            
        Returns:
            DataFrame with league strength scaling applied
        """
        # Load player mappings
        manual_mappings = self.mapper.load_manual_mappings()
        
        if manual_mappings.empty:
            logger.info("No player mappings available for multi-league data")
            return pd.DataFrame()
        
        all_player_data = []
        
        for _, mapping in manual_mappings.iterrows():
            if pd.isna(mapping.get('fbr_player_id')):
                continue
                
            fbr_player_id = str(mapping['fbr_player_id'])
            fpl_player_id = mapping['fpl_player_id']
            
            if leagues == "auto":
                # Dynamic discovery: get all available leagues for this player
                player_logs = self.fbr_api.get_player_season_logs_any_league(
                    fbr_player_id, seasons_back
                )
            else:
                # Fixed list mode: use existing method
                player_logs = self._get_player_logs_fixed_leagues(
                    fbr_player_id, fpl_player_id, leagues, seasons_back
                )
            
            if not player_logs.empty:
                # Add FPL mapping info
                player_logs['fpl_player_id'] = fpl_player_id
                player_logs['fbr_player_id'] = fbr_player_id
                
                # Apply league strength scaling and tracking
                player_logs = self._apply_league_strength_scaling(player_logs)
                
                all_player_data.append(player_logs)
        
        if not all_player_data:
            logger.info("No multi-league data retrieved for any mapped players")
            return pd.DataFrame()
        
        # Combine all player data
        combined_data = pd.concat(all_player_data, ignore_index=True)
        
        logger.info(f"Retrieved multi-league data: {len(combined_data)} records from {combined_data['league_id'].nunique()} leagues")
        return combined_data
    
    def _get_player_logs_fixed_leagues(
        self, 
        fbr_player_id: str, 
        fpl_player_id: int,
        leagues: List[int], 
        seasons_back: int
    ) -> pd.DataFrame:
        """Get player logs from a fixed list of leagues (legacy mode)."""
        # Use existing multi-league method from FBR API
        from ..common.timeutil import get_current_season, get_previous_season
        current_season = get_current_season()
        seasons = [current_season]
        
        season = current_season
        for _ in range(seasons_back - 1):
            season = get_previous_season(season)
            seasons.append(season)
        
        # Search for player across the specified leagues
        all_matches = []
        for league_id in leagues:
            for season in seasons:
                try:
                    matches = self.fbr_api.get_player_matches(
                        int(fbr_player_id), league_id, season, limit=100
                    )
                    if matches:
                        for match in matches:
                            match['league_id'] = league_id
                            match['season'] = season
                        all_matches.extend(matches)
                except Exception:
                    continue  # Player not found in this league/season
        
        if not all_matches:
            return pd.DataFrame()
        
        # Convert to DataFrame and normalize
        df = pd.DataFrame(all_matches)
        df = self.fbr_api._normalize_player_logs(df)
        return df
    
    def _apply_league_strength_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply league strength scaling to per-90 features and add league metadata.
        
        Args:
            df: Player logs DataFrame with league_id column
            
        Returns:
            DataFrame with strength scaling applied and metadata added
        """
        if df.empty or 'league_id' not in df.columns:
            return df
        
        settings = self.config.get_settings()
        
        # Track unique leagues encountered
        unique_leagues = df['league_id'].unique()
        self.seen_leagues.update(str(lid) for lid in unique_leagues)
        
        # Apply strength scaling for each league
        for league_id in unique_leagues:
            league_mask = df['league_id'] == league_id
            league_id_str = str(league_id)
            
            # Get strength and weight multiplier
            strength, weight_mult = strength_and_weight_mult(league_id_str, settings)
            
            # Scale per-90 features BEFORE they get rolled up
            per90_features = ['xg', 'xa', 'shots', 'sot', 'kp', 'bc', 'touches_box']
            for feature in per90_features:
                if feature in df.columns:
                    # Scale per-90 values by league strength
                    df.loc[league_mask, feature] *= strength
            
            # Add league metadata columns
            df.loc[league_mask, 'league_strength_mult'] = strength
            df.loc[league_mask, 'league_weight_mult'] = weight_mult
            df.loc[league_mask, 'is_lowtier_league'] = is_lowtier_league(league_id_str, settings)
            df.loc[league_mask, 'prior_league_uncertainty'] = get_uncertainty_bump(league_id_str, settings)
        
        return df
    
    def _combine_and_normalize_data(
        self, 
        fpl_data: pd.DataFrame, 
        multi_league_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine and normalize FPL and multi-league data."""
        combined_data = []
        
        if not fpl_data.empty:
            fpl_normalized = fpl_data.copy()
            fpl_normalized['data_source'] = 'fpl'
            fpl_normalized['league_id'] = 9  # Premier League
            combined_data.append(fpl_normalized)
        
        if not multi_league_data.empty:
            ml_normalized = multi_league_data.copy()
            ml_normalized['data_source'] = 'fbr'
            combined_data.append(ml_normalized)
        
        if not combined_data:
            return pd.DataFrame()
        
        # Combine all data
        all_data = pd.concat(combined_data, ignore_index=True)
        
        # Normalize columns to common format
        all_data = self._normalize_column_names(all_data)
        
        # Apply league strength adjustments
        all_data = self._apply_league_adjustments(all_data)
        
        return all_data
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names across data sources."""
        # Standard column mapping
        column_mapping = {
            'total_points': 'points',
            'round': 'gameweek',
            'goals_scored': 'goals',
            'assists': 'assists',
            'clean_sheets': 'clean_sheets',
            'goals_conceded': 'goals_conceded',
            'yellow_cards': 'yellow_cards', 
            'red_cards': 'red_cards',
            'saves': 'saves',
            'bonus': 'bonus',
            'bps': 'bps',
            'influence': 'influence',
            'creativity': 'creativity',
            'threat': 'threat',
            'ict_index': 'ict_index',
            'selected': 'selected',
            'transfers_in': 'transfers_in',
            'transfers_out': 'transfers_out'
        }
        
        # Apply mapping
        df_normalized = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['points', 'gameweek', 'goals', 'assists', 'minutes']
        for col in required_cols:
            if col not in df_normalized.columns:
                df_normalized[col] = 0
        
        return df_normalized
    
    def _apply_league_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply league strength adjustments to per-90 metrics."""
        if 'league_strength_factor' not in df.columns:
            # Add league strength factors
            league_strength = self.config.get("training.league_strength", {})
            df['league_strength_factor'] = df.get('league_id', 9).astype(str).map(league_strength).fillna(1.0)
        
        # Apply adjustments to per-90 metrics
        per90_cols = ['goals', 'assists', 'shots', 'shots_on_target', 'key_passes']
        
        for col in per90_cols:
            if col in df.columns:
                # Calculate per-90 and apply league adjustment
                df[f'{col}_per90'] = (df[col] / (df.get('minutes', 90) / 90)) * df['league_strength_factor']
        
        return df
    
    def _engineer_training_features(self, data: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Engineer comprehensive features for training."""
        if data.empty:
            return data
        
        feature_data = data.copy()
        
        # Sort by player and date for rolling features
        feature_data = feature_data.sort_values(['element_id', 'kickoff_time'])
        
        # Calculate rolling features
        feature_data = self._calculate_rolling_features(feature_data)
        
        # Add team form features
        feature_data = self._add_team_form_features(feature_data)
        
        # Add availability features
        feature_data = self._add_availability_features(feature_data)
        
        # Add set piece features
        feature_data = self._add_setpiece_features(feature_data)
        
        # Add market features
        feature_data = self._add_market_features(feature_data)
        
        # Add fixture features
        feature_data = self._add_fixture_features(feature_data, next_gw)
        
        # Add H2H features
        feature_data = self._add_h2h_features(feature_data)
        
        return feature_data
    
    def _calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling window features."""
        rolling_features = []
        
        for window in self.rolling_windows:
            for col in self.per90_features:
                if col in df.columns:
                    rolling_col = f"{col}_r{window}"
                    df[rolling_col] = df.groupby('element_id')[col].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(level=0, drop=True)
        
        # Calculate rolling minutes for availability assessment
        for window in self.rolling_windows:
            minutes_col = f"minutes_r{window}"
            df[minutes_col] = df.groupby('element_id')['minutes'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(level=0, drop=True)
        
        return df
    
    def _add_team_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team and opponent form features."""
        # Calculate team form using team_form module
        team_form_data = calculate_team_form(df, self.rolling_windows)
        
        # Merge with main data
        df = df.merge(
            team_form_data, 
            on=['team_id', 'gameweek'], 
            how='left'
        )
        
        return df
    
    def _add_availability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add player availability and rotation features."""
        # Get latest bootstrap data for current availability
        bootstrap_data = self.fpl_api.get_bootstrap_data()
        
        if bootstrap_data:
            availability_data = self.injuries.get_comprehensive_availability(
                bootstrap_data, df, 
                df.get('minutes_r3'), 
                df.groupby('team_id')['gameweek'].count()  # Team congestion proxy
            )
            
            # Merge availability data
            availability_lookup = availability_data.set_index('element_id')['final_avail_prob'].to_dict()
            df['avail_prob'] = df['element_id'].map(availability_lookup).fillna(0.85)
        else:
            df['avail_prob'] = 0.85  # Default availability
        
        return df
    
    def _add_setpiece_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add set piece role features."""
        # Get set piece roles
        setpiece_roles = self.setpieces.get_final_roles()
        
        if not setpiece_roles.empty:
            # Create lookup by player name and team
            setpiece_lookup = {}
            for _, role in setpiece_roles.iterrows():
                key = (role['player_name'].lower(), role['team'].lower())
                setpiece_lookup[key] = {
                    'pen_taker': role['pen_share'],
                    'fk_taker': role['fk_share'],
                    'corner_taker': role['corner_share']
                }
            
            # Map to players
            df['pen_taker'] = 0.0
            df['fk_taker'] = 0.0
            df['corner_taker'] = 0.0
            
            for idx, row in df.iterrows():
                player_key = (row.get('web_name', '').lower(), row.get('team_name', '').lower())
                if player_key in setpiece_lookup:
                    roles = setpiece_lookup[player_key]
                    df.loc[idx, 'pen_taker'] = roles['pen_taker']
                    df.loc[idx, 'fk_taker'] = roles['fk_taker']
                    df.loc[idx, 'corner_taker'] = roles['corner_taker']
        else:
            df['pen_taker'] = 0.0
            df['fk_taker'] = 0.0
            df['corner_taker'] = 0.0
        
        return df
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market and ownership features."""
        # Market features (price, ownership, transfers)
        df['value_per_point'] = df['now_cost'] / (df.get('points', 1) + 1)  # Avoid division by zero
        
        # Ownership momentum (if transfers data available)
        if 'transfers_in' in df.columns and 'transfers_out' in df.columns:
            df['net_transfers'] = df['transfers_in'] - df['transfers_out']
            df['transfer_momentum'] = df.groupby('element_id')['net_transfers'].rolling(
                window=3, min_periods=1
            ).mean().reset_index(level=0, drop=True)
        else:
            df['net_transfers'] = 0
            df['transfer_momentum'] = 0
        
        return df
    
    def _add_fixture_features(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Add fixture difficulty and context features."""
        # Get fixture data
        fixtures_df = self.fixtures.get_all_fixtures()
        
        if fixtures_df is not None and not fixtures_df.empty:
            # Add fixture difficulty for each match
            fixture_lookup = {}
            for _, fixture in fixtures_df.iterrows():
                gw = fixture['event']
                home_team = fixture['team_h']
                away_team = fixture['team_a']
                
                # Home team fixture
                fixture_lookup[(home_team, gw)] = {
                    'difficulty': fixture['team_h_difficulty'],
                    'home_away': 'H',
                    'opponent_id': away_team
                }
                
                # Away team fixture
                fixture_lookup[(away_team, gw)] = {
                    'difficulty': fixture['team_a_difficulty'],
                    'home_away': 'A',
                    'opponent_id': home_team
                }
            
            # Map fixture data
            df['fixture_difficulty'] = df.apply(
                lambda row: fixture_lookup.get(
                    (row.get('team_id'), row.get('gameweek')), {}
                ).get('difficulty', 3),
                axis=1
            )
            
            df['home_away'] = df.apply(
                lambda row: fixture_lookup.get(
                    (row.get('team_id'), row.get('gameweek')), {}
                ).get('home_away', 'H'),
                axis=1
            )
            
            df['opponent_id'] = df.apply(
                lambda row: fixture_lookup.get(
                    (row.get('team_id'), row.get('gameweek')), {}
                ).get('opponent_id', 0),
                axis=1
            )
        else:
            df['fixture_difficulty'] = 3  # Neutral
            df['home_away'] = 'H'
            df['opponent_id'] = 0
        
        return df
    
    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head features."""
        h2h_features = calculate_h2h_features(df)
        
        # Merge H2H features
        df = df.merge(h2h_features, on=['element_id', 'opponent_id'], how='left')
        
        return df
    
    def _create_targets_and_weights(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Create target variables and sample weights."""
        if df.empty:
            return df
        
        # Sort by player and gameweek
        df = df.sort_values(['element_id', 'gameweek'])
        
        # Create target (next gameweek points)
        target_name = self.config.get("training.target.name", "points_next")
        df[target_name] = df.groupby('element_id')['points'].shift(-1)
        
        # Remove rows without targets (last gameweek for each player)
        df = df.dropna(subset=[target_name])
        
        # Calculate sample weights
        df = self._calculate_sample_weights(df, next_gw)
        
        # Remove rows with zero weights
        df = df[df['sample_weight'] > 0]
        
        return df
    
    def _calculate_sample_weights(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Calculate sample weights based on recency and other factors."""
        # Get weighting parameters
        lambda_games = self.config.get("training.sample_weighting.lambda_games", 0.08)
        current_boost = self.config.get("training.sample_weighting.current_season_boost", 1.3)
        last_boost = self.config.get("training.sample_weighting.last_season_boost", 1.1)
        older_boost = self.config.get("training.sample_weighting.older_seasons_boost", 0.7)
        
        # Calculate recency weights
        if 'kickoff_time' in df.columns:
            weights = get_recency_weights(
                df['kickoff_time'], 
                current_date=datetime.now(),
                lambda_games=lambda_games,
                current_season_boost=current_boost,
                last_season_boost=last_boost,
                older_seasons_boost=older_boost
            )
        else:
            # Fallback: weight by gameweek distance
            gw_distance = abs(df.get('gameweek', next_gw) - next_gw)
            weights = np.exp(-lambda_games * gw_distance)
        
        # Apply league strength dampening (new system)
        if 'league_weight_mult' in df.columns:
            weights *= df['league_weight_mult'].fillna(1.0)
        elif 'league_strength_factor' in df.columns:
            # Legacy fallback
            weights *= df['league_strength_factor']
        
        # Apply cold start adjustment for players with few minutes
        alpha_minutes = self.config.get("training.cold_start_shrink.alpha_minutes", 600.0)
        if 'minutes_r8' in df.columns:
            total_minutes = df['minutes_r8'] * 8  # Approximate total minutes
            cold_start_factor = total_minutes / (total_minutes + alpha_minutes)
            weights *= cold_start_factor
        
        df['sample_weight'] = weights
        return df
    
    def _get_current_player_data(self, bootstrap_data: Dict) -> pd.DataFrame:
        """Get current player data for prediction frame."""
        players_df = pd.DataFrame(bootstrap_data['elements'])
        
        # Add team and position names
        teams = {team['id']: team for team in bootstrap_data['teams']}
        positions = {pos['id']: pos for pos in bootstrap_data['element_types']}
        
        players_df['team_name'] = players_df['team'].map(lambda x: teams.get(x, {}).get('name', ''))
        players_df['position'] = players_df['element_type'].map(lambda x: positions.get(x, {}).get('singular_name', ''))
        
        return players_df
    
    def _get_recent_performance_data(self, players_df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Get recent performance data for prediction features."""
        # Get recent matches for feature calculation
        recent_data = []
        
        for _, player in players_df.iterrows():
            player_id = player['id']
            
            # Get player summary for recent games
            summary_data = self.fpl_api.get_player_summary(player_id)
            if summary_data and 'history' in summary_data:
                # Take last 8 games for rolling features
                recent_matches = summary_data['history'][-8:]
                
                for match in recent_matches:
                    match_record = match.copy()
                    match_record['element_id'] = player_id
                    recent_data.append(match_record)
        
        if not recent_data:
            return pd.DataFrame()
        
        return pd.DataFrame(recent_data)
    
    def _engineer_prediction_features(
        self,
        players_df: pd.DataFrame,
        recent_data: pd.DataFrame,
        next_gw: int,
        bootstrap_data: Dict
    ) -> pd.DataFrame:
        """Engineer features for prediction frame."""
        if players_df.empty:
            return pd.DataFrame()
        
        # Start with current player data
        prediction_df = players_df.copy()
        
        # Add recent form features if available
        if not recent_data.empty:
            # Calculate rolling features for recent performance
            recent_features = self._calculate_rolling_features(recent_data)
            
            # Aggregate to player level (latest values)
            latest_features = recent_features.groupby('element_id').last()
            
            # Merge with prediction frame
            prediction_df = prediction_df.merge(
                latest_features,
                left_on='id',
                right_index=True,
                how='left'
            )
        
        # Add availability features
        prediction_df = self._add_availability_features(prediction_df)
        
        # Add set piece features
        prediction_df = self._add_setpiece_features(prediction_df)
        
        # Add fixture features for next gameweek
        prediction_df['gameweek'] = next_gw
        prediction_df = self._add_fixture_features(prediction_df, next_gw)
        
        # Fill missing values with defaults
        prediction_df = self._fill_missing_prediction_values(prediction_df)
        
        return prediction_df
    
    def _fill_missing_prediction_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in prediction frame."""
        # Default values for missing features
        defaults = {
            'avail_prob': 0.85,
            'pen_taker': 0.0,
            'fk_taker': 0.0,
            'corner_taker': 0.0,
            'fixture_difficulty': 3,
            'value_per_point': 1.0,
            'net_transfers': 0,
            'transfer_momentum': 0
        }
        
        # Fill rolling features with current season averages
        for window in self.rolling_windows:
            for col in self.per90_features:
                rolling_col = f"{col}_r{window}"
                if rolling_col not in df.columns:
                    df[rolling_col] = 0.0
        
        # Apply defaults
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)
        
        return df


# Standalone functions for easy importing
def build_training_table(next_gw: int) -> pd.DataFrame:
    """
    Build training table for ML models.
    
    Args:
        next_gw: Next gameweek to predict for
        
    Returns:
        Training DataFrame
    """
    builder = FeatureBuilder()
    return builder.build_training_table(next_gw)


def build_prediction_frame(next_gw: int) -> pd.DataFrame:
    """
    Build prediction frame for current gameweek.
    
    Args:
        next_gw: Next gameweek to predict for
        
    Returns:
        Prediction DataFrame
    """
    builder = FeatureBuilder()
    return builder.build_prediction_frame(next_gw)
