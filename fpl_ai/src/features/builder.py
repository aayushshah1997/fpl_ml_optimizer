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
from pathlib import Path
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
from .loaders import HistoricalDataLoader, CurrentSeasonLoader
from .rolling import RollingCalculator
from .market import MarketFeatureCalculator
from .enrichment import InjuryAvailabilityEnricher, SetPieceEnricher

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
        # Only initialize FBR API if enabled
        if self.config.get("fbrapi.enabled", False):
            self.fbr_api = FBRAPIClient()
        else:
            self.fbr_api = None
            logger.info("FBR API disabled, using only Vaastav data")
        
        self.mapper = FPLMapper()
        self.fixtures = FixturesProvider()
        self.injuries = InjuryProvider()
        self.setpieces = SetPieceRolesManager()
        self.odds = OddsProvider()
        
        # Initialize modular components
        self.historical_loader = HistoricalDataLoader()
        self.current_loader = CurrentSeasonLoader()
        self.rolling_calculator = RollingCalculator()
        self.market_calculator = MarketFeatureCalculator()
        self.injury_enricher = InjuryAvailabilityEnricher()
        self.setpiece_enricher = SetPieceEnricher()
        
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
        
        # Step 2: Get Excel FBRef data (primary multi-league source)
        excel_fbref_data = self._get_excel_fbref_data(extra_seasons_back)
        
        # Step 2b: Get additional multi-league data for mapped players
        multi_league_data = self._get_multi_league_data(extra_leagues, extra_seasons_back)
        
        # Step 3: Combine and normalize data
        combined_data = self._combine_and_normalize_data(fpl_training_data, excel_fbref_data, multi_league_data)
        
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
        
        # Consolidate fixture difficulty columns (handle _x, _y suffixes from merges)
        if 'fixture_difficulty' not in final_data.columns:
            difficulty_cols = [col for col in final_data.columns if 'fixture_difficulty' in col]
            if difficulty_cols:
                logger.info(f"Consolidating fixture difficulty from columns: {difficulty_cols}")
                # Take the first non-null value across all fixture difficulty columns
                final_data['fixture_difficulty'] = final_data[difficulty_cols].bfill(axis=1).iloc[:, 0]
                logger.info(f"Training fixture difficulty stats: {final_data['fixture_difficulty'].describe()}")
            else:
                logger.warning("No fixture difficulty columns found in training data, using default value 3")
                final_data['fixture_difficulty'] = 3
        else:
            logger.info(f"Training fixture difficulty already exists: {final_data['fixture_difficulty'].describe()}")
        
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
        
        # Consolidate fixture difficulty columns (handle _x, _y suffixes from merges)
        if 'fixture_difficulty' not in prediction_features.columns:
            difficulty_cols = [col for col in prediction_features.columns if 'fixture_difficulty' in col]
            if difficulty_cols:
                logger.info(f"Consolidating fixture difficulty from columns: {difficulty_cols}")
                # Take the first non-null value across all fixture difficulty columns
                prediction_features['fixture_difficulty'] = prediction_features[difficulty_cols].bfill(axis=1).iloc[:, 0]
                logger.info(f"Fixture difficulty stats: {prediction_features['fixture_difficulty'].describe()}")
            else:
                logger.warning("No fixture difficulty columns found, using default value 3")
                prediction_features['fixture_difficulty'] = 3
        else:
            logger.info(f"Fixture difficulty already exists: {prediction_features['fixture_difficulty'].describe()}")
        
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
        
        # Convert kickoff_time to datetime with error handling
        if 'kickoff_time' in df.columns:
            try:
                df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], utc=True, errors='coerce')
                # Drop any rows where datetime conversion failed
                df = df.dropna(subset=['kickoff_time'])
            except Exception as e:
                logger.warning(f"Failed to convert kickoff_time to datetime: {e}")
                # If conversion fails completely, drop the column to avoid downstream errors
                df = df.drop(columns=['kickoff_time'])
        
        return df
    
    def _get_historical_season_data(self, season: str, start_gw: int, end_gw: int) -> pd.DataFrame:
        """Get historical season data from vaastav superior dataset."""
        import os

        # Load from superior vaastav dataset (126k+ rows, 9 seasons)
        vaastav_data_path = os.path.join(os.path.dirname(__file__), '../../data/vaastav/cleaned_merged_seasons.csv')

        if os.path.exists(vaastav_data_path):
            try:
                # Load the superior vaastav dataset (126k+ rows across 9 seasons)
                logger.info("Loading vaastav dataset - most comprehensive FPL data available")
                full_data = pd.read_csv(vaastav_data_path, low_memory=False)

                # Resolve season and GW column names robustly
                season_col = 'season_x' if 'season_x' in full_data.columns else (
                    'season' if 'season' in full_data.columns else None
                )
                gw_col = 'GW' if 'GW' in full_data.columns else (
                    'gameweek' if 'gameweek' in full_data.columns else (
                        'round' if 'round' in full_data.columns else None
                    )
                )

                season_data = pd.DataFrame()
                if season_col is not None and gw_col is not None:
                    # Normalize season labels to support both "YYYY-YY" and "YYYY-YYYY"
                    def _normalize_label(s: str) -> str:
                        if isinstance(s, str) and len(s) == 9 and s.count('-') == 1:
                            # e.g. 2024-2025 -> 2024-25
                            try:
                                start, end = s.split('-')
                                return f"{start}-{end[-2:]}"
                            except Exception:
                                return s
                        return s
                    target_labels = {season, _normalize_label(season)}

                    mask = full_data[season_col].astype(str).isin(target_labels)
                    gw_mask = (full_data[gw_col] >= start_gw) & (full_data[gw_col] <= end_gw)
                    season_data = full_data[mask & gw_mask].copy()
                else:
                    logger.warning(
                        f"Unexpected columns in merged vaastav data. Available: {list(full_data.columns)[:20]}"
                    )
                
                # Map vaastav schema to expected feature builder schema
                season_data = self._map_vaastav_schema(season_data)

                logger.info(f"Loaded {len(season_data)} records for {season} GW {start_gw}-{end_gw}")

                # If merged file did not contain this season, try individual GW files as fallback
                if season_data.empty:
                    logger.info(
                        f"Merged dataset returned 0 rows for {season}. Falling back to individual season files."
                    )
                    return self._load_individual_season_data(season, start_gw, end_gw)

                # The vaastav schema mapping already handles all column transformations
                # Remove duplicate columns that might exist
                if not season_data.empty:
                    season_data = season_data.loc[:, ~season_data.columns.duplicated()]

                return season_data

            except Exception as e:
                logger.error(f"Error loading historical data: {e}")
                return pd.DataFrame()
        else:
            # Try individual season files for current seasons (2024-25, 2025-26)
            logger.info(f"Merged file not found, trying individual season files for {season}")
            return self._load_individual_season_data(season, start_gw, end_gw)

    def _map_vaastav_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map vaastav dataset schema to feature builder expected schema.
        
        The vaastav dataset has superior metrics that we want to leverage:
        - Advanced FPL metrics: creativity, ict_index, influence, threat, bps
        - Market data: transfers_in, transfers_out, selected, value
        - Match context: was_home, opponent_team scores
        """
        if df.empty:
            return df
        
        logger.info("Mapping vaastav schema with advanced FPL metrics")
        
        # Create mapped DataFrame with enhanced features
        mapped_df = df.copy()
        
        # Standard mappings (these should already exist in vaastav)
        schema_mapping = {
            'season_x': 'season',
            'name': 'web_name', 
            'position': 'element_type_name',
            'team': 'team_name',
            'GW': 'gameweek',
            'goals_scored': 'goals',
            'minutes': 'minutes',
            'total_points': 'points',
            'assists': 'assists',
            'clean_sheets': 'clean_sheets',
            'goals_conceded': 'goals_conceded',
            'saves': 'saves',
            'yellow_cards': 'yellow_cards',
            'red_cards': 'red_cards',
            'bonus': 'bonus',
            'own_goals': 'own_goals',
            'penalties_missed': 'penalties_missed',
            'penalties_saved': 'penalties_saved',
            # Advanced metrics from vaastav (superior to basic FPL data)
            'bps': 'bps',
            'creativity': 'creativity', 
            'ict_index': 'ict_index',
            'influence': 'influence',
            'threat': 'threat',
            # Market and context data
            'selected': 'selected_by_percent',
            'transfers_in': 'transfers_in',
            'transfers_out': 'transfers_out',
            'transfers_balance': 'transfers_balance',
            'value': 'now_cost',
            'was_home': 'was_home',
            'opponent_team': 'opponent_team'
        }
        
        # Apply schema mapping
        for vaastav_col, expected_col in schema_mapping.items():
            if vaastav_col in mapped_df.columns:
                if expected_col != vaastav_col:
                    mapped_df[expected_col] = mapped_df[vaastav_col]
            else:
                logger.warning(f"Missing vaastav column: {vaastav_col}")

        # Ensure a canonical player identifier exists
        # Vaastav provides 'element' as the player id; our pipeline expects 'element_id'
        if 'element_id' not in mapped_df.columns:
            if 'element' in mapped_df.columns:
                mapped_df['element_id'] = mapped_df['element']
            else:
                # As a last resort, try common alternatives
                for candidate in ['player_id', 'id']:
                    if candidate in mapped_df.columns:
                        mapped_df['element_id'] = mapped_df[candidate]
                        break
                if 'element_id' not in mapped_df.columns:
                    logger.error("No player identifier found ('element' or alternatives). Downstream steps may fail.")
        
        # Convert position strings to numeric codes for consistency
        if 'position' in mapped_df.columns:
            position_mapping = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
            # Ensure we get a Series before calling fillna
            position_mapped = mapped_df['position'].map(position_mapping)
            if isinstance(position_mapped, pd.Series):
                mapped_df['element_type'] = position_mapped.fillna(0)
            else:
                # If it's a scalar, create a Series with the same length
                mapped_df['element_type'] = pd.Series([position_mapped] * len(mapped_df), index=mapped_df.index)
        
        # Ensure numeric columns are properly typed
        numeric_cols = [
            'goals', 'assists', 'minutes', 'points', 'clean_sheets', 'goals_conceded',
            'saves', 'bonus', 'bps', 'creativity', 'ict_index', 'influence', 'threat',
            'yellow_cards', 'red_cards', 'transfers_in', 'transfers_out', 'now_cost'
        ]
        
        for col in numeric_cols:
            if col in mapped_df.columns:
                # Convert string values like "0.0" to numeric, coercing errors to NaN then filling with 0
                # Ensure we get a Series before calling fillna
                numeric_series = pd.to_numeric(mapped_df[col].astype(str), errors='coerce')
                if isinstance(numeric_series, pd.Series):
                    mapped_df[col] = numeric_series.fillna(0)
                else:
                    # If it's a scalar, create a Series with the same length
                    mapped_df[col] = pd.Series([numeric_series] * len(mapped_df), index=mapped_df.index)
        
        # Calculate enhanced features not in basic FPL data
        if all(col in mapped_df.columns for col in ['creativity', 'influence', 'threat']):
            # Verify ICT calculation (should already be in vaastav but let's ensure consistency)
            calculated_ict = mapped_df['creativity'] + mapped_df['influence'] + mapped_df['threat']
            if 'ict_index' in mapped_df.columns:
                # Use vaastav ICT but flag any major discrepancies
                ict_diff = abs(mapped_df['ict_index'] - calculated_ict)
                if ict_diff.max() > 1:
                    logger.debug(f"ICT calculation discrepancy detected (max diff: {ict_diff.max():.2f})")
            else:
                mapped_df['ict_index'] = calculated_ict
        
        # Add derived metrics that boost model performance
        if 'transfers_in' in mapped_df.columns and 'transfers_out' in mapped_df.columns:
            mapped_df['net_transfers'] = mapped_df['transfers_in'] - mapped_df['transfers_out']
        
        if 'now_cost' in mapped_df.columns and 'points' in mapped_df.columns:
            mapped_df['value_efficiency'] = mapped_df['points'] / (mapped_df['now_cost'] + 1)  # +1 to avoid div by 0
        
        logger.info(f"Successfully mapped vaastav schema: {len(mapped_df)} rows with {len(mapped_df.columns)} features")
        return mapped_df
    
    def _load_individual_season_data(self, season: str, start_gw: int, end_gw: int) -> pd.DataFrame:
        """
        Load current season data from individual GW files (for 2024-25, 2025-26).
        The merged dataset only goes to 2023-24, so we need individual files for recent seasons.
        """
        import os
        
        # Use absolute path from project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        season_dir = os.path.join(project_root, 'data', 'vaastav', 'data', season, 'gws')
        
        if not os.path.exists(season_dir):
            logger.warning(f"Season directory not found: {season_dir}")
            return pd.DataFrame()
        
        all_gw_data = []
        
        for gw in range(start_gw, end_gw + 1):
            gw_file = os.path.join(season_dir, f'gw{gw}.csv')
            
            if os.path.exists(gw_file):
                try:
                    gw_data = pd.read_csv(gw_file, low_memory=False)
                    gw_data['GW'] = gw
                    gw_data['season_x'] = season
                    all_gw_data.append(gw_data)
                    logger.info(f"Loaded GW {gw}: {len(gw_data)} records")
                except Exception as e:
                    logger.warning(f"Error loading {gw_file}: {e}")
            else:
                logger.warning(f"GW file not found: {gw_file}")
        
        # Filter out empty DataFrames before concatenation
        non_empty_data = [df for df in all_gw_data if not df.empty]
        
        if not non_empty_data:
            logger.warning(f"No data found for {season} GW {start_gw}-{end_gw}")
            return pd.DataFrame()
        
        # Combine all gameweeks
        combined_data = pd.concat(non_empty_data, ignore_index=True)
        
        # Map to expected schema
        mapped_data = self._map_vaastav_schema(combined_data)
        
        logger.info(f"Loaded {len(mapped_data)} records from individual {season} GW files")
        return mapped_data
    
    
    def _get_excel_fbref_data(self, seasons_back: int) -> pd.DataFrame:
        """Get Excel FBRef data for training."""
        try:
            historical_path = Path(__file__).parent.parent / "data" / "historical_fbref_data.csv"
            if historical_path.exists():
                df = pd.read_csv(historical_path)
                logger.info(f"Loaded Excel FBRef data: {len(df)} records")
                return df
            else:
                logger.warning("Excel FBRef historical data not found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading Excel FBRef data: {e}")
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
        # Skip if FBR API is disabled
        if self.fbr_api is None:
            logger.info("FBR API disabled, skipping multi-league data")
            return pd.DataFrame()
        
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
        excel_fbref_data: pd.DataFrame = None,
        multi_league_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Combine and normalize FPL and multi-league data."""
        combined_data = []
        
        if not fpl_data.empty:
            fpl_normalized = fpl_data.copy()
            fpl_normalized['data_source'] = 'fpl'
            fpl_normalized['league_id'] = 9  # Premier League
            combined_data.append(fpl_normalized)
        
        if excel_fbref_data is not None and not excel_fbref_data.empty:
            excel_normalized = excel_fbref_data.copy()
            excel_normalized['data_source'] = 'excel_fbref'
            combined_data.append(excel_normalized)
        
        if multi_league_data is not None and not multi_league_data.empty:
            ml_normalized = multi_league_data.copy()
            ml_normalized['data_source'] = 'fbr'
            combined_data.append(ml_normalized)
        
        if not combined_data:
            return pd.DataFrame()
        
        # Combine all data
        all_data = pd.concat(combined_data, ignore_index=True)
        
        # Ensure clean index and remove duplicates
        all_data = all_data.reset_index(drop=True)
        all_data = all_data.drop_duplicates().reset_index(drop=True)
        
        # Remove duplicate columns (keep first occurrence) - more robust approach
        if all_data.columns.duplicated().any():
            logger.warning(f"Found {all_data.columns.duplicated().sum()} duplicate columns, removing duplicates")
            # Get column names and their first occurrence positions
            unique_cols = []
            seen_cols = set()
            for col in all_data.columns:
                if col not in seen_cols:
                    unique_cols.append(col)
                    seen_cols.add(col)
                else:
                    logger.debug(f"Removing duplicate column: {col}")
            
            # Select only the first occurrence of each column
            all_data = all_data[unique_cols]
        
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
        if df.empty:
            return df
            
        # Reset index to avoid duplicate label issues
        df = df.reset_index(drop=True)
        
        if 'league_strength_factor' not in df.columns:
            # Add league strength factors
            league_strength = self.config.get("training.league_strength", {})
            if 'league_id' in df.columns:
                # Ensure we get a Series, not a scalar
                league_mapped = df['league_id'].astype(str).map(league_strength)
                if isinstance(league_mapped, pd.Series):
                    df['league_strength_factor'] = league_mapped.fillna(1.0)
                else:
                    # If map returns a scalar, create a Series with the same length
                    if pd.isna(league_mapped):
                        df['league_strength_factor'] = 1.0
                    else:
                        df['league_strength_factor'] = pd.Series([float(league_mapped)] * len(df), index=df.index)
            else:
                df['league_strength_factor'] = 1.0
        
        # Apply adjustments to per-90 metrics
        per90_cols = ['goals', 'assists', 'shots', 'shots_on_target', 'key_passes']
        
        for col in per90_cols:
            if col in df.columns:
                # Calculate per-90 and apply league adjustment - ensure we get 1D arrays
                try:
                    # Handle potential duplicate columns by taking first occurrence
                    col_series = df[col]
                    if isinstance(col_series, pd.DataFrame):
                        col_series = col_series.iloc[:, 0]  # Take first column if multiple
                    
                    minutes_series = df['minutes']
                    if isinstance(minutes_series, pd.DataFrame):
                        minutes_series = minutes_series.iloc[:, 0]
                        
                    strength_series = df['league_strength_factor']
                    if isinstance(strength_series, pd.DataFrame):
                        strength_series = strength_series.iloc[:, 0]
                    
                    # Extract 1D arrays
                    col_values = col_series.fillna(0).values.flatten()
                    minutes_values = minutes_series.fillna(90).values.flatten()
                    strength_values = strength_series.fillna(1.0).values.flatten()
                    
                    # Avoid division by zero
                    minutes_per90 = np.where(minutes_values > 0, minutes_values / 90, 1)
                    per90_values = (col_values / minutes_per90) * strength_values
                    
                    df[f'{col}_per90'] = per90_values
                    
                except Exception as e:
                    logger.warning(f"Error calculating {col}_per90: {e}")
                    # Fallback to simple assignment
                    df[f'{col}_per90'] = 0
        
        return df
    
    def _engineer_training_features(self, data: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Engineer comprehensive features for training."""
        if data.empty:
            return data

        feature_data = data.copy()

        # Add position column if missing (needed for position-specific models)
        if 'position' not in feature_data.columns and 'element_type' in feature_data.columns:
            try:
                # Get bootstrap data for position mapping
                bootstrap_data = self.fpl_api.get_bootstrap_data()
                if bootstrap_data and 'element_types' in bootstrap_data:
                    positions = {pos['id']: pos for pos in bootstrap_data['element_types']}
                    feature_data['position'] = feature_data['element_type'].map(
                        lambda x: positions.get(x, {}).get('singular_name', 'Unknown')
                    )

                    # Map position names to match model expectations
                    position_mapping = {
                        'Goalkeeper': 'GK',
                        'Defender': 'DEF',
                        'Midfielder': 'MID',
                        'Forward': 'FWD'
                    }
                    # Ensure we get a Series before calling fillna
                    position_mapped = feature_data['position'].map(position_mapping)
                    if isinstance(position_mapped, pd.Series):
                        feature_data['position'] = position_mapped.fillna('UNK')
                    else:
                        # If it's a scalar, create a Series with the same length
                        feature_data['position'] = pd.Series([position_mapped] * len(feature_data), index=feature_data.index)

                    logger.info(f"Added position column with {len(feature_data)} rows")
                else:
                    logger.warning("Could not get bootstrap data for position mapping")
                    feature_data['position'] = 'UNK'  # Default fallback
            except Exception as e:
                logger.warning(f"Failed to add position column: {e}")
                feature_data['position'] = 'UNK'  # Default fallback
        
        # Sort by player and date for rolling features
        # Handle different column names for element ID
        element_col = 'element_id' if 'element_id' in feature_data.columns else 'element'
        # Handle different column names for kickoff_time
        kickoff_col = 'kickoff_time' if 'kickoff_time' in feature_data.columns else 'kickoff_time_x' if 'kickoff_time_x' in feature_data.columns else 'kickoff_time_y'
        feature_data = feature_data.sort_values([element_col, kickoff_col])
        
        # Calculate rolling features
        feature_data = self._calculate_rolling_features(feature_data)
        
        # Add fixture features first (needed for team form)
        feature_data = self._add_fixture_features(feature_data, next_gw)
        
        # Add team form features
        try:
            feature_data = self._add_team_form_features(feature_data)
            logger.info("Team form features successfully added")
        except Exception as e:
            logger.warning(f"Team form features failed: {e}")
            # Continue without team form features if they fail
        
        # Add availability features
        feature_data = self._add_availability_features(feature_data)
        
        # Add set piece features
        feature_data = self._add_setpiece_features(feature_data)
        
        # Remove any remaining duplicate columns before market features
        feature_data = feature_data.loc[:, ~feature_data.columns.duplicated()]
        
        # Add market features
        feature_data = self._add_market_features(feature_data)
        
        # Add H2H features
        feature_data = self._add_h2h_features(feature_data)
        
        return feature_data
    
    def _calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling window features using modular calculator."""
        return self.rolling_calculator.calculate_rolling_features(df)
    
    def _add_team_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team and opponent form features."""
        # Calculate team form using team_form module
        team_form_data = calculate_team_form(df, self.rolling_windows)

        # Merge with main data - handle different column names
        if not team_form_data.empty:
            # Determine correct column names for merge
            team_col = 'team_id' if 'team_id' in df.columns else 'team_x' if 'team_x' in df.columns else 'team'
            gw_col = 'gameweek' if 'gameweek' in df.columns else 'round' if 'round' in df.columns else 'GW'

            team_form_team_col = 'team_id' if 'team_id' in team_form_data.columns else 'team_x' if 'team_x' in team_form_data.columns else 'team'
            team_form_gw_col = 'gameweek' if 'gameweek' in team_form_data.columns else 'round' if 'round' in team_form_data.columns else 'GW'

            df = df.merge(
                team_form_data,
                left_on=[team_col, gw_col],
                right_on=[team_form_team_col, team_form_gw_col],
                how='left'
            )

            # Fix defense strength column naming issue after merge
            # The merge can create _x and _y suffixes, but models expect clean names
            logger.info(f"After team form merge - defense strength columns: {[col for col in df.columns if 'defense_strength' in col]}")
            if 'defense_strength_r3_x' in df.columns and 'defense_strength_r3' not in df.columns:
                df['defense_strength_r3'] = df['defense_strength_r3_x']
                logger.info("Fixed defense_strength_r3 column naming from _x variant after merge")
            elif 'defense_strength_r3_y' in df.columns and 'defense_strength_r3' not in df.columns:
                df['defense_strength_r3'] = df['defense_strength_r3_y']
                logger.info("Fixed defense_strength_r3 column naming from _y variant after merge")
            else:
                logger.info("No defense strength column renaming needed after merge")

        else:
            logger.warning("No team form data available")
        
        return df
    
    def _add_availability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add player availability and rotation features using modular enricher."""
        return self.injury_enricher.add_availability_features(df)
    
    def _add_setpiece_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add set piece role features using modular enricher."""
        return self.setpiece_enricher.add_setpiece_features(df)
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market and ownership features using modular calculator."""
        return self.market_calculator.add_market_features(df)
    
    def _add_fixture_features(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Add fixture difficulty and context features."""
        logger.info(f"Adding fixture features for GW {next_gw}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Available columns for fixture mapping: {sorted([col for col in df.columns if 'team' in col.lower() or 'week' in col.lower() or 'round' in col.lower() or 'gw' in col.lower()])}")
        
        # Check what gameweek column is available
        gw_col = None
        for col in ['gameweek', 'round', 'GW']:
            if col in df.columns:
                gw_col = col
                break
        
        # Check what team column is available  
        team_col = None
        for col in ['team_id', 'team_x', 'team']:
            if col in df.columns:
                team_col = col
                break
        
        logger.info(f"Using gameweek column: {gw_col}, team column: {team_col}")
        if gw_col and team_col:
            # Handle potential DataFrame from duplicate columns
            gw_sample = df[gw_col]
            if isinstance(gw_sample, pd.DataFrame):
                gw_sample = gw_sample.iloc[:, 0]  # Take first column if DataFrame
            team_sample = df[team_col]
            if isinstance(team_sample, pd.DataFrame):
                team_sample = team_sample.iloc[:, 0]  # Take first column if DataFrame
            logger.info(f"Sample values - {gw_col}: {gw_sample.head().tolist()}, {team_col}: {team_sample.head().tolist()}")
        
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
            
            # Map fixture data - handle different column names
            def get_fixture_info(row):
                # Try different column names for team and gameweek
                team_id = row.get('team_id', row.get('team_x', row.get('team', None)))
                gw = row.get('gameweek', row.get('round', row.get('GW', None)))

                # Ensure we have scalar values, not Series
                if hasattr(team_id, 'iloc'):
                    team_id = team_id.iloc[0] if len(team_id) > 0 else None
                if hasattr(gw, 'iloc'):
                    gw = gw.iloc[0] if len(gw) > 0 else None

                if team_id is not None and gw is not None:
                    try:
                        return fixture_lookup.get((team_id, gw), {})
                    except TypeError:
                        # If still having hashable issues, try converting to basic types
                        team_id = int(team_id) if team_id is not None else None
                        gw = int(gw) if gw is not None else None
                        return fixture_lookup.get((team_id, gw), {})
                return {}

            df['fixture_difficulty'] = df.apply(
                lambda row: get_fixture_info(row).get('difficulty', 3),
                axis=1
            )

            df['home_away'] = df.apply(
                lambda row: get_fixture_info(row).get('home_away', 'H'),
                axis=1
            )
            
            df['opponent_id'] = df.apply(
                lambda row: get_fixture_info(row).get('opponent_id', 0),
                axis=1
            )
        else:
            df['fixture_difficulty'] = 3  # Neutral
            df['home_away'] = 'H'
            df['opponent_id'] = 0
        
        return df
    
    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head features."""
        # Prepare data for H2H calculation by mapping column names
        h2h_df = df.copy()

        # Handle column mapping to avoid duplicates
        if 'element' in h2h_df.columns and 'element_id' not in h2h_df.columns:
            # If there's no element_id column, rename element to element_id
            h2h_df = h2h_df.rename(columns={'element': 'element_id'})
        elif 'element' in h2h_df.columns and 'element_id' in h2h_df.columns:
            # If both exist, drop the element column to avoid duplicates
            h2h_df = h2h_df.drop(columns=['element'])

        if 'opponent_team' in h2h_df.columns and 'opponent_id' not in h2h_df.columns:
            # If there's no opponent_id column, rename opponent_team to opponent_id
            h2h_df = h2h_df.rename(columns={'opponent_team': 'opponent_id'})
        elif 'opponent_team' in h2h_df.columns and 'opponent_id' in h2h_df.columns:
            # If both exist, drop the opponent_team column to avoid duplicates
            h2h_df = h2h_df.drop(columns=['opponent_team'])

        h2h_features = calculate_h2h_features(h2h_df, min_h2h_matches=1)

        # Rename columns in H2H features to match main DataFrame
        if not h2h_features.empty:
            reverse_mapping = {}
            if 'element_id' in h2h_features.columns:
                reverse_mapping['element_id'] = 'element'
            if 'opponent_id' in h2h_features.columns:
                reverse_mapping['opponent_id'] = 'opponent_team'
            if reverse_mapping:
                h2h_features = h2h_features.rename(columns=reverse_mapping)

        # Merge H2H features
        if not h2h_features.empty and 'element' in df.columns and 'opponent_team' in df.columns:
            df = df.merge(h2h_features, on=['element', 'opponent_team'], how='left')
        else:
            logger.debug("H2H features not available: missing required columns or insufficient data")
        
        # Fill missing H2H values
        h2h_columns = [col for col in df.columns if col.startswith('h2h_')]
        for col in h2h_columns:
            df[col] = df[col].fillna(0.0)
        
        return df
    
    def _create_targets_and_weights(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Create target variables and sample weights."""
        if df.empty:
            return df
        
        # Normalize required identifier columns before creating targets
        if 'element_id' not in df.columns:
            if 'element' in df.columns:
                df = df.rename(columns={'element': 'element_id'})
                logger.debug("Renamed 'element' to 'element_id' for target creation")
            else:
                logger.error("Missing 'element_id' (or 'element') column prior to target creation")
                # Proceeding without a valid identifier will error later; return empty to fail fast upstream
                return pd.DataFrame()
        if 'gameweek' not in df.columns:
            # Attempt to derive from common alternatives
            for candidate in ['GW', 'round']:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: 'gameweek'})
                    logger.debug(f"Renamed '{candidate}' to 'gameweek' for target creation")
                    break
            if 'gameweek' not in df.columns:
                logger.error("Missing 'gameweek' column prior to target creation")
                return pd.DataFrame()

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
            # Ensure kickoff_time is properly converted to datetime with timezone handling
            kickoff_times = pd.to_datetime(df['kickoff_time'], utc=True, errors='coerce')
            # Drop any rows where datetime conversion failed
            valid_mask = kickoff_times.notna()
            if valid_mask.sum() > 0:
                # Create timezone-aware current_date to match kickoff_time timezone
                current_date = pd.Timestamp.now(tz='UTC')

                weights = get_recency_weights(
                    kickoff_times[valid_mask],
                    current_date=current_date,
                    lambda_games=lambda_games,
                    current_season_boost=current_boost,
                    last_season_boost=last_boost,
                    older_seasons_boost=older_boost
                )
                # Apply weights only to valid rows, use default weight for invalid rows
                df_weights = pd.Series(1.0, index=df.index)
                df_weights[valid_mask] = weights
                weights = df_weights
            else:
                # All kickoff times are invalid, use uniform weights
                weights = pd.Series(1.0, index=df.index)
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
        
        # Rename 'team' column to 'team_id' to match expected column names
        players_df = players_df.rename(columns={'team': 'team_id'})
        
        # Add team and position names
        teams = {team['id']: team for team in bootstrap_data['teams']}
        positions = {pos['id']: pos for pos in bootstrap_data['element_types']}
        
        players_df['team_name'] = players_df['team_id'].map(lambda x: teams.get(x, {}).get('name', ''))
        players_df['position'] = players_df['element_type'].map(lambda x: positions.get(x, {}).get('singular_name', ''))
        
        # Map position names to match model expectations
        position_mapping = {
            'Goalkeeper': 'GK',
            'Defender': 'DEF',
            'Midfielder': 'MID',
            'Forward': 'FWD'
        }
        # Ensure we get a Series before calling fillna
        position_mapped = players_df['position'].map(position_mapping)
        if isinstance(position_mapped, pd.Series):
            players_df['position'] = position_mapped.fillna('UNK')
        else:
            # If it's a scalar, create a Series with the same length
            players_df['position'] = pd.Series([position_mapped] * len(players_df), index=players_df.index)
        
        # Filter players with at least 500 minutes played this season
        if 'minutes' in players_df.columns:
            initial_count = len(players_df)
            players_df = players_df[players_df['minutes'] >= 500]
            filtered_count = len(players_df)
            logger.info(f"Filtered players by minutes: {initial_count} -> {filtered_count} (kept players with 500+ minutes)")
        else:
            logger.warning("No 'minutes' column found, cannot filter by playing time")
        
        # Filter to only include current Premier League teams (2025-26 season)
        current_pl_teams = {
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 
            'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 
            'Leeds', 'Leicester', 'Liverpool', 'Man City', 'Man Utd', 
            'Newcastle', 'Nott\'m Forest', 'Southampton', 'Spurs', 'Sunderland',
            'West Ham', 'Wolves'
        }
        
        if 'team_name' in players_df.columns:
            initial_count = len(players_df)
            players_df = players_df[players_df['team_name'].isin(current_pl_teams)]
            filtered_count = len(players_df)
            logger.info(f"Filtered by current PL teams: {initial_count} -> {filtered_count} (kept current PL players only)")
        else:
            logger.warning("No 'team_name' column found, cannot filter by team")
        
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
        """Engineer features for prediction frame using the same pipeline as training."""
        if players_df.empty:
            return pd.DataFrame()

        # For prediction, we need to create a combined dataset similar to training
        # Use recent_data as the base and add current player info
        if not recent_data.empty:
            # Start with recent data and add current player info
            prediction_df = recent_data.copy()

            # Add current player data from players_df
            if 'id' in players_df.columns and 'element_id' in prediction_df.columns:
                # Get available columns from players_df
                available_cols = players_df.columns.tolist()

                # Map columns that exist (handle different naming conventions)
                column_mapping = {}
                if 'web_name' in available_cols:
                    column_mapping['web_name'] = 'web_name'
                if 'element_type' in available_cols:
                    column_mapping['element_type'] = 'element_type'
                elif 'position' in available_cols:
                    column_mapping['position'] = 'element_type'
                if 'team_id' in available_cols:
                    column_mapping['team_id'] = 'team_id'
                elif 'team' in available_cols:
                    column_mapping['team'] = 'team_id'
                if 'team_name' in available_cols:
                    column_mapping['team_name'] = 'team_name'
                if 'now_cost' in available_cols:
                    column_mapping['now_cost'] = 'now_cost'
                elif 'value' in available_cols:
                    column_mapping['value'] = 'now_cost'
                if 'selected' in available_cols:
                    column_mapping['selected'] = 'selected_by_percent'
                elif 'selected_by_percent' in available_cols:
                    column_mapping['selected_by_percent'] = 'selected_by_percent'

                # Create a mapping of current player data using available columns
                if column_mapping:
                    current_player_data = players_df.set_index('id')[list(column_mapping.keys())]
                    current_player_data = current_player_data.rename(columns=column_mapping)

                # Merge current data with recent performance data
                # Ensure we have the right ID column for merging
                if 'id' in players_df.columns:
                    current_player_data = current_player_data.reset_index()
                    current_player_data = current_player_data.rename(columns={'id': 'element_id'})
                    prediction_df = prediction_df.merge(
                        current_player_data,
                        on='element_id',
                        how='left'
                    )
                else:
                    # Fallback: merge on index if no id column
                    prediction_df = prediction_df.merge(
                        current_player_data,
                        left_on='element_id',
                        right_index=True,
                        how='left'
                    )

            # Ensure we have all required columns before feature engineering
            prediction_df = self._ensure_prediction_columns(prediction_df, next_gw)

            # Use the same comprehensive training feature engineering pipeline
            prediction_df = self._engineer_training_features(prediction_df, next_gw)

            return prediction_df
        else:
            # Fallback if no recent data - create minimal prediction frame
            prediction_df = players_df.copy()
            prediction_df['gameweek'] = next_gw

            # Ensure required columns exist
            prediction_df = self._ensure_prediction_columns(prediction_df, next_gw)

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

    def _ensure_prediction_columns(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Ensure prediction data has all columns required by the training pipeline."""
        logger.info(f"Ensuring prediction columns - starting with {len(df.columns)} columns")

        # Add position column if missing (needed for position-specific models)
        if 'position' not in df.columns and 'element_type' in df.columns:
            try:
                # Get bootstrap data for position mapping
                bootstrap_data = self.fpl_api.get_bootstrap_data()
                if bootstrap_data and 'element_types' in bootstrap_data:
                    positions = {pos['id']: pos for pos in bootstrap_data['element_types']}
                    df['position'] = df['element_type'].map(
                        lambda x: positions.get(x, {}).get('singular_name', 'Unknown')
                    )

                    # Map position names to match model expectations
                    position_mapping = {
                        'Goalkeeper': 'GK',
                        'Defender': 'DEF',
                        'Midfielder': 'MID',
                        'Forward': 'FWD'
                    }
                    # Ensure we get a Series before calling fillna
                    position_mapped = df['position'].map(position_mapping)
                    if isinstance(position_mapped, pd.Series):
                        df['position'] = position_mapped.fillna('UNK')
                    else:
                        # If it's a scalar, create a Series with the same length
                        df['position'] = pd.Series([position_mapped] * len(df), index=df.index)
                    logger.info(f"Added position column for prediction data")
                else:
                    logger.warning("Could not get bootstrap data for position mapping")
                    df['position'] = 'UNK'  # Default fallback
            except Exception as e:
                logger.warning(f"Failed to add position column: {e}")
                df['position'] = 'UNK'  # Default fallback

        # Add essential missing columns with sensible defaults
        missing_defaults = {
            # Basic identifiers and context
            'element_id': df.get('element_id', df.get('element')),
            'web_name': df.get('web_name', 'Unknown'),
            'team_name': df.get('team_name', 'Unknown'),

            # Game context
            'gameweek': next_gw,
            'season': '2024-25',
            'data_source': 'fpl_api',

            # Basic stats (defaults for prediction)
            'total_points': 0,
            'points': 0,
            'minutes': 0,
            'goals_scored': 0,
            'assists': 0,
            'clean_sheets': 0,
            'goals_conceded': 0,
            'own_goals': 0,
            'penalties_saved': 0,
            'penalties_missed': 0,
            'yellow_cards': 0,
            'red_cards': 0,
            'saves': 0,
            'bonus': 0,
            'bps': 0,
            'influence': 0.0,
            'creativity': 0.0,
            'threat': 0.0,
            'ict_index': 0.0,

            # Expected stats
            'expected_goals': 0.0,
            'expected_assists': 0.0,
            'expected_goal_involvements': 0.0,
            'expected_goals_conceded': 0.0,

            # Market data
            'now_cost': 50,  # Default only used if column missing
            'selected_by_percent': df.get('selected_by_percent', df.get('selected', 0.0)),
            'transfers_balance': 0,
            'transfers_in': 0,
            'transfers_out': 0,

            # League info
            'league_id': 0,
            'league_strength_factor': 1.0,

            # Fixture data (defaults)
            'fixture': 0,
            'opponent_team': 1,
            'was_home': True,
            'kickoff_time': pd.Timestamp.now(tz='UTC'),
            'team_h_score': 0,
            'team_a_score': 0,
            'round': next_gw,

            # Availability and rotation
            'avail_prob': 0.85,
            'rotation_risk': 0.0,
            'fixture_congestion': 0.5,

            # Set piece roles
            'pen_taker': False,
            'fk_taker': False,
            'corner_taker': False,

            # Injury status
            'injury_status': 'a',

            # H2H features
            'h2h_points_avg_shrunk': 0.0,
            'h2h_goals_avg_shrunk': 0.0,

            # League strength
            'league_strength_mult': 1.0,
            'is_lowtier_league': False,
            'prior_league_uncertainty': 0.0,

            # Transfer momentum
            'transfer_momentum': 0.0,

            # Value metrics
            'value_per_point': 0.0,

            # Position-specific features
            'gk_vs_high_scoring': 0.0,
            'def_vs_strong_attack': 0.0,
            'key_passes': 0,
            'big_chances_created': 0,
            'passes_completed': 0,
            'involvement_intensity': 0.0,
            'fwd_vs_strong_defense': 0.0,
            'shooting_efficiency': 0.0,
            'conversion_rate': 0.0,

            # Rolling features (3, 5, 8 game windows) - will be calculated from historical data
            'points_r3': 0.0, 'points_r5': 0.0, 'points_r8': 0.0,
            'minutes_r3': 0.0, 'minutes_r5': 0.0, 'minutes_r8': 0.0,
            'goals_r3': 0.0, 'goals_r5': 0.0, 'assists_r3': 0.0, 'assists_r5': 0.0,
            'clean_sheets_r3': 0.0, 'clean_sheets_r5': 0.0,
            'goals_conceded_r3': 0.0, 'goals_conceded_r5': 0.0,
            'yellow_cards_r3': 0.0, 'yellow_cards_r5': 0.0,
            'red_cards_r3': 0.0, 'red_cards_r5': 0.0,
            'saves_r3': 0.0, 'saves_r5': 0.0,
            'bonus_r3': 0.0, 'bonus_r5': 0.0,
            'bps_r3': 0.0, 'bps_r5': 0.0,
            'ict_index_r3': 0.0, 'ict_index_r5': 0.0,
            'creativity_r3': 0.0, 'creativity_r5': 0.0,
            'influence_r3': 0.0, 'influence_r5': 0.0,
            'threat_r3': 0.0, 'threat_r5': 0.0,
            'expected_goals_r3': 0.0, 'expected_goals_r5': 0.0,
            'expected_assists_r3': 0.0, 'expected_assists_r5': 0.0,

            # Team form features
            'team_form_r3': 0.0, 'team_form_r5': 0.0,
            'attack_strength_r3': 0.0, 'attack_strength_r5': 0.0,
            'defense_strength_r3': 0.0, 'defense_strength_r5': 0.0,
            'fixture_difficulty': 3.0,
            'home_away_H': 1.0,  # One-hot encoded for home
        }

        # Note: Defense strength column renaming is now handled after team form merge

        # Add missing columns with default values
        for col, default_value in missing_defaults.items():
            if col not in df.columns:
                if callable(default_value):
                    df[col] = default_value()
                else:
                    df[col] = default_value
                logger.debug(f"Added missing column: {col} = {default_value}")

        # Calculate rolling features from historical data
        df = self._calculate_rolling_features_for_prediction(df, next_gw)

        # Consolidate duplicate rolling features (remove _x, _y suffixes)
        df = self._consolidate_rolling_features(df)

        # Ensure data types are correct
        df = self._ensure_correct_data_types(df)

        logger.info(f"Prediction columns ensured - now have {len(df.columns)} columns")
        return df

    def _ensure_correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure prediction data has correct data types."""
        # Convert numeric columns to proper types
        numeric_cols = [
            'total_points', 'points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
            'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
            'selected_by_percent', 'transfers_balance', 'transfers_in', 'transfers_out',
            'fixture_difficulty', 'value', 'now_cost', 'element_id', 'element_type', 'team_id',
            'team_h_score', 'team_a_score', 'fixture', 'opponent_team', 'round', 'gameweek',
            'key_passes', 'big_chances_created', 'passes_completed', 'league_id',
            'avail_prob', 'rotation_risk', 'fixture_congestion',
            'h2h_points_avg_shrunk', 'h2h_goals_avg_shrunk',
            'league_strength_mult', 'prior_league_uncertainty', 'transfer_momentum',
            'value_per_point', 'gk_vs_high_scoring', 'def_vs_strong_attack',
            'involvement_intensity', 'fwd_vs_strong_defense', 'shooting_efficiency', 'conversion_rate'
        ] + [f'{stat}_r{window}' for stat in ['points', 'minutes', 'goals', 'assists', 'clean_sheets',
                                              'goals_conceded', 'yellow_cards', 'red_cards', 'saves', 'bonus',
                                              'bps', 'ict_index', 'creativity', 'influence', 'threat',
                                              'expected_goals', 'expected_assists', 'team_form',
                                              'attack_strength', 'defense_strength'] for window in [3, 5, 8]]

        for col in numeric_cols:
            if col in df.columns:
                if col in ['influence', 'creativity', 'threat', 'ict_index', 'expected_goals',
                          'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
                          'selected_by_percent', 'avail_prob', 'rotation_risk', 'fixture_congestion',
                          'h2h_points_avg_shrunk', 'h2h_goals_avg_shrunk', 'league_strength_mult',
                          'prior_league_uncertainty', 'transfer_momentum', 'value_per_point',
                          'gk_vs_high_scoring', 'def_vs_strong_attack', 'involvement_intensity',
                          'fwd_vs_strong_defense', 'shooting_efficiency', 'conversion_rate'] + \
                         [f'{stat}_r{window}' for stat in ['points', 'minutes', 'goals', 'assists', 'clean_sheets',
                                                          'goals_conceded', 'yellow_cards', 'red_cards', 'saves', 'bonus',
                                                          'bps', 'ict_index', 'creativity', 'influence', 'threat',
                                                          'expected_goals', 'expected_assists', 'team_form',
                                                          'attack_strength', 'defense_strength'] for window in [3, 5, 8]]:
                    # Ensure we get a Series before calling fillna
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if isinstance(numeric_series, pd.Series):
                        df[col] = numeric_series.fillna(0.0)
                    else:
                        # If it's a scalar, create a Series with the same length
                        df[col] = pd.Series([numeric_series] * len(df), index=df.index)
                else:
                    # Ensure we get a Series before calling fillna
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if isinstance(numeric_series, pd.Series):
                        df[col] = numeric_series.fillna(0)
                    else:
                        # If it's a scalar, create a Series with the same length
                        df[col] = pd.Series([numeric_series] * len(df), index=df.index)

        # Convert boolean columns
        bool_cols = ['pen_taker', 'fk_taker', 'corner_taker', 'was_home', 'is_lowtier_league']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Convert string columns
        str_cols = ['web_name', 'season', 'data_source', 'injury_status', 'position', 'team_name']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Convert datetime columns
        if 'kickoff_time' in df.columns:
            df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], utc=True, errors='coerce')

        return df

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

    def _calculate_rolling_features_for_prediction(self, df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
        """Calculate rolling features from historical data for prediction frame."""
        logger.info("Calculating rolling features from historical data...")
        
        # Get historical data for current season
        try:
            # Determine current season dynamically (e.g., '2025-26')
            from src.common.timeutil import get_current_season
            current_season = get_current_season()
            # Try to load current season data directly from individual GW files
            historical_data = self._load_individual_season_data(current_season, 1, next_gw - 1)
            
            if historical_data.empty:
                logger.warning("No historical data available for rolling features")
                return df
            
            logger.info(f"Loaded {len(historical_data)} historical records for rolling features")
            
            # Calculate rolling features for each player
            rolling_features = self._calculate_player_rolling_features(historical_data)
            
            # Merge rolling features with prediction frame
            if 'element_id' in df.columns and 'element_id' in rolling_features.columns:
                rolling_cols = [col for col in rolling_features.columns if '_r' in col]
                df = df.merge(
                    rolling_features[['element_id'] + rolling_cols],
                    on='element_id',
                    how='left'
                )
                logger.info(f"Merged rolling features: {len(rolling_cols)} features")
            else:
                logger.warning("Cannot merge rolling features - missing element_id column")
                
        except Exception as e:
            logger.error(f"Failed to calculate rolling features: {e}")
            
        return df

    def _calculate_player_rolling_features(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling features for each player from historical data."""
        if historical_data.empty or 'element_id' not in historical_data.columns:
            return pd.DataFrame()
        
        # Sort by player and gameweek
        historical_data = historical_data.sort_values(['element_id', 'gameweek'])
        
        # Features to calculate rolling averages for
        rolling_features = [
            'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'goals_conceded', 'yellow_cards', 'red_cards', 'saves', 'bonus',
            'bps', 'ict_index', 'creativity', 'influence', 'threat',
            'expected_goals', 'expected_assists'
        ]
        
        # Windows for rolling calculations
        windows = [3, 5, 8]
        
        result_data = []
        
        for player_id, player_data in historical_data.groupby('element_id'):
            player_features = {'element_id': player_id}
            
            for window in windows:
                for feature in rolling_features:
                    if feature in player_data.columns:
                        # Map feature names to expected output names
                        if feature == 'total_points':
                            rolling_col = f"points_r{window}"
                        elif feature == 'goals_scored':
                            rolling_col = f"goals_r{window}"
                        else:
                            rolling_col = f"{feature}_r{window}"
                        
                        # Calculate rolling mean for the last 'window' games
                        rolling_values = player_data[feature].rolling(window=window, min_periods=1).mean()
                        # Take the last value (most recent rolling average)
                        player_features[rolling_col] = rolling_values.iloc[-1] if len(rolling_values) > 0 else 0.0
                    else:
                        # Map feature names to expected output names
                        if feature == 'total_points':
                            rolling_col = f"points_r{window}"
                        elif feature == 'goals_scored':
                            rolling_col = f"goals_r{window}"
                        else:
                            rolling_col = f"{feature}_r{window}"
                        player_features[rolling_col] = 0.0
            
            result_data.append(player_features)
        
        return pd.DataFrame(result_data)

    def _consolidate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Consolidate duplicate rolling features with _x, _y suffixes."""
        logger.info("Consolidating duplicate rolling features...")
        
        # Define rolling feature patterns
        rolling_patterns = [
            'points_r3', 'points_r5', 'points_r8',
            'goals_r3', 'goals_r5', 'goals_r8',
            'assists_r3', 'assists_r5', 'assists_r8',
            'minutes_r3', 'minutes_r5', 'minutes_r8',
            'clean_sheets_r3', 'clean_sheets_r5', 'clean_sheets_r8',
            'goals_conceded_r3', 'goals_conceded_r5', 'goals_conceded_r8',
            'yellow_cards_r3', 'yellow_cards_r5', 'yellow_cards_r8',
            'red_cards_r3', 'red_cards_r5', 'red_cards_r8',
            'saves_r3', 'saves_r5', 'saves_r8',
            'bonus_r3', 'bonus_r5', 'bonus_r8',
            'bps_r3', 'bps_r5', 'bps_r8',
            'ict_index_r3', 'ict_index_r5', 'ict_index_r8',
            'creativity_r3', 'creativity_r5', 'creativity_r8',
            'influence_r3', 'influence_r5', 'influence_r8',
            'threat_r3', 'threat_r5', 'threat_r8',
            'expected_goals_r3', 'expected_goals_r5', 'expected_goals_r8',
            'expected_assists_r3', 'expected_assists_r5', 'expected_assists_r8'
        ]
        
        for pattern in rolling_patterns:
            # Find columns with this pattern + suffixes
            x_col = f"{pattern}_x"
            y_col = f"{pattern}_y"
            
            if x_col in df.columns and y_col in df.columns:
                # Use _x column if it has non-zero values, otherwise use _y
                x_values = df[x_col]
                y_values = df[y_col]
                
                # Choose the column with more non-zero values
                x_nonzero = (x_values != 0).sum()
                y_nonzero = (y_values != 0).sum()
                
                if x_nonzero >= y_nonzero:
                    df[pattern] = x_values
                    logger.debug(f"Using {x_col} for {pattern} ({x_nonzero} non-zero values)")
                else:
                    df[pattern] = y_values
                    logger.debug(f"Using {y_col} for {pattern} ({y_nonzero} non-zero values)")
                
                # Drop the duplicate columns
                df = df.drop(columns=[x_col, y_col])
                
            elif x_col in df.columns:
                df[pattern] = df[x_col]
                df = df.drop(columns=[x_col])
                logger.debug(f"Renamed {x_col} to {pattern}")
                
            elif y_col in df.columns:
                df[pattern] = df[y_col]
                df = df.drop(columns=[y_col])
                logger.debug(f"Renamed {y_col} to {pattern}")
        
        logger.info("Rolling features consolidated")
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
