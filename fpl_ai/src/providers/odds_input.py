"""
Odds data input provider.

Provides functionality for loading and processing betting odds data
from CSV files and external sources.
"""

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from ..common.config import get_config, get_logger
from ..common.cache import get_cache

logger = get_logger(__name__)


class OddsProvider:
    """
    Provider for betting odds and market data.
    """
    
    def __init__(self):
        """Initialize odds provider."""
        self.config = get_config()
        self.cache = get_cache()
        
        # Odds file path
        self.odds_file = self.config.data_dir / "odds_team.csv"
        
        logger.info("Odds provider initialized")
    
    def load_odds_data(self) -> pd.DataFrame:
        """
        Load odds data from CSV file.
        
        Returns:
            DataFrame with odds data
        """
        try:
            if self.odds_file.exists():
                df = pd.read_csv(self.odds_file, comment='#')
                # Filter out empty/comment rows
                df = df.dropna(subset=['team'])
                df = df[~df['team'].astype(str).str.startswith('#')]
                logger.info(f"Loaded odds data for {len(df)} team-gameweek combinations")
                return df
            else:
                logger.warning(f"Odds file not found: {self.odds_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load odds data: {e}")
            return pd.DataFrame()
    
    def validate_odds_data(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean odds data.
        
        Args:
            odds_df: Raw odds data
            
        Returns:
            Cleaned odds data
        """
        if odds_df.empty:
            return odds_df
        
        cleaned_df = odds_df.copy()
        
        # Required columns
        required_cols = ['team', 'gw', 'opponent']
        for col in required_cols:
            if col not in cleaned_df.columns:
                logger.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Convert numeric columns
        numeric_cols = [
            'win_odds', 'draw_odds', 'loss_odds',
            'clean_sheet_odds', 'over_2_5_odds', 'btts_odds'
        ]
        
        for col in numeric_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Validate odds are positive
        for col in numeric_cols:
            if col in cleaned_df.columns:
                invalid_odds = cleaned_df[col] <= 0
                if invalid_odds.any():
                    logger.warning(f"Found {invalid_odds.sum()} invalid {col} values (<=0)")
                    cleaned_df.loc[invalid_odds, col] = None
        
        # Convert home_away to standard format
        if 'home_away' in cleaned_df.columns:
            cleaned_df['home_away'] = cleaned_df['home_away'].str.upper()
            cleaned_df['home_away'] = cleaned_df['home_away'].map({'H': 'H', 'A': 'A', 'HOME': 'H', 'AWAY': 'A'})
        
        # Remove rows with missing essential data
        essential_cols = ['team', 'gw', 'opponent']
        cleaned_df = cleaned_df.dropna(subset=essential_cols)
        
        logger.info(f"Validated odds data: {len(cleaned_df)} valid rows")
        return cleaned_df
    
    def convert_odds_to_probabilities(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert odds to implied probabilities.
        
        Args:
            odds_df: DataFrame with odds data
            
        Returns:
            DataFrame with added probability columns
        """
        if odds_df.empty:
            return odds_df
        
        prob_df = odds_df.copy()
        
        # Odds to probability conversion
        odds_prob_mapping = {
            'win_odds': 'win_prob',
            'draw_odds': 'draw_prob',
            'loss_odds': 'loss_prob',
            'clean_sheet_odds': 'clean_sheet_prob',
            'over_2_5_odds': 'over_2_5_prob',
            'btts_odds': 'btts_prob'
        }
        
        for odds_col, prob_col in odds_prob_mapping.items():
            if odds_col in prob_df.columns:
                # Probability = 1 / odds (for decimal odds)
                prob_df[prob_col] = 1.0 / prob_df[odds_col].replace(0, None)
        
        # Normalize win/draw/loss probabilities to sum to 1
        if all(col in prob_df.columns for col in ['win_prob', 'draw_prob', 'loss_prob']):
            total_prob = prob_df[['win_prob', 'draw_prob', 'loss_prob']].sum(axis=1)
            
            for col in ['win_prob', 'draw_prob', 'loss_prob']:
                prob_df[col] = prob_df[col] / total_prob
        
        return prob_df
    
    def get_team_odds(
        self,
        team: str,
        gameweek: Optional[int] = None,
        opponent: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get odds for a specific team.
        
        Args:
            team: Team name
            gameweek: Optional gameweek filter
            opponent: Optional opponent filter
            
        Returns:
            DataFrame with team's odds
        """
        odds_df = self.load_odds_data()
        
        if odds_df.empty:
            return pd.DataFrame()
        
        # Filter by team
        team_odds = odds_df[odds_df['team'].str.contains(team, case=False, na=False)]
        
        # Filter by gameweek if specified
        if gameweek is not None:
            team_odds = team_odds[team_odds['gw'] == gameweek]
        
        # Filter by opponent if specified
        if opponent is not None:
            team_odds = team_odds[team_odds['opponent'].str.contains(opponent, case=False, na=False)]
        
        return team_odds
    
    def get_gameweek_odds(self, gameweek: int) -> pd.DataFrame:
        """
        Get all odds for a specific gameweek.
        
        Args:
            gameweek: Gameweek number
            
        Returns:
            DataFrame with gameweek odds
        """
        odds_df = self.load_odds_data()
        
        if odds_df.empty:
            return pd.DataFrame()
        
        return odds_df[odds_df['gw'] == gameweek]
    
    def calculate_derived_metrics(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived metrics from odds data.
        
        Args:
            odds_df: DataFrame with odds and probabilities
            
        Returns:
            DataFrame with additional derived metrics
        """
        if odds_df.empty:
            return odds_df
        
        derived_df = odds_df.copy()
        
        # Calculate expected goals based on probabilities
        if 'over_2_5_prob' in derived_df.columns:
            # Simple approximation: if over 2.5 is likely, expect higher goals
            derived_df['expected_goals_for'] = 1.5 + derived_df['over_2_5_prob'] * 1.0
            derived_df['expected_goals_against'] = 1.0 + (1 - derived_df.get('clean_sheet_prob', 0.5)) * 1.5
        
        # Calculate attacking/defensive strength
        if 'win_prob' in derived_df.columns:
            derived_df['attack_strength'] = derived_df['win_prob'] * 1.2 + derived_df.get('over_2_5_prob', 0.5) * 0.8
            derived_df['defense_strength'] = derived_df.get('clean_sheet_prob', 0.5) * 1.5 + (1 - derived_df.get('btts_prob', 0.5)) * 0.5
        
        # Calculate fixture difficulty (lower odds = easier fixture)
        if 'win_odds' in derived_df.columns:
            # Invert odds to get difficulty (higher odds = easier fixture)
            derived_df['fixture_difficulty'] = 5.0 - (derived_df['win_odds'] - 1.0) / 2.0
            derived_df['fixture_difficulty'] = derived_df['fixture_difficulty'].clip(1, 5)
        
        return derived_df
    
    def merge_with_fpl_teams(
        self,
        odds_df: pd.DataFrame,
        fpl_teams: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge odds data with FPL team data.
        
        Args:
            odds_df: Odds data
            fpl_teams: FPL team data
            
        Returns:
            Merged DataFrame
        """
        if odds_df.empty or fpl_teams.empty:
            return odds_df
        
        # Create team name mapping
        team_mapping = {}
        for _, team in fpl_teams.iterrows():
            name = team.get('name', '')
            short_name = team.get('short_name', '')
            
            # Map both full and short names
            team_mapping[name.lower()] = team['id']
            team_mapping[short_name.lower()] = team['id']
        
        merged_df = odds_df.copy()
        
        # Map team names to IDs
        merged_df['team_id'] = merged_df['team'].str.lower().map(team_mapping)
        merged_df['opponent_id'] = merged_df['opponent'].str.lower().map(team_mapping)
        
        # Add team information
        team_info = fpl_teams.set_index('id')[['name', 'short_name']].to_dict('index')
        
        merged_df['team_full_name'] = merged_df['team_id'].map(lambda x: team_info.get(x, {}).get('name', ''))
        merged_df['opponent_full_name'] = merged_df['opponent_id'].map(lambda x: team_info.get(x, {}).get('name', ''))
        
        return merged_df
    
    def get_comprehensive_odds(
        self,
        fpl_teams: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get comprehensive odds data with all enhancements.
        
        Args:
            fpl_teams: Optional FPL team data for merging
            
        Returns:
            Complete odds DataFrame
        """
        # Load and validate odds
        odds_df = self.load_odds_data()
        if odds_df.empty:
            return pd.DataFrame()
        
        odds_df = self.validate_odds_data(odds_df)
        if odds_df.empty:
            return pd.DataFrame()
        
        # Convert to probabilities
        odds_df = self.convert_odds_to_probabilities(odds_df)
        
        # Calculate derived metrics
        odds_df = self.calculate_derived_metrics(odds_df)
        
        # Merge with FPL teams if provided
        if fpl_teams is not None:
            odds_df = self.merge_with_fpl_teams(odds_df, fpl_teams)
        
        logger.info(f"Processed comprehensive odds data: {len(odds_df)} rows")
        return odds_df
    
    def save_enhanced_odds(self, odds_df: pd.DataFrame, filename: str = "odds_enhanced.csv") -> bool:
        """
        Save enhanced odds data to file.
        
        Args:
            odds_df: Enhanced odds data
            filename: Output filename
            
        Returns:
            True if saved successfully
        """
        try:
            output_path = self.config.artifacts_dir / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            odds_df.to_csv(output_path, index=False)
            logger.info(f"Saved enhanced odds data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enhanced odds: {e}")
            return False
