"""
Fixtures and scheduling data provider.

Provides functionality for retrieving fixture information, difficulty ratings,
and double gameweek detection.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
from ..common.config import get_config, get_logger
from ..common.cache import get_cache
from .fpl_api import FPLAPIClient

logger = get_logger(__name__)


class FixturesProvider:
    """
    Provider for fixture and scheduling data.
    """
    
    def __init__(self):
        """Initialize fixtures provider."""
        self.config = get_config()
        self.cache = get_cache()
        self.fpl_api = FPLAPIClient()
        
        logger.info("Fixtures provider initialized")
    
    def get_all_fixtures(self) -> Optional[pd.DataFrame]:
        """
        Get all fixtures for the season.
        
        Returns:
            DataFrame with fixture information
        """
        fixtures_data = self.fpl_api.get_fixtures()
        if not fixtures_data:
            return None
        
        # Convert to DataFrame
        fixtures_df = pd.DataFrame(fixtures_data)
        
        # Add derived columns
        if not fixtures_df.empty:
            fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'])
            fixtures_df['is_finished'] = fixtures_df['finished']
            fixtures_df['is_started'] = fixtures_df['started']
            
            # Add home/away indicators
            fixtures_df['home_team_id'] = fixtures_df['team_h']
            fixtures_df['away_team_id'] = fixtures_df['team_a']
            fixtures_df['home_difficulty'] = fixtures_df['team_h_difficulty']
            fixtures_df['away_difficulty'] = fixtures_df['team_a_difficulty']
        
        return fixtures_df
    
    def get_team_fixtures(
        self,
        team_id: int,
        from_gw: Optional[int] = None,
        to_gw: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get fixtures for a specific team.
        
        Args:
            team_id: Team ID
            from_gw: Start gameweek (inclusive)
            to_gw: End gameweek (inclusive)
            
        Returns:
            DataFrame with team's fixtures
        """
        fixtures_df = self.get_all_fixtures()
        if fixtures_df is None or fixtures_df.empty:
            return pd.DataFrame()
        
        # Filter for team (home or away)
        team_fixtures = fixtures_df[
            (fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)
        ].copy()
        
        # Add home/away indicator for the team
        team_fixtures['home_away'] = team_fixtures.apply(
            lambda row: 'H' if row['team_h'] == team_id else 'A',
            axis=1
        )
        
        # Add opponent information
        team_fixtures['opponent_id'] = team_fixtures.apply(
            lambda row: row['team_a'] if row['team_h'] == team_id else row['team_h'],
            axis=1
        )
        
        # Add difficulty for this team
        team_fixtures['difficulty'] = team_fixtures.apply(
            lambda row: row['team_h_difficulty'] if row['team_h'] == team_id else row['team_a_difficulty'],
            axis=1
        )
        
        # Filter by gameweek range
        if from_gw is not None:
            team_fixtures = team_fixtures[team_fixtures['event'] >= from_gw]
        if to_gw is not None:
            team_fixtures = team_fixtures[team_fixtures['event'] <= to_gw]
        
        return team_fixtures.sort_values('event')
    
    def get_gameweek_fixtures(self, gameweek: int) -> pd.DataFrame:
        """
        Get all fixtures for a specific gameweek.
        
        Args:
            gameweek: Gameweek number
            
        Returns:
            DataFrame with gameweek fixtures
        """
        fixtures_df = self.get_all_fixtures()
        if fixtures_df is None or fixtures_df.empty:
            return pd.DataFrame()
        
        return fixtures_df[fixtures_df['event'] == gameweek].copy()
    
    def get_difficulty_matrix(
        self,
        from_gw: int,
        to_gw: int,
        bootstrap_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Create fixture difficulty matrix for all teams.
        
        Args:
            from_gw: Start gameweek
            to_gw: End gameweek
            bootstrap_data: FPL bootstrap data for team names
            
        Returns:
            DataFrame with teams as index and GWs as columns, values are difficulty
        """
        fixtures_df = self.get_all_fixtures()
        if fixtures_df is None or fixtures_df.empty:
            return pd.DataFrame()
        
        # Get team names
        if bootstrap_data is None:
            bootstrap_data = self.fpl_api.get_bootstrap_data()
        
        teams = {}
        if bootstrap_data:
            teams = {team['id']: team['short_name'] for team in bootstrap_data['teams']}
        
        # Filter fixtures by gameweek range
        period_fixtures = fixtures_df[
            (fixtures_df['event'] >= from_gw) & (fixtures_df['event'] <= to_gw)
        ]
        
        # Create matrix
        team_ids = list(teams.keys()) if teams else fixtures_df['team_h'].unique()
        gameweeks = list(range(from_gw, to_gw + 1))
        
        difficulty_matrix = pd.DataFrame(
            index=team_ids,
            columns=gameweeks,
            dtype=float
        )
        
        # Fill matrix
        for _, fixture in period_fixtures.iterrows():
            gw = fixture['event']
            if pd.notna(gw):
                gw = int(gw)
                if gw in gameweeks:
                    # Home team difficulty
                    home_team = fixture['team_h']
                    home_difficulty = fixture['team_h_difficulty']
                    if home_team in difficulty_matrix.index:
                        difficulty_matrix.loc[home_team, gw] = home_difficulty
                    
                    # Away team difficulty
                    away_team = fixture['team_a']
                    away_difficulty = fixture['team_a_difficulty']
                    if away_team in difficulty_matrix.index:
                        difficulty_matrix.loc[away_team, gw] = away_difficulty
        
        # Replace team IDs with names if available
        if teams:
            difficulty_matrix.index = [teams.get(tid, f"Team {tid}") for tid in difficulty_matrix.index]
        
        return difficulty_matrix
    
    def get_double_gameweeks(self) -> List[int]:
        """
        Identify gameweeks with double fixtures.
        
        Returns:
            List of gameweek numbers with double fixtures
        """
        fixtures_df = self.get_all_fixtures()
        if fixtures_df is None or fixtures_df.empty:
            return []
        
        # Count fixtures per team per gameweek
        fixture_counts = {}
        
        for _, fixture in fixtures_df.iterrows():
            gw = fixture['event']
            if pd.isna(gw):
                continue
                
            gw = int(gw)
            if gw not in fixture_counts:
                fixture_counts[gw] = {}
            
            # Count for both teams
            for team_id in [fixture['team_h'], fixture['team_a']]:
                if pd.notna(team_id):
                    team_id = int(team_id)
                    fixture_counts[gw][team_id] = fixture_counts[gw].get(team_id, 0) + 1
        
        # Find gameweeks where any team has >1 fixture
        double_gws = []
        for gw, team_counts in fixture_counts.items():
            if any(count > 1 for count in team_counts.values()):
                double_gws.append(gw)
        
        return sorted(double_gws)
    
    def get_blank_gameweeks(self) -> List[int]:
        """
        Identify gameweeks where teams have no fixtures (blanks).
        
        Returns:
            List of gameweek numbers with team blanks
        """
        fixtures_df = self.get_all_fixtures()
        if fixtures_df is None or fixtures_df.empty:
            return []
        
        # Get all gameweeks with fixtures
        gws_with_fixtures = set(fixtures_df['event'].dropna().astype(int))
        
        # Get expected gameweek range (1-38)
        all_gws = set(range(1, 39))
        
        # Find missing gameweeks
        blank_gws = sorted(all_gws - gws_with_fixtures)
        
        return blank_gws
    
    def calculate_fixture_strength(
        self,
        team_id: int,
        gameweeks: List[int],
        weight_recent: bool = True
    ) -> float:
        """
        Calculate fixture difficulty strength for a team over multiple gameweeks.
        
        Args:
            team_id: Team ID
            gameweeks: List of gameweeks to analyze
            weight_recent: Whether to weight recent fixtures more heavily
            
        Returns:
            Average fixture strength (lower = easier)
        """
        fixtures = self.get_team_fixtures(team_id)
        if fixtures.empty:
            return 3.0  # Neutral if no data
        
        # Filter for specified gameweeks
        period_fixtures = fixtures[fixtures['event'].isin(gameweeks)]
        
        if period_fixtures.empty:
            return 3.0
        
        difficulties = period_fixtures['difficulty'].dropna()
        
        if len(difficulties) == 0:
            return 3.0
        
        if weight_recent:
            # Weight more recent fixtures higher
            weights = [1.2 ** i for i in range(len(difficulties))]
            return sum(diff * weight for diff, weight in zip(difficulties, weights)) / sum(weights)
        else:
            return difficulties.mean()
    
    def get_team_fixture_summary(
        self,
        team_id: int,
        next_n_gws: int = 5,
        bootstrap_data: Optional[Dict] = None
    ) -> Dict:
        """
        Get comprehensive fixture summary for a team.
        
        Args:
            team_id: Team ID
            next_n_gws: Number of upcoming gameweeks to analyze
            bootstrap_data: FPL bootstrap data
            
        Returns:
            Dictionary with fixture analysis
        """
        from ..common.timeutil import get_current_gw
        
        current_gw = get_current_gw()
        end_gw = current_gw + next_n_gws
        
        fixtures = self.get_team_fixtures(team_id, current_gw, end_gw)
        
        if bootstrap_data is None:
            bootstrap_data = self.fpl_api.get_bootstrap_data()
        
        # Get team and opponent names
        teams = {}
        if bootstrap_data:
            teams = {team['id']: team for team in bootstrap_data['teams']}
        
        team_info = teams.get(team_id, {})
        
        summary = {
            'team_id': team_id,
            'team_name': team_info.get('name', f'Team {team_id}'),
            'team_short_name': team_info.get('short_name', f'T{team_id}'),
            'fixtures': [],
            'avg_difficulty': 0.0,
            'home_fixtures': 0,
            'away_fixtures': 0,
            'double_gameweeks': [],
            'blank_gameweeks': []
        }
        
        if fixtures.empty:
            return summary
        
        # Process fixtures
        for _, fixture in fixtures.iterrows():
            opponent_info = teams.get(fixture['opponent_id'], {})
            
            fixture_data = {
                'gameweek': int(fixture['event']),
                'opponent_id': int(fixture['opponent_id']),
                'opponent_name': opponent_info.get('short_name', f"Team {fixture['opponent_id']}"),
                'home_away': fixture['home_away'],
                'difficulty': fixture['difficulty'],
                'kickoff_time': fixture.get('kickoff_time'),
                'is_finished': fixture.get('finished', False)
            }
            
            summary['fixtures'].append(fixture_data)
            
            if fixture['home_away'] == 'H':
                summary['home_fixtures'] += 1
            else:
                summary['away_fixtures'] += 1
        
        # Calculate average difficulty
        difficulties = [f['difficulty'] for f in summary['fixtures'] if pd.notna(f['difficulty'])]
        if difficulties:
            summary['avg_difficulty'] = sum(difficulties) / len(difficulties)
        
        # Check for doubles/blanks in the period
        period_gws = list(range(current_gw, end_gw + 1))
        fixture_gws = [int(f['gameweek']) for f in summary['fixtures']]
        
        # Count fixtures per gameweek
        gw_counts = {}
        for gw in fixture_gws:
            gw_counts[gw] = gw_counts.get(gw, 0) + 1
        
        summary['double_gameweeks'] = [gw for gw, count in gw_counts.items() if count > 1]
        summary['blank_gameweeks'] = [gw for gw in period_gws if gw not in fixture_gws]
        
        return summary
