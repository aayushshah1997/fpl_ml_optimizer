"""
FPL Scoring System Implementation

Comprehensive implementation of the Fantasy Premier League scoring system
including position-specific points, bonus points system (BPS), and all FPL rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = get_logger(__name__)

@dataclass
class FPLScoringRules:
    """FPL scoring rules configuration."""
    
    # Basic scoring
    GOAL_POINTS = {
        'GK': 6,
        'DEF': 6, 
        'MID': 5,
        'FWD': 4
    }
    
    ASSIST_POINTS = 3  # Same for all positions
    
    # Clean sheet points (only if team doesn't concede and player plays 60+ minutes)
    CLEAN_SHEET_POINTS = {
        'GK': 4,
        'DEF': 4,
        'MID': 1,  # Midfielders get 1 point for clean sheet
        'FWD': 0   # Forwards don't get clean sheet points
    }
    
    # Minutes played
    MINUTES_60_PLUS = 1  # 1 point for playing 60+ minutes
    MINUTES_90_PLUS = 2  # 2 points for playing full match (90+ minutes)
    
    # Cards and penalties
    YELLOW_CARD = -1
    RED_CARD = -3
    PENALTY_MISSED = -2
    
    # Bonus Points System (BPS) - key actions
    BPS_GOAL = 24
    BPS_ASSIST = 9
    BPS_CLEAN_SHEET = 12
    BPS_SAVE = 2  # per save
    BPS_GOAL_LINE_CLEARANCE = 9
    BPS_PENALTY_GOAL = 12  # All players get 12 points for penalty goals
    BPS_TACKLE = 2  # per tackle won
    BPS_DEFENSIVE_ACTION = 0.2  # per 10 defensive actions (max 2 points)
    BPS_KEY_PASS = 1
    BPS_BIG_CHANCE_CREATED = 3
    BPS_BIG_CHANCE_MISSED = -2
    BPS_ERROR_LEADING_TO_GOAL = -2
    BPS_ERROR_LEADING_TO_GOAL_ATTEMPT = -1
    BPS_PENALTY_SAVE = 15
    BPS_PENALTY_MISS = -6
    BPS_YELLOW_CARD = -3
    BPS_RED_CARD = -9
    BPS_OWN_GOAL = -6
    
    # Goals conceded (negative BPS)
    BPS_GOALS_CONCEDED = -1  # per goal conceded (GK/DEF only)
    
    # Minutes thresholds
    MINUTES_THRESHOLD_60 = 60
    MINUTES_THRESHOLD_90 = 90


class FPLScorer:
    """FPL scoring calculator."""
    
    def __init__(self):
        self.rules = FPLScoringRules()
    
    def calculate_basic_points(self, df: pd.DataFrame) -> pd.Series:
        """Calculate basic FPL points (goals, assists, clean sheets, minutes)."""
        points = pd.Series(0.0, index=df.index)
        
        # Goals (position-specific)
        if 'goals_scored' in df.columns and 'position' in df.columns:
            goal_points = df['position'].map(self.rules.GOAL_POINTS).fillna(0)
            points += df['goals_scored'].fillna(0) * goal_points
        
        # Assists (same for all positions)
        if 'assists' in df.columns:
            points += df['assists'].fillna(0) * self.rules.ASSIST_POINTS
        
        # Clean sheets (position-specific, only if team doesn't concede)
        if 'clean_sheets' in df.columns and 'position' in df.columns and 'goals_conceded' in df.columns:
            clean_sheet_points = df['position'].map(self.rules.CLEAN_SHEET_POINTS).fillna(0)
            # Only give clean sheet points if team didn't concede
            clean_sheet_mask = (df['goals_conceded'].fillna(0) == 0) & (df['clean_sheets'].fillna(0) == 1)
            points += clean_sheet_mask * clean_sheet_points
        
        # Minutes played
        if 'minutes' in df.columns:
            minutes = df['minutes'].fillna(0)
            # 1 point for 60+ minutes
            points += (minutes >= self.rules.MINUTES_THRESHOLD_60).astype(int)
            # Additional 1 point for 90+ minutes (total 2 for full match)
            points += (minutes >= self.rules.MINUTES_THRESHOLD_90).astype(int)
        
        # Cards (negative points)
        if 'yellow_cards' in df.columns:
            points += df['yellow_cards'].fillna(0) * self.rules.YELLOW_CARD
        
        if 'red_cards' in df.columns:
            points += df['red_cards'].fillna(0) * self.rules.RED_CARD
        
        # Penalty missed
        if 'penalties_missed' in df.columns:
            points += df['penalties_missed'].fillna(0) * self.rules.PENALTY_MISSED
        
        return points
    
    def calculate_bps(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bonus Points System (BPS) score."""
        bps = pd.Series(0.0, index=df.index)
        
        # Goals
        if 'goals_scored' in df.columns:
            bps += df['goals_scored'].fillna(0) * self.rules.BPS_GOAL
        
        # Assists
        if 'assists' in df.columns:
            bps += df['assists'].fillna(0) * self.rules.BPS_ASSIST
        
        # Clean sheets
        if 'clean_sheets' in df.columns:
            bps += df['clean_sheets'].fillna(0) * self.rules.BPS_CLEAN_SHEET
        
        # Saves (GK only)
        if 'saves' in df.columns and 'position' in df.columns:
            gk_mask = df['position'] == 'GK'
            bps += gk_mask * df['saves'].fillna(0) * self.rules.BPS_SAVE
        
        # Goal line clearances
        if 'goal_line_clearances' in df.columns:
            bps += df['goal_line_clearances'].fillna(0) * self.rules.BPS_GOAL_LINE_CLEARANCE
        
        # Penalty goals (all players get 12 points)
        if 'penalty_goals' in df.columns:
            bps += df['penalty_goals'].fillna(0) * self.rules.BPS_PENALTY_GOAL
        
        # Tackles
        if 'tackles' in df.columns:
            bps += df['tackles'].fillna(0) * self.rules.BPS_TACKLE
        
        # Defensive actions (clearances, blocks, interceptions)
        if 'clearances_blocks_interceptions' in df.columns:
            defensive_actions = df['clearances_blocks_interceptions'].fillna(0)
            # 0.2 points per 10 defensive actions, max 2 points
            bps += np.minimum(defensive_actions * self.rules.BPS_DEFENSIVE_ACTION, 2.0)
        
        # Key passes
        if 'key_passes' in df.columns:
            bps += df['key_passes'].fillna(0) * self.rules.BPS_KEY_PASS
        
        # Big chances created
        if 'big_chances_created' in df.columns:
            bps += df['big_chances_created'].fillna(0) * self.rules.BPS_BIG_CHANCE_CREATED
        
        # Big chances missed
        if 'big_chances_missed' in df.columns:
            bps += df['big_chances_missed'].fillna(0) * self.rules.BPS_BIG_CHANCE_MISSED
        
        # Errors leading to goal
        if 'errors_leading_to_goal' in df.columns:
            bps += df['errors_leading_to_goal'].fillna(0) * self.rules.BPS_ERROR_LEADING_TO_GOAL
        
        # Errors leading to goal attempt
        if 'errors_leading_to_goal_attempt' in df.columns:
            bps += df['errors_leading_to_goal_attempt'].fillna(0) * self.rules.BPS_ERROR_LEADING_TO_GOAL_ATTEMPT
        
        # Penalty saves (GK only)
        if 'penalty_saves' in df.columns and 'position' in df.columns:
            gk_mask = df['position'] == 'GK'
            bps += gk_mask * df['penalty_saves'].fillna(0) * self.rules.BPS_PENALTY_SAVE
        
        # Penalty misses
        if 'penalties_missed' in df.columns:
            bps += df['penalties_missed'].fillna(0) * self.rules.BPS_PENALTY_MISS
        
        # Cards (negative BPS)
        if 'yellow_cards' in df.columns:
            bps += df['yellow_cards'].fillna(0) * self.rules.BPS_YELLOW_CARD
        
        if 'red_cards' in df.columns:
            bps += df['red_cards'].fillna(0) * self.rules.BPS_RED_CARD
        
        # Own goals
        if 'own_goals' in df.columns:
            bps += df['own_goals'].fillna(0) * self.rules.BPS_OWN_GOAL
        
        # Goals conceded (GK/DEF only)
        if 'goals_conceded' in df.columns and 'position' in df.columns:
            gk_def_mask = df['position'].isin(['GK', 'DEF'])
            bps += gk_def_mask * df['goals_conceded'].fillna(0) * self.rules.BPS_GOALS_CONCEDED
        
        return bps
    
    def calculate_bonus_points(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bonus points based on BPS ranking within each match."""
        if 'bps' not in df.columns:
            df['bps'] = self.calculate_bps(df)
        
        # Group by match (fixture) and calculate bonus points
        if 'fixture' not in df.columns:
            logger.warning("No fixture column found, cannot calculate bonus points")
            return pd.Series(0.0, index=df.index)
        
        bonus_points = pd.Series(0.0, index=df.index)
        
        for fixture_id, fixture_group in df.groupby('fixture'):
            if len(fixture_group) < 2:
                continue
            
            # Sort by BPS (descending)
            sorted_group = fixture_group.sort_values('bps', ascending=False)
            
            # Award bonus points: 3, 2, 1 for top 3 BPS scores
            for i, (idx, row) in enumerate(sorted_group.iterrows()):
                if i == 0:  # Highest BPS
                    bonus_points.loc[idx] = 3
                elif i == 1:  # Second highest BPS
                    bonus_points.loc[idx] = 2
                elif i == 2:  # Third highest BPS
                    bonus_points.loc[idx] = 1
                else:
                    break  # Only top 3 get bonus points
        
        return bonus_points
    
    def calculate_total_points(self, df: pd.DataFrame) -> pd.Series:
        """Calculate total FPL points including basic points and bonus points."""
        # Calculate basic points
        basic_points = self.calculate_basic_points(df)
        
        # Calculate bonus points
        bonus_points = self.calculate_bonus_points(df)
        
        # Total points
        total_points = basic_points + bonus_points
        
        return total_points
    
    def add_fpl_scoring_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add FPL scoring columns to DataFrame."""
        df = df.copy()
        
        # Calculate BPS
        df['bps'] = self.calculate_bps(df)
        
        # Calculate basic points
        df['basic_points'] = self.calculate_basic_points(df)
        
        # Calculate bonus points
        df['bonus_points'] = self.calculate_bonus_points(df)
        
        # Calculate total points
        df['total_points'] = self.calculate_total_points(df)
        
        return df


def apply_fpl_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Apply FPL scoring to a DataFrame."""
    scorer = FPLScorer()
    return scorer.add_fpl_scoring_columns(df)


# Import logger
try:
    from .logging_setup import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
