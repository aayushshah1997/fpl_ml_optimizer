"""
Injury and availability data provider.

Provides functionality for retrieving player injury status and availability
probabilities for match selection.
"""

import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ..common.config import get_config, get_logger
from ..common.cache import get_cache

logger = get_logger(__name__)


class InjuryProvider:
    """
    Provider for player injury and availability data.
    """
    
    def __init__(self):
        """Initialize injury provider."""
        self.config = get_config()
        self.cache = get_cache()
        
        # Default availability probabilities by status
        self.status_probabilities = {
            'available': 0.95,
            'fit': 0.95,
            '75% chance of playing': 0.75,
            '50% chance of playing': 0.50,
            '25% chance of playing': 0.25,
            'doubtful': 0.25,
            'unlikely': 0.15,
            'injured': 0.05,
            'suspended': 0.0,
            'unavailable': 0.0
        }
        
        logger.info("Injury provider initialized")
    
    def get_fpl_availability_data(self, bootstrap_data: Dict) -> pd.DataFrame:
        """
        Extract availability data from FPL bootstrap data.
        
        Args:
            bootstrap_data: FPL bootstrap static data
            
        Returns:
            DataFrame with player availability information
        """
        if not bootstrap_data or 'elements' not in bootstrap_data:
            return pd.DataFrame()
        
        players = []
        
        for player in bootstrap_data['elements']:
            # Extract availability info
            status = player.get('status', 'a')  # 'a' = available
            chance_of_playing_this_round = player.get('chance_of_playing_this_round')
            chance_of_playing_next_round = player.get('chance_of_playing_next_round')
            news = player.get('news', '')
            
            # Convert status codes to text
            status_map = {
                'a': 'available',
                'd': 'doubtful', 
                'i': 'injured',
                's': 'suspended',
                'u': 'unavailable',
                'n': 'not available'
            }
            
            status_text = status_map.get(status, 'available')
            
            # Calculate availability probability
            if chance_of_playing_this_round is not None:
                avail_prob = chance_of_playing_this_round / 100
            else:
                avail_prob = self.status_probabilities.get(status_text, 0.95)
            
            players.append({
                'element_id': player['id'],
                'web_name': player['web_name'],
                'team_id': player['team'],
                'status': status_text,
                'chance_this_round': chance_of_playing_this_round,
                'chance_next_round': chance_of_playing_next_round,
                'news': news,
                'avail_prob': avail_prob
            })
        
        return pd.DataFrame(players)
    
    def parse_injury_news(self, news_text: str) -> Tuple[str, float]:
        """
        Parse injury news text to extract status and probability.
        
        Args:
            news_text: News/injury text
            
        Returns:
            Tuple of (parsed_status, probability)
        """
        if not news_text:
            return 'available', 0.95
        
        news_lower = news_text.lower()
        
        # Look for percentage chances
        pct_match = re.search(r'(\d+)%.*?chance.*?playing', news_lower)
        if pct_match:
            pct = int(pct_match.group(1))
            return f'{pct}% chance of playing', pct / 100
        
        # Look for specific keywords
        if any(word in news_lower for word in ['suspended', 'banned', 'red card']):
            return 'suspended', 0.0
        
        if any(word in news_lower for word in ['injured', 'injury', 'hurt', 'strain', 'tear']):
            # Check severity
            if any(word in news_lower for word in ['serious', 'long', 'major', 'surgery']):
                return 'injured', 0.05
            elif any(word in news_lower for word in ['minor', 'slight', 'small']):
                return 'doubtful', 0.25
            else:
                return 'injured', 0.15
        
        if any(word in news_lower for word in ['doubtful', 'doubt', 'uncertain']):
            return 'doubtful', 0.25
        
        if any(word in news_lower for word in ['unlikely', 'probably not']):
            return 'unlikely', 0.15
        
        if any(word in news_lower for word in ['fitness test', 'late decision']):
            return '50% chance of playing', 0.50
        
        if any(word in news_lower for word in ['expected', 'likely', 'should play']):
            return '75% chance of playing', 0.75
        
        if any(word in news_lower for word in ['fit', 'available', 'ready']):
            return 'available', 0.95
        
        # Default for any news (slightly reduced from full availability)
        return 'available', 0.85
    
    def get_external_injury_data(self) -> Optional[pd.DataFrame]:
        """
        PLACEHOLDER: Future feature not yet implemented.
        
        Planned functionality: Scrape external injury data sources including:
        - Premier League official injury reports
        - BBC Sport injury list
        - Sky Sports injury updates
        - Fantasy Football Scout injury news
        - Team official social media updates
        
        Status: Stub/placeholder - returns empty DataFrame
        Implementation priority: Medium - would improve availability predictions
        """
        logger.info("External injury data scraping not implemented")
        return pd.DataFrame()
    
    def merge_availability_sources(
        self,
        fpl_data: pd.DataFrame,
        external_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge availability data from multiple sources.
        
        Args:
            fpl_data: FPL availability data
            external_data: External injury data
            
        Returns:
            Merged availability DataFrame
        """
        merged_data = fpl_data.copy()
        
        if external_data is not None and not external_data.empty:
            # Merge external data (implementation depends on external data structure)
            # This is a placeholder for the merge logic
            logger.info(f"Merging {len(external_data)} external injury records")
        
        # Add confidence scores
        merged_data['confidence'] = merged_data['avail_prob'].apply(
            lambda x: 'high' if x >= 0.8 else 'medium' if x >= 0.5 else 'low'
        )
        
        return merged_data
    
    def get_rotation_risk(
        self,
        player_data: pd.DataFrame,
        recent_minutes: pd.Series,
        team_congestion: pd.Series
    ) -> pd.Series:
        """
        Calculate rotation risk based on recent minutes and team congestion.
        
        Args:
            player_data: Player data with positions
            recent_minutes: Recent minutes played
            team_congestion: Team fixture congestion indicator
            
        Returns:
            Series with rotation risk scores (0-1, higher = more risk)
        """
        if len(player_data) == 0:
            return pd.Series(dtype=float)
        
        rotation_risk = pd.Series(0.0, index=player_data.index)
        
        # Base risk by position
        position_risk = {
            'GK': 0.1,   # Goalkeepers rarely rotated
            'DEF': 0.2,  # Defenders moderately rotated
            'MID': 0.3,  # Midfielders more rotated
            'FWD': 0.25  # Forwards moderately rotated
        }
        
        for idx, player in player_data.iterrows():
            pos = player.get('position', 'MID')
            base_risk = position_risk.get(pos, 0.25)
            
            # Adjust based on recent minutes
            if idx in recent_minutes.index:
                mins_val = recent_minutes.loc[idx]
                # Ensure scalar value
                if hasattr(mins_val, 'iloc'):
                    mins = mins_val.iloc[0] if len(mins_val) > 0 else 0
                else:
                    mins = float(mins_val) if mins_val is not None else 0
                    
                if mins >= 270:  # Played 90+ mins in last 3 games
                    mins_factor = 1.5  # Higher rotation risk
                elif mins >= 180:  # Played 60+ mins average
                    mins_factor = 1.0  # Normal risk
                else:  # Low minutes
                    mins_factor = 0.5  # Lower rotation risk
            else:
                mins_factor = 1.0
            
            # Adjust for team congestion
            if idx in team_congestion.index:
                congestion_val = team_congestion.loc[idx]
                # Ensure scalar value
                if hasattr(congestion_val, 'iloc'):
                    congestion = congestion_val.iloc[0] if len(congestion_val) > 0 else 0
                else:
                    congestion = float(congestion_val) if congestion_val is not None else 0
                    
                congestion_factor = 1.0 + (congestion * 0.3)  # Up to 30% increase
            else:
                congestion_factor = 1.0
            
            # Ensure all values are scalars before calculation
            final_risk = float(base_risk) * float(mins_factor) * float(congestion_factor)
            rotation_risk.loc[idx] = min(0.8, final_risk)
        
        return rotation_risk
    
    def calculate_final_availability(
        self,
        availability_data: pd.DataFrame,
        rotation_risk: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate final availability probabilities combining injury and rotation.
        
        Args:
            availability_data: Base availability data
            rotation_risk: Optional rotation risk scores
            
        Returns:
            DataFrame with final availability probabilities
        """
        final_data = availability_data.copy()
        
        if rotation_risk is not None:
            # Combine injury availability with rotation risk
            final_data = final_data.join(rotation_risk.rename('rotation_risk'), how='left')
            final_data['rotation_risk'] = final_data['rotation_risk'].fillna(0.25)
            
            # Final availability = injury_prob * (1 - rotation_risk)
            final_data['final_avail_prob'] = (
                final_data['avail_prob'] * (1 - final_data['rotation_risk'])
            )
        else:
            final_data['final_avail_prob'] = final_data['avail_prob']
        
        # Add availability categories
        final_data['avail_category'] = pd.cut(
            final_data['final_avail_prob'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['unlikely', 'doubtful', 'possible', 'likely']
        )
        
        return final_data
    
    def get_comprehensive_availability(
        self,
        bootstrap_data: Dict,
        player_data: Optional[pd.DataFrame] = None,
        recent_minutes: Optional[pd.Series] = None,
        team_congestion: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Get comprehensive availability analysis.
        
        Args:
            bootstrap_data: FPL bootstrap data
            player_data: Player information
            recent_minutes: Recent minutes data
            team_congestion: Team congestion data
            
        Returns:
            Complete availability analysis
        """
        # Get FPL availability data
        fpl_availability = self.get_fpl_availability_data(bootstrap_data)
        
        # Get external data (if available)
        external_data = self.get_external_injury_data()
        
        # Merge sources
        merged_availability = self.merge_availability_sources(fpl_availability, external_data)
        
        # Calculate rotation risk
        rotation_risk = None
        if player_data is not None and recent_minutes is not None:
            rotation_risk = self.get_rotation_risk(player_data, recent_minutes, team_congestion)
        
        # Calculate final availability
        final_availability = self.calculate_final_availability(merged_availability, rotation_risk)
        
        logger.info(f"Calculated availability for {len(final_availability)} players")
        return final_availability
