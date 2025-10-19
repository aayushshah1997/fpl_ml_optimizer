"""
Set piece proxy and keyword detection provider.

Provides functionality for detecting set piece takers through text analysis
and proxy methods when direct data is not available.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup
from ..common.config import get_config, get_logger
from ..common.cache import get_cache

logger = get_logger(__name__)


class SetPieceProxy:
    """
    Provider for set piece detection through proxy methods.
    """
    
    def __init__(self):
        """Initialize set piece proxy."""
        self.config = get_config()
        self.cache = get_cache()
        
        # Keywords for set piece detection
        self.penalty_keywords = {
            'primary': ['penalty', 'penalties', 'pen taker', 'penalty taker', 'from the spot'],
            'secondary': ['converted', 'missed pen', 'penalty scored', 'penalty miss']
        }
        
        self.freekick_keywords = {
            'primary': ['free kick', 'freekick', 'free-kick', 'fk taker', 'direct free kick'],
            'secondary': ['curled', 'whipped', 'dead ball', 'set piece specialist']
        }
        
        self.corner_keywords = {
            'primary': ['corner', 'corners', 'corner kick', 'corner taker'],
            'secondary': ['delivery', 'whipped in', 'corner routine', 'from the corner']
        }
        
        logger.info("Set piece proxy initialized")
    
    def extract_from_news_text(self, news_data: List[Dict]) -> pd.DataFrame:
        """
        Extract set piece information from news text.
        
        Args:
            news_data: List of news articles with player mentions
            
        Returns:
            DataFrame with inferred set piece roles
        """
        setpiece_mentions = []
        
        for article in news_data:
            text = article.get('text', '').lower()
            players_mentioned = article.get('players', [])
            
            # Search for set piece keywords
            for player in players_mentioned:
                player_name = player.get('name', '').lower()
                
                # Check for penalty mentions
                pen_score = self._calculate_keyword_score(text, player_name, self.penalty_keywords)
                
                # Check for free kick mentions
                fk_score = self._calculate_keyword_score(text, player_name, self.freekick_keywords)
                
                # Check for corner mentions  
                corner_score = self._calculate_keyword_score(text, player_name, self.corner_keywords)
                
                if any(score > 0 for score in [pen_score, fk_score, corner_score]):
                    setpiece_mentions.append({
                        'player_name': player.get('name'),
                        'team': player.get('team'),
                        'penalty_score': pen_score,
                        'freekick_score': fk_score,
                        'corner_score': corner_score,
                        'source': 'news_analysis',
                        'confidence': max(pen_score, fk_score, corner_score),
                        'article_date': article.get('date'),
                        'article_title': article.get('title', '')[:100]
                    })
        
        return pd.DataFrame(setpiece_mentions)
    
    def _calculate_keyword_score(
        self, 
        text: str, 
        player_name: str, 
        keywords: Dict[str, List[str]]
    ) -> float:
        """
        Calculate keyword relevance score for a player.
        
        Args:
            text: Article text
            player_name: Player name to search for
            keywords: Dictionary of primary/secondary keywords
            
        Returns:
            Relevance score (0-1)
        """
        if not player_name or player_name not in text:
            return 0.0
        
        # Find player mention positions
        player_positions = [m.start() for m in re.finditer(re.escape(player_name), text)]
        
        if not player_positions:
            return 0.0
        
        score = 0.0
        context_window = 100  # Characters around player mention
        
        for pos in player_positions:
            # Extract context around player mention
            start = max(0, pos - context_window)
            end = min(len(text), pos + len(player_name) + context_window)
            context = text[start:end]
            
            # Check for primary keywords (higher weight)
            for keyword in keywords.get('primary', []):
                if keyword in context:
                    score += 0.8
            
            # Check for secondary keywords (lower weight)
            for keyword in keywords.get('secondary', []):
                if keyword in context:
                    score += 0.3
        
        return min(1.0, score)  # Cap at 1.0
    
    def scrape_team_pages(self, team_urls: Dict[str, str]) -> pd.DataFrame:
        """
        Scrape team pages for set piece information.
        
        Args:
            team_urls: Dictionary mapping team names to URLs
            
        Returns:
            DataFrame with scraped set piece data
        """
        scraped_data = []
        
        for team_name, url in team_urls.items():
            try:
                cache_key = f"team_page_{team_name.replace(' ', '_')}"
                cached_content = self.cache.get(cache_key, "scraping", ttl=86400)  # 24h cache
                
                if cached_content:
                    page_content = cached_content
                else:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    page_content = response.text
                    self.cache.set(cache_key, page_content, "scraping")
                
                # Parse page content
                soup = BeautifulSoup(page_content, 'html.parser')
                
                # Look for set piece mentions in various sections
                text_content = soup.get_text().lower()
                
                # Extract player mentions (this would need team-specific logic)
                players = self._extract_players_from_page(soup, team_name)
                
                # Analyze for set piece roles
                for player in players:
                    player_data = {
                        'player_name': player['name'],
                        'team': team_name,
                        'penalty_score': self._calculate_keyword_score(
                            text_content, player['name'].lower(), self.penalty_keywords
                        ),
                        'freekick_score': self._calculate_keyword_score(
                            text_content, player['name'].lower(), self.freekick_keywords
                        ),
                        'corner_score': self._calculate_keyword_score(
                            text_content, player['name'].lower(), self.corner_keywords
                        ),
                        'source': 'team_page',
                        'url': url
                    }
                    
                    # Only include if any score > 0
                    if any(score > 0 for score in [
                        player_data['penalty_score'],
                        player_data['freekick_score'], 
                        player_data['corner_score']
                    ]):
                        scraped_data.append(player_data)
                
                logger.debug(f"Scraped set piece data for {team_name}")
                
            except Exception as e:
                logger.warning(f"Failed to scrape {team_name} page: {e}")
                continue
        
        return pd.DataFrame(scraped_data)
    
    def _extract_players_from_page(self, soup: BeautifulSoup, team_name: str) -> List[Dict]:
        """
        Extract player names from team page HTML.
        
        Args:
            soup: BeautifulSoup object
            team_name: Team name for context
            
        Returns:
            List of player dictionaries
        """
        players = []
        
        # Look for common player listing patterns
        player_selectors = [
            '.player-name',
            '.squad-player',
            '[data-player]',
            '.player-link',
            'a[href*="player"]'
        ]
        
        for selector in player_selectors:
            elements = soup.select(selector)
            for element in elements:
                name = element.get_text().strip()
                if name and len(name.split()) >= 2:  # At least first and last name
                    players.append({'name': name})
        
        # Fallback: extract from text using common name patterns
        if not players:
            text = soup.get_text()
            # This is a simplified pattern - would need more sophisticated NER
            name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            potential_names = re.findall(name_pattern, text)
            
            # Filter to likely player names (this would need refinement)
            for name in set(potential_names):
                if len(name.split()) == 2:  # Simple filter
                    players.append({'name': name})
        
        return players[:25]  # Limit to reasonable squad size
    
    def analyze_social_media(self, social_data: List[Dict]) -> pd.DataFrame:
        """
        PLACEHOLDER: Future feature not yet implemented.
        
        Planned functionality: Analyze social media posts for set piece information including:
        - Twitter API for team/player tweets about penalties/free kicks
        - Instagram posts analysis for training ground videos
        - YouTube video descriptions from team channels
        - Reddit r/FantasyPL discussions and leaks
        - Fantasy Football Scout community insights
        
        Status: Stub/placeholder - returns empty DataFrame
        Implementation priority: Low - nice to have for edge cases
        """
        logger.info("Social media analysis not implemented")
        return pd.DataFrame()
    
    def infer_from_match_events(self, match_events: pd.DataFrame) -> pd.DataFrame:
        """
        PLACEHOLDER: Future feature not yet implemented.
        
        Planned functionality: Infer set piece takers from match event data including:
        - Penalty kick events to identify penalty takers
        - Free kick events to identify free kick specialists
        - Corner kick events to identify corner takers
        - Match event data from Opta/StatsBomb
        - Historical set piece performance analysis
        
        Args:
            match_events: DataFrame with match events (goals, assists, etc.)
            
        Returns:
            DataFrame with inferred set piece roles
        """
        if match_events.empty:
            return pd.DataFrame()
        
        inferred_roles = []
        
        # Group by player
        for player_name, player_events in match_events.groupby('player_name'):
            
            # Count penalty-related events
            penalty_events = player_events[
                player_events['event_type'].str.contains('penalty', case=False, na=False)
            ]
            
            # Count free kick events
            freekick_events = player_events[
                player_events['event_type'].str.contains('free.?kick', case=False, na=False)
            ]
            
            # Count corner events
            corner_events = player_events[
                player_events['event_type'].str.contains('corner', case=False, na=False)
            ]
            
            # Calculate confidence based on event frequency
            total_matches = player_events['match_id'].nunique()
            
            if total_matches > 0:
                pen_share = len(penalty_events) / max(1, total_matches)
                fk_share = len(freekick_events) / max(1, total_matches)
                corner_share = len(corner_events) / max(1, total_matches)
                
                if any(share > 0.1 for share in [pen_share, fk_share, corner_share]):
                    inferred_roles.append({
                        'player_name': player_name,
                        'team': player_events['team'].iloc[0] if 'team' in player_events.columns else None,
                        'penalty_share': pen_share,
                        'freekick_share': fk_share,
                        'corner_share': corner_share,
                        'matches_analyzed': total_matches,
                        'source': 'match_events',
                        'confidence': max(pen_share, fk_share, corner_share)
                    })
        
        return pd.DataFrame(inferred_roles)
    
    def combine_proxy_sources(
        self,
        news_data: Optional[pd.DataFrame] = None,
        scraped_data: Optional[pd.DataFrame] = None,
        social_data: Optional[pd.DataFrame] = None,
        event_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Combine multiple proxy data sources.
        
        Args:
            news_data: News analysis results
            scraped_data: Scraped page results
            social_data: Social media analysis
            event_data: Match event analysis
            
        Returns:
            Combined and weighted set piece data
        """
        all_sources = []
        
        # Collect all non-empty sources
        for source_name, data in [
            ('news', news_data),
            ('scraped', scraped_data),
            ('social', social_data),
            ('events', event_data)
        ]:
            if data is not None and not data.empty:
                data = data.copy()
                data['source_type'] = source_name
                all_sources.append(data)
        
        if not all_sources:
            return pd.DataFrame()
        
        # Combine all sources
        combined_df = pd.concat(all_sources, ignore_index=True)
        
        # Group by player and aggregate
        aggregated_roles = []
        
        for (player_name, team), group in combined_df.groupby(['player_name', 'team']):
            
            # Weight different sources
            source_weights = {
                'events': 1.0,     # Match events are most reliable
                'news': 0.8,       # News analysis is quite reliable
                'scraped': 0.6,    # Scraped data is moderately reliable
                'social': 0.4      # Social media is least reliable
            }
            
            # Calculate weighted averages
            pen_scores = []
            fk_scores = []
            corner_scores = []
            
            for _, row in group.iterrows():
                weight = source_weights.get(row['source_type'], 0.5)
                
                pen_score = row.get('penalty_score', row.get('penalty_share', 0))
                fk_score = row.get('freekick_score', row.get('freekick_share', 0))
                corner_score = row.get('corner_score', row.get('corner_share', 0))
                
                pen_scores.append(pen_score * weight)
                fk_scores.append(fk_score * weight)
                corner_scores.append(corner_score * weight)
            
            # Calculate final weighted scores
            final_pen = sum(pen_scores) / len(pen_scores) if pen_scores else 0
            final_fk = sum(fk_scores) / len(fk_scores) if fk_scores else 0
            final_corner = sum(corner_scores) / len(corner_scores) if corner_scores else 0
            
            # Calculate overall confidence
            confidence = max(final_pen, final_fk, final_corner)
            num_sources = group['source_type'].nunique()
            
            # Boost confidence if multiple sources agree
            if num_sources > 1:
                confidence = min(1.0, confidence * (1 + 0.2 * (num_sources - 1)))
            
            aggregated_roles.append({
                'player_name': player_name,
                'team': team,
                'pen_share': final_pen,
                'fk_share': final_fk,
                'corner_share': final_corner,
                'confidence': confidence,
                'num_sources': num_sources,
                'sources': list(group['source_type'].unique()),
                'last_updated': pd.Timestamp.now(tz='UTC').isoformat()
            })
        
        result_df = pd.DataFrame(aggregated_roles)
        
        if not result_df.empty:
            # Filter to meaningful results
            result_df = result_df[
                (result_df['pen_share'] >= 0.1) |
                (result_df['fk_share'] >= 0.1) |
                (result_df['corner_share'] >= 0.1)
            ]
        
        logger.info(f"Combined proxy sources: {len(result_df)} players with set piece roles")
        return result_df
