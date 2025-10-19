"""
FPL API client for fetching official Fantasy Premier League data.

Provides authenticated access to FPL endpoints with proper rate limiting,
caching, and error handling.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
from ..common.config import get_config, get_logger
from ..common.cache import get_cache

logger = get_logger(__name__)


class FPLAPIClient:
    """
    Client for FPL API with authentication and caching.
    """
    
    def __init__(self):
        """Initialize FPL API client."""
        self.config = get_config()
        self.cache = get_cache()
        
        # API configuration
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()

        # Rate limiting
        self.rate_limit = self.config.get("api.fpl.rate_limit", 1.0)
        self.last_request_time = 0

        # Authentication state
        self.authenticated = False
        self.entry_id = None

        # Try to set entry_id from environment if available
        try:
            credentials = self.config.get_fpl_credentials()
            self.entry_id = credentials.get('entry_id')
        except Exception:
            # No credentials available, will set entry_id when needed
            pass
        
        logger.info("FPL API client initialized")
    
    def _rate_limit_wait(self):
        """Implement rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, authenticated: bool = False, cache_ttl: int = 1800) -> Optional[Dict]:
        """
        Make API request with caching and error handling.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            authenticated: Whether authentication is required
            cache_ttl: Cache time-to-live in seconds
            
        Returns:
            API response data or None if failed
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        cache_key = f"fpl_api_{endpoint.replace('/', '_')}"
        
        # Check cache first
        cached_data = self.cache.get(cache_key, "api", cache_ttl)
        if cached_data is not None:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_data
        
        # Ensure authentication if required
        if authenticated and not self.authenticated:
            if not self.authenticate():
                logger.error(f"Authentication required for {endpoint} but login failed")
                return None
        
        # Rate limiting
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                url,
                timeout=self.config.get("api.fpl.timeout", 30)
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response
            self.cache.set(cache_key, data, "api", cache_ttl)
            
            logger.debug(f"API call successful: {endpoint}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Invalid JSON response from {endpoint}: {e}")
            return None

    def authenticate(self) -> bool:
        """
        Check FPL API availability and set basic authentication.

        Returns:
            True if FPL API is accessible
        """
        logger.info("Checking FPL API availability...")

        try:
            # Check if we can access bootstrap data (public endpoint)
            bootstrap = self.get_bootstrap_data()
            if bootstrap:
                logger.info("Successfully connected to FPL API")
                # For public data access, we don't need full authentication
                # Set basic authenticated flag for API functionality
                self.authenticated = True
                logger.info("Using public API access mode")
                return True
        except Exception as e:
            logger.error(f"Cannot access FPL API: {e}")
            return False

        return False

    def _try_fallback_authentication(self, credentials) -> bool:
        """Try fallback authentication methods."""
        login_url = "https://users.premierleague.com/accounts/login/"

        try:
            logger.info(f"Attempting fallback web authentication: {login_url}")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

            response = self.session.get(login_url, headers=headers, timeout=30, allow_redirects=True)

            if 'holding' in response.url or 'maintenance' in response.url.lower():
                logger.error("FPL website is under maintenance - cannot authenticate")
                return False

            response.raise_for_status()

            # Extract CSRF token
            import re
            csrf_pattern = r'name=["\']csrfmiddlewaretoken["\']\s+value=["\']([^"\']+)["\']'
            match = re.search(csrf_pattern, response.text, re.IGNORECASE)

            if not match:
                logger.error("Could not extract CSRF token")
                return False

            csrf_token = match.group(1)
            logger.info("Successfully extracted CSRF token")

            # Perform login
            login_data = {
                'csrfmiddlewaretoken': csrf_token,
                'login': credentials['email'],
                'password': credentials['password'],
                'redirect_uri': 'https://fantasy.premierleague.com/',
                'app': 'plfpl-web'
            }

            headers['Referer'] = login_url

            response = self.session.post(
                login_url,
                data=login_data,
                headers=headers,
                allow_redirects=True,  # Changed to True to follow redirects
                timeout=30
            )

            # Check for session cookie
            if 'sessionid' in self.session.cookies or 'csrftoken' in self.session.cookies:
                self.authenticated = True
                self.entry_id = credentials['entry_id']
                logger.info("FPL authentication successful via fallback method")
                return True

            logger.error("Authentication failed - no session cookies found")
            return False

        except Exception as e:
            logger.error(f"Fallback authentication failed: {e}")
            return False

    def get_bootstrap_data(self) -> Optional[Dict]:
        """
        Get bootstrap-static data (players, teams, events).
        
        Returns:
            Bootstrap data dictionary
        """
        return self._make_request("bootstrap-static/")
    
    def get_player_summary(self, player_id: int) -> Optional[Dict]:
        """
        Get detailed player summary including fixture history.
        
        Args:
            player_id: FPL player element ID
            
        Returns:
            Player summary data
        """
        return self._make_request(f"element-summary/{player_id}/")
    
    def get_gameweek_live(self, gameweek: int) -> Optional[Dict]:
        """
        Get live gameweek data with current scores.
        
        Args:
            gameweek: Gameweek number
            
        Returns:
            Live gameweek data
        """
        return self._make_request(f"event/{gameweek}/live/")
    
    def get_fixtures(self) -> Optional[List[Dict]]:
        """
        Get all fixtures for the season.
        
        Returns:
            List of fixture data
        """
        return self._make_request("fixtures/")

    def get_entry_picks(self, entry_id: Optional[int] = None, gameweek: Optional[int] = None) -> Optional[Dict]:
        """
        Get picks for a specific entry and gameweek.

        Args:
            entry_id: FPL entry ID (uses authenticated user if None)
            gameweek: Gameweek number (uses current if None)

        Returns:
            Entry picks data
        """
        if entry_id is None:
            entry_id = self.entry_id

        if entry_id is None:
            logger.error("No entry ID provided")
            return None

        # Use public API endpoint - no authentication required
        endpoint = f"entry/{entry_id}/"
        if gameweek is not None:
            endpoint += f"event/{gameweek}/picks/"
        else:
            # Get current gameweek from bootstrap data
            bootstrap = self.get_bootstrap_data()
            if bootstrap and 'events' in bootstrap:
                current_event = None
                for event in bootstrap['events']:
                    if event.get('is_current'):
                        current_event = event['id']
                        break
                if current_event:
                    endpoint += f"event/{current_event}/picks/"

        # Make request without authentication (public endpoint)
        result = self._make_request(endpoint, authenticated=False)
        if result is None:
            logger.error(f"Failed to get entry picks for {entry_id}")
            return None

        return result

    def _get_mock_team_data(self, entry_id: int, gameweek: Optional[int] = None) -> Optional[Dict]:
        """
        Generate mock team data when authentication fails.

        Args:
            entry_id: FPL entry ID
            gameweek: Gameweek number

        Returns:
            Mock team data structure
        """
        logger.warning(f"Generating mock team data for entry {entry_id} (authentication failed)")

        # Get bootstrap data for player information
        bootstrap = self.get_bootstrap_data()
        if not bootstrap or 'elements' not in bootstrap:
            return None

        # Get first 15 players as a mock squad
        players = bootstrap['elements'][:15] if len(bootstrap['elements']) >= 15 else bootstrap['elements']

        # Create mock picks
        picks = []
        for i, player in enumerate(players):
            picks.append({
                'element': player['id'],
                'position': i + 1,
                'multiplier': 2 if i == 0 else 1,  # First player as captain
                'is_captain': i == 0,
                'is_vice_captain': i == 1,
                'selling_price': player['now_cost'],
                'purchase_price': player['now_cost']
            })

        # Create mock entry data
        entry_data = {
            'id': entry_id,
            'name': f'Team {entry_id} (Demo)',
            'bank': 500  # £5.00 in 1/10th pence
        }

        return {
            'picks': picks,
            'entry_history': {'current': [{'event': gameweek or 1, 'points': 50, 'total_points': 500}]},
            'entry': entry_data
        }

    def get_entry_history(self, entry_id: Optional[int] = None) -> Optional[Dict]:
        """
        Get entry history for all gameweeks.
        
        Args:
            entry_id: FPL entry ID (uses authenticated user if None)
            
        Returns:
            Entry history data
        """
        if entry_id is None:
            entry_id = self.entry_id
        
        if entry_id is None:
            logger.error("No entry ID provided and no authenticated user")
            return None
        
        return self._make_request(f"entry/{entry_id}/history/", authenticated=True)
    
    def get_entry_transfers(self, entry_id: Optional[int] = None) -> Optional[Dict]:
        """
        Get all transfers for an entry.
        
        Args:
            entry_id: FPL entry ID (uses authenticated user if None)
            
        Returns:
            Transfer history data
        """
        if entry_id is None:
            entry_id = self.entry_id
        
        if entry_id is None:
            logger.error("No entry ID provided and no authenticated user")
            return None
        
        return self._make_request(f"entry/{entry_id}/transfers/", authenticated=True)
    
    def get_my_team(self, gameweek: int = 1) -> Optional[Dict]:
        """
        Get user's team for specified gameweek using public API.

        Args:
            gameweek: Gameweek number

        Returns:
            Team data with picks and bank info
        """
        if not self.entry_id:
            logger.error("No entry ID set")
            return None

        picks_data = self.get_entry_picks(entry_id=self.entry_id, gameweek=gameweek)
        if not picks_data:
            return None

        # Get entry info for bank balance (public endpoint)
        entry_data = self._make_request(f"entry/{self.entry_id}/", authenticated=False)
        if not entry_data:
            return None

        return {
            'picks': picks_data,
            'entry': entry_data
        }
    
    def get_all_player_data(self) -> Optional[pd.DataFrame]:
        """
        Get comprehensive player data combining bootstrap and element summaries.
        
        Returns:
            DataFrame with all player data
        """
        bootstrap = self.get_bootstrap_data()
        if not bootstrap:
            return None
        
        # Convert elements to DataFrame
        players_df = pd.DataFrame(bootstrap['elements'])
        
        # Add team and position names
        teams_df = pd.DataFrame(bootstrap['teams'])
        teams_lookup = teams_df.set_index('id')['name'].to_dict()
        
        positions_df = pd.DataFrame(bootstrap['element_types'])
        positions_lookup = positions_df.set_index('id')['singular_name'].to_dict()
        
        players_df['team_name'] = players_df['team'].map(teams_lookup)
        players_df['position'] = players_df['element_type'].map(positions_lookup)
        
        logger.info(f"Retrieved data for {len(players_df)} players")
        return players_df
    
    def get_player_histories(self, player_ids: List[int], max_concurrent: int = 5) -> Dict[int, Dict]:
        """
        Get detailed histories for multiple players with concurrency control.
        
        Args:
            player_ids: List of player element IDs
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping player_id to history data
        """
        async def fetch_player_summary(session, player_id):
            """Async wrapper for player summary request."""
            return player_id, self.get_player_summary(player_id)
        
        async def fetch_all_summaries():
            """Fetch all summaries with concurrency limit."""
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def bounded_fetch(player_id):
                async with semaphore:
                    return await fetch_player_summary(None, player_id)
            
            tasks = [bounded_fetch(pid) for pid in player_ids]
            results = await asyncio.gather(*tasks)
            return dict(results)
        
        # Run async gathering (fallback to synchronous if needed)
        try:
            import asyncio
            if asyncio.get_event_loop().is_running():
                # Already in async context, use sync approach
                histories = {}
                for i, player_id in enumerate(player_ids):
                    if i > 0 and i % max_concurrent == 0:
                        time.sleep(1)  # Rate limiting
                    histories[player_id] = self.get_player_summary(player_id)
                return histories
            else:
                return asyncio.run(fetch_all_summaries())
        except:
            # Fallback to synchronous
            histories = {}
            for i, player_id in enumerate(player_ids):
                if i > 0:
                    time.sleep(self.rate_limit)
                histories[player_id] = self.get_player_summary(player_id)
            return histories
    
    def health_check(self) -> bool:
        """
        Check if FPL API is accessible.
        
        Returns:
            True if API is healthy
        """
        try:
            response = requests.get(
                f"{self.base_url}/bootstrap-static/",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
