"""
Team data utilities for FPL dashboard.

Provides functionality for loading and managing FPL team data,
including current GW, player prices, fixtures, and team strength.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# Add src to path for imports
app_dir = Path(__file__).parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from fpl_ai.src.common.timeutil import get_current_gw
from fpl_ai.src.providers.fpl_picks import FPLPicksClient
from fpl_ai.src.providers.fpl_api import FPLAPIClient
from fpl_ai.src.providers.fixtures import FixturesProvider
from fpl_ai.src.common.cache import get_cache
from fpl_ai.src.common.config import get_config, get_logger

logger = get_logger(__name__)


class TeamDataManager:
    """Manages FPL team data for dashboard display."""

    def __init__(self):
        """Initialize team data manager."""
        self.config = get_config()
        self.cache = get_cache()
        self.fpl_api = FPLAPIClient()
        self.picks_client = FPLPicksClient()
        self.fixtures_provider = FixturesProvider()

    def validate_team_id(self, team_id: str) -> Tuple[bool, str]:
        """
        Validate FPL team ID format.

        Args:
            team_id: Team ID string

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not team_id or not team_id.strip():
            return False, "Team ID cannot be empty"

        try:
            team_id_int = int(team_id.strip())
            if team_id_int <= 0:
                return False, "Team ID must be a positive number"
            return True, ""
        except ValueError:
            return False, "Team ID must be a valid number"

    def load_team_data(self, team_id: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Load comprehensive team data for a given team ID.

        Args:
            team_id: FPL team ID
            force_refresh: Force refresh from API

        Returns:
            Dictionary with team data or None if failed
        """
        is_valid, error_msg = self.validate_team_id(team_id)
        if not is_valid:
            st.error(f"❌ Invalid team ID: {error_msg}")
            return None

        team_id_int = int(team_id.strip())

        # Try to load from cache first
        cache_key = f"team_data_{team_id_int}"
        cached_data = self.cache.get(cache_key, "team_data", 1800)  # 30 min cache

        if cached_data and not force_refresh:
            logger.info(f"Loaded team data for {team_id_int} from cache")
            return cached_data

        try:
            with st.spinner("🔄 Loading your FPL team data..."):
                # Set entry ID from the provided team_id
                self.fpl_api.entry_id = team_id_int

                # Get current gameweek
                current_gw = get_current_gw()

                # Try to get team picks for current GW, fallback to previous GWs if not available
                team_data = None
                for gw in range(current_gw, max(1, current_gw - 3), -1):  # Try current GW down to 3 GWs ago
                    try:
                        team_data = self.picks_client.get_current_team(gw)
                        if team_data:
                            team_data['gameweek_loaded'] = gw
                            logger.info(f"Loaded team data for GW {gw}")
                            break
                    except Exception as e:
                        logger.debug(f"Failed to load team data for GW {gw}: {e}")
                        continue
                
                if not team_data:
                    st.error("❌ Could not load team data. Please check your team ID and try again.")
                    return None

                # Get bootstrap data for player details
                bootstrap = self.fpl_api.get_bootstrap_data()
                if not bootstrap:
                    st.error("❌ Could not load FPL data. Please try again later.")
                    return None

                # Process team data
                processed_data = self._process_team_data(team_data, bootstrap, current_gw, team_id_int)

                # Cache the data
                self.cache.set(cache_key, processed_data, "team_data", 1800)

                logger.info(f"Successfully loaded and cached team data for {team_id_int}")
                return processed_data

        except Exception as e:
            import traceback
            logger.error(f"Error loading team data for {team_id}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            st.error(f"❌ Error loading team data: {str(e)}")
            # Print to console for debugging
            print(f"❌ TEAM LOADING ERROR: {e}")
            print(f"Full traceback:")
            traceback.print_exc()
            return None

    def _process_team_data(self, team_data: Dict, bootstrap: Dict, current_gw: int, team_id: int) -> Dict:
        """
        Process raw team data into dashboard-friendly format.

        Args:
            team_data: Raw team data from API
            bootstrap: FPL bootstrap data
            current_gw: Current gameweek
            team_id: Team ID

        Returns:
            Processed team data dictionary
        """
        picks_data = team_data.get('picks', {})
        entry_data = team_data.get('entry', {})

        # Create lookup dictionaries
        elements = {elem['id']: elem for elem in bootstrap['elements']}
        teams = {team['id']: team for team in bootstrap['teams']}
        positions = {pos['id']: pos for pos in bootstrap['element_types']}

        # Process squad
        squad = []
        total_value = 0

        # Get actual bank from entry data (this is the real available money)
        bank = 0
        actual_bank = entry_data.get('last_deadline_bank', 0) / 10  # Convert from tenths to millions

        for pick in picks_data.get('picks', []):
            element_id = pick['element']
            if element_id in elements:
                player = elements[element_id]
                team_info = teams.get(player['team'], {})
                position_info = positions.get(player['element_type'], {})

                # Ensure proper price calculation
                now_cost = player['now_cost']
                if now_cost > 20:  # If price is in tenths, convert to millions
                    now_cost = now_cost / 10
                
                player_data = {
                    'element_id': element_id,
                    'team_id': player['team'],
                    'web_name': player['web_name'],
                    'team_name': team_info.get('short_name', f'Team {player["team"]}'),
                    'position': position_info.get('singular_name_short', f'Pos {player["element_type"]}'),
                    'now_cost': now_cost,  # Already converted to millions
                    'selling_price': pick.get('selling_price', 0) / 10,
                    'purchase_price': pick.get('purchase_price', 0) / 10,
                    'is_captain': pick.get('is_captain', False),
                    'is_vice_captain': pick.get('is_vice_captain', False),
                    'multiplier': pick.get('multiplier', 1),
                    'form': float(player.get('form', 0)),
                    'points_per_game': float(player.get('points_per_game', 0)),
                    'total_points': player.get('total_points', 0),
                    'minutes': player.get('minutes', 0),
                    'goals_scored': player.get('goals_scored', 0),
                    'assists': player.get('assists', 0),
                    'clean_sheets': player.get('clean_sheets', 0),
                    'bonus': player.get('bonus', 0)
                }

                squad.append(player_data)
                total_value += player_data['now_cost']  # Sum current prices for team value

        # Get current GW info
        current_gw_info = self._get_current_gw_info(bootstrap, current_gw)

        # Get fixture data for next few GWs
        fixture_data = self._get_team_fixtures(squad, current_gw)

        # Calculate team strength metrics
        team_strength = self._calculate_team_strength(squad, current_gw)

        # Use actual bank from FPL API (this is the real available spending money)
        available_bank = actual_bank
        
        # Calculate team value growth from starting budget
        starting_budget = 100.0
        value_change = total_value - starting_budget
        
        # Log the calculation for debugging  
        logger.info(f"Bank calculation: Available Bank=£{available_bank:.1f}m, Team Value=£{total_value:.1f}m, Value Change={value_change:+.1f}m")
        
        # Positive value change is good news!
        if value_change > 0:
            logger.info(f"✅ Team value has grown by £{value_change:.1f}m since season start - player prices have increased!")

        return {
            'team_id': team_id,
            'team_name': entry_data.get('name', f'Team {team_id}'),
            'current_gw': current_gw,
            'data_gw': team_data.get('gameweek_loaded', current_gw),  # Which GW data was actually loaded
            'gw_info': current_gw_info,
            'squad': squad,
            'bank': available_bank,  # Actual available spending money from FPL API
            'team_value': total_value,  # Sum of current player prices
            'squad_value': total_value,
            'value_change': value_change,  # How much team value has changed since season start
            'starting_budget': starting_budget,  # Original 100.0m budget
            'free_transfers': entry_data.get('free_transfers', 1),
            'fixture_data': fixture_data,
            'team_strength': team_strength,
            'last_updated': pd.Timestamp.now(tz='UTC').isoformat().replace('+00:00', 'Z')
        }

    def _get_current_gw_info(self, bootstrap: Dict, current_gw: int) -> Dict:
        """Get current gameweek information."""
        events = bootstrap.get('events', [])

        for event in events:
            if event['id'] == current_gw:
                return {
                    'id': event['id'],
                    'name': event['name'],
                    'deadline_time': event['deadline_time'],
                    'is_current': event['is_current'],
                    'is_next': event['is_next'],
                    'finished': event['finished'],
                    'data_checked': event['data_checked']
                }

        return {
            'id': current_gw,
            'name': f'Gameweek {current_gw}',
            'deadline_time': None,
            'is_current': True,
            'is_next': False,
            'finished': False,
            'data_checked': False
        }

    def _get_team_fixtures(self, squad: List[Dict], current_gw: int) -> Dict:
        """Get fixture data for the team's players."""
        try:
            # Get unique FPL team IDs for players' clubs
            team_ids = list(set(player['team_id'] for player in squad if 'team_id' in player))

            if not team_ids:
                return {'upcoming': [], 'next_5_gws': []}

            # Get fixtures for next 5 gameweeks
            all_fixtures = []
            for team_id in team_ids:
                try:
                    team_fixtures = self.fixtures_provider.get_team_fixtures(team_id, current_gw, current_gw + 5)
                    if not team_fixtures.empty:
                        all_fixtures.append(team_fixtures)
                except Exception as e:
                    logger.warning(f"Error getting fixtures for team {team_id}: {e}")
                    continue

            if not all_fixtures:
                return {'upcoming': [], 'next_5_gws': []}

            # Combine all fixtures
            combined_fixtures = pd.concat(all_fixtures, ignore_index=True)

            # Remove duplicates and sort
            combined_fixtures = combined_fixtures.drop_duplicates(subset=['event', 'team_h', 'team_a'])
            combined_fixtures = combined_fixtures.sort_values(['event', 'kickoff_time'])

            # Get bootstrap data for team names
            bootstrap = self.fpl_api.get_bootstrap_data()
            teams = {team['id']: team for team in bootstrap['teams']} if bootstrap else {}

            upcoming_fixtures = []
            for _, fixture in combined_fixtures.head(10).iterrows():  # Next 10 fixtures
                fixture_data = {
                    'gameweek': int(fixture['event']),
                    'home_team': teams.get(fixture['team_h'], {}).get('short_name', f'Team {fixture["team_h"]}'),
                    'away_team': teams.get(fixture['team_a'], {}).get('short_name', f'Team {fixture["team_a"]}'),
                    'kickoff_time': fixture.get('kickoff_time'),
                    'difficulty': fixture.get('difficulty', 3)
                }
                upcoming_fixtures.append(fixture_data)

            return {
                'upcoming': upcoming_fixtures,
                'next_5_gws': upcoming_fixtures[:15]  # Next 15 fixtures for analysis
            }

        except Exception as e:
            logger.error(f"Error getting team fixtures: {e}")
            return {'upcoming': [], 'next_5_gws': []}

    def _calculate_team_strength(self, squad: List[Dict], current_gw: int) -> Dict:
        """Calculate team strength metrics."""
        try:
            # Calculate basic stats
            total_form = sum(float(player.get('form', 0)) for player in squad if player.get('form'))
            avg_form = total_form / len(squad) if squad else 0

            total_ppg = sum(float(player.get('points_per_game', 0)) for player in squad if player.get('points_per_game'))
            avg_ppg = total_ppg / len(squad) if squad else 0

            # Count by position
            position_counts = {}
            for player in squad:
                pos = player.get('position', 'Unknown')
                position_counts[pos] = position_counts.get(pos, 0) + 1

            # Calculate fixture difficulty (simplified)
            fixture_difficulty = 3.0  # Neutral default
            if squad:
                # Get team IDs and calculate average fixture difficulty
                team_ids = [player['element_id'] for player in squad if 'element_id' in player]
                difficulties = []

                for team_id in team_ids[:5]:  # Check first 5 players' teams
                    try:
                        team_fixtures = self.fixtures_provider.get_team_fixtures(team_id, current_gw, current_gw + 2)
                        if not team_fixtures.empty:
                            avg_diff = team_fixtures['difficulty'].mean()
                            difficulties.append(avg_diff)
                    except:
                        continue

                if difficulties:
                    fixture_difficulty = sum(difficulties) / len(difficulties)

            return {
                'avg_form': round(avg_form, 2),
                'avg_ppg': round(avg_ppg, 2),
                'position_counts': position_counts,
                'fixture_difficulty': round(fixture_difficulty, 2),
                'squad_size': len(squad),
                'total_value': sum(player.get('now_cost', 0) for player in squad)
            }

        except Exception as e:
            logger.error(f"Error calculating team strength: {e}")
            return {
                'avg_form': 0.0,
                'avg_ppg': 0.0,
                'position_counts': {},
                'fixture_difficulty': 3.0,
                'squad_size': len(squad),
                'total_value': 0.0
            }


def display_team_data(team_data: Dict):
    """Display team data in Streamlit components."""

    if not team_data or not team_data.get('squad'):
        st.warning("⚠️ No team data available")
        return

    # Team Overview
    st.markdown("### 👤 Your Team Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Team Value", f"£{team_data['team_value']:.1f}m")

    with col2:
        st.metric("Bank", f"£{team_data['bank']:.1f}m")

    with col3:
        st.metric("Free Transfers", team_data['free_transfers'])

    with col4:
        st.metric("Squad Size", f"{team_data['team_strength']['squad_size']}/15")

    # Current Gameweek Info
    st.markdown("### 🗓️ Current Gameweek")
    gw_info = team_data['gw_info']

    gw_col1, gw_col2, gw_col3 = st.columns(3)

    with gw_col1:
        status = "✅ Finished" if gw_info['finished'] else "🔄 Active" if gw_info['is_current'] else "⏳ Upcoming"
        st.metric("GW Status", status)

    with gw_col2:
        st.metric("Gameweek", f"GW {gw_info['id']}")

    with gw_col3:
        if gw_info.get('deadline_time'):
            try:
                # Handle timezone-aware datetime from FPL API
                deadline = pd.to_datetime(gw_info['deadline_time'], utc=True)
                # Convert to local timezone for display
                local_deadline = deadline.tz_convert(None) if deadline.tzinfo else deadline
                st.metric("Deadline", local_deadline.strftime("%d/%m %H:%M"))
            except Exception as e:
                logger.warning(f"Error parsing deadline time: {e}")
                st.metric("Deadline", "TBD")
        else:
            st.metric("Deadline", "TBD")

    # Team Strength
    st.markdown("### 📊 Team Strength")

    strength = team_data['team_strength']

    str_col1, str_col2, str_col3, str_col4 = st.columns(4)

    with str_col1:
        st.metric("Avg Form", f"{strength['avg_form']:.2f}")

    with str_col2:
        st.metric("Avg PPG", f"{strength['avg_ppg']:.2f}")

    with str_col3:
        st.metric("Fixture Difficulty", f"{strength['fixture_difficulty']:.1f}/5")

    with str_col4:
        st.metric("Total Value", f"£{strength['total_value']:.1f}m")

    # Position Breakdown
    st.markdown("#### Position Counts")
    pos_counts = strength['position_counts']
    pos_cols = st.columns(len(pos_counts))

    for i, (pos, count) in enumerate(pos_counts.items()):
        with pos_cols[i]:
            st.metric(pos, count)

    # Squad List
    st.markdown("### 📋 Your Squad")

    squad_df = pd.DataFrame(team_data['squad'])

    # Format for display
    display_cols = [
        'web_name', 'team_name', 'position', 'now_cost',
        'form', 'points_per_game', 'total_points'
    ]

    display_df = squad_df[display_cols].copy()
    display_df.columns = ['Name', 'Team', 'Pos', 'Price (£m)', 'Form', 'PPG', 'Total Pts']

    # Add captain/vice captain indicators
    display_df['Role'] = ''
    for idx, player in squad_df.iterrows():
        if player['is_captain']:
            display_df.loc[idx, 'Role'] = 'C'
        elif player['is_vice_captain']:
            display_df.loc[idx, 'Role'] = 'VC'

    display_df = display_df[['Name', 'Team', 'Pos', 'Role', 'Price (£m)', 'Form', 'PPG', 'Total Pts']]

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Price (£m)": st.column_config.NumberColumn(format="£%.1f"),
            "Form": st.column_config.NumberColumn(format="%.2f"),
            "PPG": st.column_config.NumberColumn(format="%.2f"),
            "Total Pts": st.column_config.NumberColumn(format="%.0f")
        }
    )

    # Upcoming Fixtures
    st.markdown("### 📅 Upcoming Fixtures")

    fixtures = team_data.get('fixture_data', {}).get('upcoming', [])

    if fixtures:
        fixture_df = pd.DataFrame(fixtures)

        # Create display dataframe
        display_fixtures = []
        for fixture in fixtures:
            try:
                # Handle timezone-aware datetime from fixtures
                kickoff_time = fixture.get('kickoff_time')
                if kickoff_time:
                    kickoff_dt = pd.to_datetime(kickoff_time, utc=True)
                    # Convert to local timezone for display
                    local_time = kickoff_dt.tz_convert(None) if kickoff_dt.tzinfo else kickoff_dt
                    time_str = local_time.strftime('%d/%m %H:%M')
                else:
                    time_str = 'TBD'
            except Exception as e:
                logger.warning(f"Error parsing fixture time: {e}")
                time_str = 'TBD'

            display_fixtures.append({
                'GW': fixture['gameweek'],
                'Match': f"{fixture['home_team']} vs {fixture['away_team']}",
                'Difficulty': f"{fixture['difficulty']}/5",
                'Time': time_str
            })

        fixture_display_df = pd.DataFrame(display_fixtures)

        st.dataframe(
            fixture_display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No upcoming fixture data available")


def display_player_prices(team_data: Dict):
    """Display player price information."""

    if not team_data or not team_data.get('squad'):
        st.warning("⚠️ No player price data available")
        return

    st.markdown("### 💰 Player Prices")

    squad_df = pd.DataFrame(team_data['squad'])

    # Calculate price changes
    price_data = []
    for _, player in squad_df.iterrows():
        current_price = player['now_cost']
        selling_price = player['selling_price']

        price_change = current_price - selling_price
        price_change_pct = (price_change / selling_price * 100) if selling_price > 0 else 0

        price_data.append({
            'Player': player['web_name'],
            'Team': player['team_name'],
            'Position': player['position'],
            'Current Price': current_price,
            'Selling Price': selling_price,
            'Price Change': price_change,
            'Change %': price_change_pct
        })

    price_df = pd.DataFrame(price_data)

    # Sort by price change
    price_df = price_df.sort_values('Change %', ascending=False)

    st.dataframe(
        price_df,
        use_container_width=True,
        column_config={
            "Current Price": st.column_config.NumberColumn(format="£%.1f"),
            "Selling Price": st.column_config.NumberColumn(format="£%.1f"),
            "Price Change": st.column_config.NumberColumn(format="£%.1f"),
            "Change %": st.column_config.NumberColumn(format="%.1f%%")
        },
        hide_index=True
    )


# Global instance for reuse
_team_manager = None

def get_team_manager() -> TeamDataManager:
    """Get global team data manager instance."""
    global _team_manager
    if _team_manager is None:
        _team_manager = TeamDataManager()
    return _team_manager
