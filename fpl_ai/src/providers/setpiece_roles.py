"""
Set piece roles manager for combining all sources.

Manages the hierarchy of set piece data sources: overrides > community > inferred,
and provides the final merged roles for modeling.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ..common.config import get_config, get_logger
from ..common.cache import get_cache
from .setpieces_proxy import SetPieceProxy

logger = get_logger(__name__)


class SetPieceRolesManager:
    """
    Manager for set piece roles with hierarchical data sources.
    """
    
    def __init__(self):
        """Initialize set piece roles manager."""
        self.config = get_config()
        self.cache = get_cache()
        self.proxy = SetPieceProxy()
        
        # Data file paths
        self.overrides_path = Path(self.config.data_dir) / "setpiece_overrides.csv"
        self.community_path = Path(self.config.data_dir) / "community_roles.csv"
        self.roles_output_path = Path(self.config.data_dir) / "setpiece_roles.csv"
        
        logger.info("Set piece roles manager initialized")
    
    def load_override_data(self) -> pd.DataFrame:
        """
        Load manual override data (highest priority).
        
        Returns:
            DataFrame with override set piece data
        """
        try:
            if self.overrides_path.exists():
                df = pd.read_csv(self.overrides_path, comment='#')
                # Filter out empty/comment rows
                df = df.dropna(subset=['player_name'])
                df = df[~df['player_name'].str.startswith('#')]
                logger.info(f"Loaded {len(df)} set piece overrides")
                return df
            else:
                logger.warning(f"Override file not found: {self.overrides_path}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load overrides: {e}")
            return pd.DataFrame()
    
    def load_community_data(self) -> pd.DataFrame:
        """
        Load community-sourced data (medium priority).
        
        Returns:
            DataFrame with community set piece data
        """
        try:
            if self.community_path.exists():
                df = pd.read_csv(self.community_path, comment='#')
                # Filter out empty/comment rows
                df = df.dropna(subset=['player_name'])
                df = df[~df['player_name'].str.startswith('#')]
                logger.info(f"Loaded {len(df)} community set piece roles")
                return df
            else:
                logger.warning(f"Community file not found: {self.community_path}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load community data: {e}")
            return pd.DataFrame()
    
    def infer_setpiece_roles(
        self,
        player_data: pd.DataFrame,
        match_events: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Infer set piece roles using EWMA and match events.
        
        Args:
            player_data: Player data with recent statistics
            match_events: Optional match event data
            
        Returns:
            DataFrame with inferred set piece roles
        """
        if player_data.empty:
            return pd.DataFrame()
        
        inferred_roles = []
        
        # EWMA parameters
        alpha = self.config.get("roles.ewma_alpha", 0.75)
        
        # Group by player
        for player_id, group in player_data.groupby('element_id'):
            if len(group) < 2:  # Need at least 2 games for EWMA
                continue
            
            # Sort by date
            group = group.sort_values('kickoff_time')
            
            # Calculate EWMA for key metrics that indicate set piece roles
            penalties_taken = group.get('penalties_order', 0)
            penalties_saved = group.get('penalties_saved', 0)
            
            # Infer penalty role from penalty events
            pen_events = penalties_taken + penalties_saved
            if pen_events.sum() > 0:
                # Use EWMA to weight recent games more
                weights = [alpha ** i for i in range(len(pen_events)-1, -1, -1)]
                weights = np.array(weights) / sum(weights)
                pen_share = min(1.0, np.average(pen_events, weights=weights))
            else:
                pen_share = 0.0
            
            # Infer free kick role from goals/assists and shot stats
            # Players with high shot conversion from outside box likely take FKs
            total_shots = group.get('total_shots', 0)
            goals_scored = group.get('goals_scored', 0)
            
            # Simple heuristic for free kick takers
            if total_shots.sum() > 0 and goals_scored.sum() > 0:
                conversion_rate = goals_scored.sum() / total_shots.sum()
                if conversion_rate > 0.15:  # High conversion suggests set piece specialist
                    fk_share = min(0.8, conversion_rate * 2)
                else:
                    fk_share = 0.0
            else:
                fk_share = 0.0
            
            # Infer corner role from assists and key passes
            assists = group.get('assists', 0)
            key_passes = group.get('key_passes', 0)
            
            # Players with high assist rate and key passes likely take corners
            total_assists = assists.sum()
            total_key_passes = key_passes.sum()
            
            if total_key_passes > 0:
                assist_rate = total_assists / max(1, total_key_passes)
                if assist_rate > 0.2 and total_key_passes > 10:
                    corner_share = min(0.7, assist_rate * 2)
                else:
                    corner_share = 0.0
            else:
                corner_share = 0.0
            
            # Only include if any role is significant
            if any(share >= 0.2 for share in [pen_share, fk_share, corner_share]):
                player_info = group.iloc[-1]  # Most recent game for player info
                
                inferred_roles.append({
                    'player_name': player_info.get('web_name', f"Player_{player_id}"),
                    'team': player_info.get('team_name', 'Unknown'),
                    'pen_share': pen_share,
                    'fk_share': fk_share,
                    'corner_share': corner_share,
                    'source': 'inferred_ewma',
                    'confidence': max(pen_share, fk_share, corner_share),
                    'matches_analyzed': len(group),
                    'last_updated': pd.Timestamp.now().isoformat()
                })
        
        # Use proxy methods if match events available
        if match_events is not None:
            proxy_roles = self.proxy.infer_from_match_events(match_events)
            if not proxy_roles.empty:
                # Convert proxy format to standard format
                proxy_standard = proxy_roles.rename(columns={
                    'penalty_share': 'pen_share',
                    'freekick_share': 'fk_share'
                })
                inferred_roles.extend(proxy_standard.to_dict('records'))
        
        result_df = pd.DataFrame(inferred_roles)
        
        if not result_df.empty:
            logger.info(f"Inferred set piece roles for {len(result_df)} players")
        
        return result_df
    
    def merge_all_sources(
        self,
        override_data: pd.DataFrame,
        community_data: pd.DataFrame,
        inferred_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge all set piece data sources with proper priority.
        
        Args:
            override_data: Manual overrides (highest priority)
            community_data: Community data (medium priority)
            inferred_data: Inferred data (lowest priority)
            
        Returns:
            Final merged set piece roles
        """
        # Standardize column names
        standard_columns = ['player_name', 'team', 'pen_share', 'fk_share', 'corner_share']
        
        # Prepare each source
        sources = []
        
        if not override_data.empty:
            override_clean = override_data.copy()
            override_clean['source'] = 'override'
            override_clean['priority'] = 1
            sources.append(override_clean)
        
        if not community_data.empty:
            community_clean = community_data.copy()
            community_clean['source'] = 'community'
            community_clean['priority'] = 2
            sources.append(community_clean)
        
        if not inferred_data.empty:
            inferred_clean = inferred_data.copy()
            inferred_clean['source'] = 'inferred'
            inferred_clean['priority'] = 3
            sources.append(inferred_clean)
        
        if not sources:
            logger.warning("No set piece data sources available")
            return pd.DataFrame()
        
        # Combine all sources
        all_data = pd.concat(sources, ignore_index=True)
        
        # Fill missing values
        for col in ['pen_share', 'fk_share', 'corner_share']:
            if col in all_data.columns:
                all_data[col] = pd.to_numeric(all_data[col], errors='coerce').fillna(0.0)
        
        # Merge with priority-based selection
        final_roles = []
        
        for (player_name, team), group in all_data.groupby(['player_name', 'team']):
            # Sort by priority (lowest number = highest priority)
            group = group.sort_values('priority')
            
            # Take the highest priority row for base data
            primary_row = group.iloc[0]
            
            # But allow higher priority sources to override specific roles
            final_role = {
                'player_name': player_name,
                'team': team,
                'pen_share': 0.0,
                'fk_share': 0.0,
                'corner_share': 0.0,
                'source': [],
                'confidence': 0.0,
                'last_updated': pd.Timestamp.now().isoformat()
            }
            
            # Merge each role type, taking highest priority non-zero value
            for role in ['pen_share', 'fk_share', 'corner_share']:
                for _, row in group.iterrows():
                    value = row.get(role, 0.0)
                    if value > 0 and final_role[role] == 0.0:
                        final_role[role] = value
                        if row['source'] not in final_role['source']:
                            final_role['source'].append(row['source'])
                        break
            
            # Calculate overall confidence
            final_role['confidence'] = max(
                final_role['pen_share'],
                final_role['fk_share'], 
                final_role['corner_share']
            )
            
            # Convert source list to string
            final_role['source'] = ','.join(final_role['source'])
            
            # Only include players with significant roles
            if final_role['confidence'] >= 0.1:
                final_roles.append(final_role)
        
        result_df = pd.DataFrame(final_roles)
        
        if not result_df.empty:
            # Apply Bayesian smoothing for small sample sizes
            result_df = self._apply_bayesian_smoothing(result_df)
            
            logger.info(f"Merged set piece roles for {len(result_df)} players")
        
        return result_df
    
    def _apply_bayesian_smoothing(self, roles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Bayesian smoothing to set piece roles.
        
        Args:
            roles_df: DataFrame with set piece roles
            
        Returns:
            DataFrame with smoothed roles
        """
        # Get smoothing parameters
        prior_m = self.config.get("roles.prior_m", 6)  # Prior pseudo-matches
        
        # Global priors (based on typical set piece distribution)
        priors = {
            'pen_share': 0.1,    # 10% of players take penalties
            'fk_share': 0.15,    # 15% take free kicks
            'corner_share': 0.2  # 20% take corners
        }
        
        smoothed_df = roles_df.copy()
        
        for role in ['pen_share', 'fk_share', 'corner_share']:
            if role in smoothed_df.columns:
                # Apply Bayesian smoothing: (observed * n + prior * m) / (n + m)
                # Assume n = confidence * 10 (higher confidence = more observations)
                n = smoothed_df['confidence'] * 10
                prior = priors[role]
                
                smoothed_df[role] = (
                    (smoothed_df[role] * n + prior * prior_m) / (n + prior_m)
                )
        
        return smoothed_df
    
    def save_final_roles(self, roles_df: pd.DataFrame) -> bool:
        """
        Save final merged roles to file.
        
        Args:
            roles_df: Final set piece roles
            
        Returns:
            True if saved successfully
        """
        try:
            # Ensure output directory exists
            self.roles_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add header comment
            header_lines = [
                "# Final merged set piece roles (computed)",
                "# This file is auto-generated by combining overrides > community > inferred roles",
                "# Sources: override = manual overrides, community = community data, inferred = ML inference",
                ""
            ]
            
            with open(self.roles_output_path, 'w') as f:
                for line in header_lines:
                    f.write(f"{line}\n")
            
            # Append CSV data
            roles_df.to_csv(self.roles_output_path, mode='a', index=False)
            
            logger.info(f"Saved {len(roles_df)} set piece roles to {self.roles_output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save set piece roles: {e}")
            return False
    
    def get_final_roles(
        self,
        player_data: Optional[pd.DataFrame] = None,
        match_events: Optional[pd.DataFrame] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get final set piece roles, using cache if available.
        
        Args:
            player_data: Player data for inference
            match_events: Match event data
            force_refresh: Force refresh from sources
            
        Returns:
            Final set piece roles DataFrame
        """
        cache_key = "final_setpiece_roles"
        
        # Check cache first (unless forcing refresh)
        if not force_refresh:
            cached_roles = self.cache.get(cache_key, "setpieces", ttl=86400)  # 24h cache
            if cached_roles is not None:
                logger.info("Using cached set piece roles")
                return cached_roles
        
        # Load all sources
        override_data = self.load_override_data()
        community_data = self.load_community_data()
        
        # Infer roles if player data available
        if player_data is not None:
            inferred_data = self.infer_setpiece_roles(player_data, match_events)
        else:
            inferred_data = pd.DataFrame()
        
        # Merge all sources
        final_roles = self.merge_all_sources(override_data, community_data, inferred_data)
        
        # Save to file
        if not final_roles.empty:
            self.save_final_roles(final_roles)
        
        # Cache result
        self.cache.set(cache_key, final_roles, "setpieces")
        
        return final_roles
    
    def get_player_setpiece_info(self, player_name: str, team: str) -> Optional[Dict]:
        """
        Get set piece information for a specific player.
        
        Args:
            player_name: Player name
            team: Team name
            
        Returns:
            Dictionary with player's set piece roles
        """
        final_roles = self.get_final_roles()
        
        if final_roles.empty:
            return None
        
        # Try exact match first
        exact_match = final_roles[
            (final_roles['player_name'] == player_name) & 
            (final_roles['team'] == team)
        ]
        
        if not exact_match.empty:
            return exact_match.iloc[0].to_dict()
        
        # Try name-only match
        name_match = final_roles[final_roles['player_name'] == player_name]
        
        if not name_match.empty:
            return name_match.iloc[0].to_dict()
        
        return None
