"""
FPL player mapping utilities.

Provides functionality for mapping FPL players to external data sources
like FBRef, handling name variations and team changes.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from rapidfuzz import fuzz, process
from ..common.config import get_config, get_logger
from ..common.cache import get_cache

logger = get_logger(__name__)


class FPLMapper:
    """
    Mapper for linking FPL players to external data sources.
    """
    
    def __init__(self):
        """Initialize FPL mapper."""
        self.config = get_config()
        self.cache = get_cache()
        
        # Similarity thresholds
        self.exact_match_threshold = 95
        self.fuzzy_match_threshold = 80
        self.min_match_threshold = 70
        
        logger.info("FPL mapper initialized")
    
    def load_manual_mappings(self) -> pd.DataFrame:
        """
        Load manually defined player mappings.
        
        Returns:
            DataFrame with manual mappings
        """
        mapping_file = self.config.data_dir / "fbr_player_map.csv"
        
        try:
            if mapping_file.exists():
                df = pd.read_csv(mapping_file, comment='#')
                # Filter out empty/comment rows
                df = df.dropna(subset=['fpl_player_id'])
                df = df[~df['fpl_player_id'].astype(str).str.startswith('#')]
                logger.info(f"Loaded {len(df)} manual player mappings")
                return df
            else:
                logger.warning(f"Manual mapping file not found: {mapping_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load manual mappings: {e}")
            return pd.DataFrame()
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize player name for matching.
        
        Args:
            name: Raw player name
            
        Returns:
            Normalized name
        """
        if not isinstance(name, str):
            return ""
        
        # Basic normalization
        name = name.strip().lower()
        
        # Remove common prefixes/suffixes
        prefixes = ['mr.', 'dr.', 'sir']
        suffixes = ['jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv']
        
        for prefix in prefixes:
            if name.startswith(prefix + ' '):
                name = name[len(prefix):].strip()
        
        for suffix in suffixes:
            if name.endswith(' ' + suffix):
                name = name[:-len(suffix)].strip()
        
        # Handle common name variations
        replacements = {
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n',
            '-': ' ', '_': ' ',
            "'": '', "'": '', '`': ''
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        # Normalize whitespace
        name = ' '.join(name.split())
        
        return name
    
    def extract_name_variants(self, name: str) -> List[str]:
        """
        Generate name variants for better matching.
        
        Args:
            name: Player name
            
        Returns:
            List of name variants
        """
        variants = [name]
        
        # Split into parts
        parts = name.split()
        
        if len(parts) >= 2:
            # First Last
            variants.append(f"{parts[0]} {parts[-1]}")
            
            # Last, First
            variants.append(f"{parts[-1]}, {parts[0]}")
            
            # First initial + Last
            variants.append(f"{parts[0][0]}. {parts[-1]}")
            
            # Full First + Last initial
            variants.append(f"{parts[0]} {parts[-1][0]}.")
            
            # Just last name
            variants.append(parts[-1])
            
            # Middle names handling
            if len(parts) > 2:
                # First Middle Last -> First Last
                variants.append(f"{parts[0]} {parts[-1]}")
                
                # Include middle initials
                middle_initials = ''.join([p[0] + '.' for p in parts[1:-1]])
                variants.append(f"{parts[0]} {middle_initials} {parts[-1]}")
        
        # Remove duplicates while preserving order
        unique_variants = []
        for variant in variants:
            if variant not in unique_variants:
                unique_variants.append(variant)
        
        return unique_variants
    
    def find_best_match(
        self,
        target_name: str,
        candidate_names: List[str],
        candidate_data: Optional[List[Dict]] = None
    ) -> Optional[Tuple[str, float, Optional[Dict]]]:
        """
        Find best matching name from candidates.
        
        Args:
            target_name: Name to match
            candidate_names: List of candidate names
            candidate_data: Optional list of candidate data dicts
            
        Returns:
            Tuple of (best_match_name, similarity_score, candidate_data) or None
        """
        if not candidate_names:
            return None
        
        # Normalize target name
        target_normalized = self.normalize_name(target_name)
        target_variants = self.extract_name_variants(target_normalized)
        
        best_match = None
        best_score = 0
        best_data = None
        
        # Normalize all candidate names
        candidates_normalized = [self.normalize_name(name) for name in candidate_names]
        
        # Try each target variant
        for variant in target_variants:
            # Use rapidfuzz for efficient fuzzy matching
            match_result = process.extractOne(
                variant,
                candidates_normalized,
                scorer=fuzz.ratio
            )
            
            if match_result and match_result[1] > best_score:
                best_score = match_result[1]
                best_match_idx = candidates_normalized.index(match_result[0])
                best_match = candidate_names[best_match_idx]
                
                if candidate_data and best_match_idx < len(candidate_data):
                    best_data = candidate_data[best_match_idx]
        
        # Return result only if above minimum threshold
        if best_score >= self.min_match_threshold:
            return best_match, best_score, best_data
        
        return None
    
    def map_fpl_to_fbr(
        self,
        fpl_players: pd.DataFrame,
        fbr_players: pd.DataFrame,
        manual_mappings: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Map FPL players to FBR players.
        
        Args:
            fpl_players: FPL player data
            fbr_players: FBR player data
            manual_mappings: Manual mappings to apply first
            
        Returns:
            DataFrame with mapping results
        """
        mappings = []
        
        # Apply manual mappings first
        manual_mapped_ids = set()
        if manual_mappings is not None and not manual_mappings.empty:
            for _, mapping in manual_mappings.iterrows():
                fpl_id = mapping.get('fpl_player_id')
                fbr_id = mapping.get('fbr_player_id')
                player_name = mapping.get('player_name', '')
                
                if pd.notna(fpl_id) and pd.notna(fbr_id):
                    mappings.append({
                        'fpl_player_id': int(fpl_id),
                        'fbr_player_id': int(fbr_id) if pd.notna(fbr_id) else None,
                        'fpl_name': player_name,
                        'fbr_name': player_name,
                        'match_type': 'manual',
                        'similarity_score': 100.0,
                        'confidence': 'high'
                    })
                    manual_mapped_ids.add(int(fpl_id))
        
        # Prepare FBR candidate data
        fbr_candidates = []
        fbr_names = []
        
        for _, player in fbr_players.iterrows():
            name = player.get('name', player.get('player_name', ''))
            fbr_names.append(name)
            fbr_candidates.append({
                'id': player.get('id', player.get('player_id')),
                'name': name,
                'team': player.get('team', ''),
                'league': player.get('league_id', '')
            })
        
        # Map remaining FPL players
        for _, fpl_player in fpl_players.iterrows():
            fpl_id = fpl_player['id']
            
            # Skip if manually mapped
            if fpl_id in manual_mapped_ids:
                continue
            
            fpl_name = fpl_player.get('web_name', fpl_player.get('name', ''))
            fpl_full_name = f"{fpl_player.get('first_name', '')} {fpl_player.get('second_name', '')}".strip()
            
            # Try matching with both web_name and full name
            best_match = None
            best_score = 0
            
            for name_to_try in [fpl_name, fpl_full_name]:
                if not name_to_try:
                    continue
                
                match_result = self.find_best_match(name_to_try, fbr_names, fbr_candidates)
                
                if match_result and match_result[1] > best_score:
                    best_match = match_result
                    best_score = match_result[1]
            
            if best_match:
                match_name, similarity, fbr_data = best_match
                
                # Determine match type and confidence
                if similarity >= self.exact_match_threshold:
                    match_type = 'exact'
                    confidence = 'high'
                elif similarity >= self.fuzzy_match_threshold:
                    match_type = 'fuzzy'
                    confidence = 'medium'
                else:
                    match_type = 'weak'
                    confidence = 'low'
                
                mappings.append({
                    'fpl_player_id': fpl_id,
                    'fbr_player_id': fbr_data['id'] if fbr_data else None,
                    'fpl_name': fpl_name,
                    'fbr_name': match_name,
                    'match_type': match_type,
                    'similarity_score': similarity,
                    'confidence': confidence
                })
        
        mapping_df = pd.DataFrame(mappings)
        
        if not mapping_df.empty:
            logger.info(f"Created {len(mapping_df)} FPL-to-FBR mappings")
            
            # Log summary by confidence
            confidence_counts = mapping_df['confidence'].value_counts()
            for confidence, count in confidence_counts.items():
                logger.info(f"  {confidence} confidence: {count} mappings")
        
        return mapping_df
    
    def validate_mappings(
        self,
        mappings: pd.DataFrame,
        fpl_players: pd.DataFrame,
        fbr_players: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Validate and enrich mapping data.
        
        Args:
            mappings: Mapping data
            fpl_players: FPL player data
            fbr_players: FBR player data
            
        Returns:
            Validated and enriched mappings
        """
        if mappings.empty:
            return mappings
        
        validated_mappings = []
        
        # Create lookup dictionaries
        fpl_lookup = fpl_players.set_index('id').to_dict('index')
        fbr_lookup = fbr_players.set_index('id').to_dict('index') if 'id' in fbr_players.columns else {}
        
        for _, mapping in mappings.iterrows():
            fpl_id = mapping['fpl_player_id']
            fbr_id = mapping.get('fbr_player_id')
            
            # Validate FPL player exists
            if fpl_id not in fpl_lookup:
                logger.warning(f"FPL player {fpl_id} not found in player data")
                continue
            
            fpl_data = fpl_lookup[fpl_id]
            
            # Enrich with FPL data
            enriched_mapping = mapping.to_dict()
            enriched_mapping.update({
                'fpl_team': fpl_data.get('team_name', ''),
                'fpl_position': fpl_data.get('position', ''),
                'fpl_cost': fpl_data.get('now_cost', 0)
            })
            
            # Validate and enrich FBR data if available
            if fbr_id and fbr_id in fbr_lookup:
                fbr_data = fbr_lookup[fbr_id]
                enriched_mapping.update({
                    'fbr_team': fbr_data.get('team', ''),
                    'fbr_league': fbr_data.get('league_id', ''),
                    'fbr_valid': True
                })
            else:
                enriched_mapping.update({
                    'fbr_team': '',
                    'fbr_league': '',
                    'fbr_valid': False
                })
            
            validated_mappings.append(enriched_mapping)
        
        validated_df = pd.DataFrame(validated_mappings)
        
        logger.info(f"Validated {len(validated_df)} mappings")
        return validated_df
    
    def save_mappings(self, mappings: pd.DataFrame, filename: str = "fbr_player_map_generated.csv") -> bool:
        """
        Save mappings to CSV file.
        
        Args:
            mappings: Mapping data
            filename: Output filename
            
        Returns:
            True if saved successfully
        """
        try:
            output_path = self.config.data_dir / filename
            
            # Add header comments
            header_lines = [
                "# Auto-generated FPL to FBR player mappings",
                "# fpl_player_id: FPL element ID",
                "# fbr_player_id: FBR player ID (null if no match)",
                "# match_type: exact, fuzzy, weak, or manual",
                "# confidence: high, medium, or low",
                ""
            ]
            
            with open(output_path, 'w') as f:
                for line in header_lines:
                    f.write(f"{line}\n")
            
            # Append CSV data
            mappings.to_csv(output_path, mode='a', index=False)
            
            logger.info(f"Saved {len(mappings)} mappings to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save mappings: {e}")
            return False
    
    def get_unmapped_players(
        self,
        fpl_players: pd.DataFrame,
        mappings: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get FPL players that don't have mappings.
        
        Args:
            fpl_players: FPL player data
            mappings: Current mappings
            
        Returns:
            DataFrame with unmapped players
        """
        if mappings.empty:
            mapped_ids = set()
        else:
            mapped_ids = set(mappings['fpl_player_id'])
        
        unmapped = fpl_players[~fpl_players['id'].isin(mapped_ids)].copy()
        
        logger.info(f"Found {len(unmapped)} unmapped FPL players")
        return unmapped
