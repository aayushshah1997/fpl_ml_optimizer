"""
Caching utilities for FPL AI system.

Provides intelligent caching for API responses, model outputs, and expensive computations
with configurable TTL, compression, and cache invalidation strategies.
"""

import os
import json
import pickle
import hashlib
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Union
import pandas as pd
from .config import get_config, get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Intelligent cache manager with TTL, compression, and type-specific handling.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Custom cache directory path
        """
        config = get_config()
        self.cache_dir = cache_dir or config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Default TTL settings
        self.default_ttl = config.get("performance.cache_ttl_hours", 24) * 3600
        self.api_cache_ttl = 1800  # 30 minutes for API responses
        self.model_cache_ttl = 86400 * 7  # 1 week for model outputs
        
        logger.info(f"Cache manager initialized: {self.cache_dir}")
    
    def _get_cache_path(self, key: str, category: str = "general") -> Path:
        """Get cache file path for a given key."""
        # Create category subdirectory
        category_dir = self.cache_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Hash key to handle long/special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return category_dir / f"{key_hash}.cache"
    
    def _get_metadata_path(self, cache_path: Path) -> Path:
        """Get metadata file path for cache entry."""
        return cache_path.with_suffix('.meta')
    
    def _is_expired(self, cache_path: Path, ttl: int) -> bool:
        """Check if cache entry is expired."""
        if not cache_path.exists():
            return True
            
        metadata_path = self._get_metadata_path(cache_path)
        if not metadata_path.exists():
            return True
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            created_at = datetime.fromisoformat(metadata['created_at'])
            # Handle timezone-aware vs timezone-naive comparison
            now = datetime.now()
            if created_at.tzinfo is not None and now.tzinfo is None:
                # created_at is timezone-aware, make now timezone-aware
                now = now.replace(tzinfo=created_at.tzinfo)
            elif created_at.tzinfo is None and now.tzinfo is not None:
                # now is timezone-aware, make created_at timezone-aware
                created_at = created_at.replace(tzinfo=now.tzinfo)
            return now - created_at > timedelta(seconds=ttl)
        except (json.JSONDecodeError, KeyError, ValueError):
            return True
    
    def _save_metadata(self, cache_path: Path, original_key: str, data_type: str):
        """Save cache metadata."""
        metadata = {
            'created_at': datetime.now().replace(tzinfo=None).isoformat(),
            'key': original_key,
            'data_type': data_type,
            'file_size': cache_path.stat().st_size if cache_path.exists() else 0
        }
        
        metadata_path = self._get_metadata_path(cache_path)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _serialize_data(self, data: Any) -> tuple[bytes, str]:
        """
        Serialize data based on type.
        
        Returns:
            Tuple of (serialized_bytes, data_type)
        """
        if isinstance(data, pd.DataFrame):
            # Clean DataFrame before serialization to avoid parquet conversion errors
            try:
                # Convert object columns to string to avoid serialization issues
                df_clean = data.copy()
                for col in df_clean.columns:
                    if df_clean[col].dtype == 'object':
                        df_clean[col] = df_clean[col].astype(str)
                
                # Use parquet for DataFrames (better compression + preserves dtypes)
                return df_clean.to_parquet(), "dataframe"
            except Exception as e:
                # Fallback to pickle if parquet fails
                logger.debug(f"Parquet serialization failed, using pickle: {e}")
                return pickle.dumps(data), "pickle"
        elif isinstance(data, dict) or isinstance(data, list):
            # Use JSON for simple structures
            return json.dumps(data, indent=2).encode(), "json"
        else:
            # Use pickle for everything else
            return pickle.dumps(data), "pickle"
    
    def _deserialize_data(self, data_bytes: bytes, data_type: str) -> Any:
        """Deserialize data based on type."""
        if data_type == "dataframe":
            return pd.read_parquet(data_bytes)
        elif data_type == "json":
            return json.loads(data_bytes.decode())
        else:
            return pickle.loads(data_bytes)
    
    def get(self, key: str, category: str = "general", ttl: Optional[int] = None) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            category: Cache category for organization
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            Cached data or None if not found/expired
        """
        cache_path = self._get_cache_path(key, category)
        ttl = ttl or self.default_ttl
        
        if self._is_expired(cache_path, ttl):
            return None
        
        try:
            # Read metadata to get data type
            metadata_path = self._get_metadata_path(cache_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            data_type = metadata.get('data_type', 'pickle')
            
            # Read and decompress data
            with gzip.open(cache_path, 'rb') as f:
                compressed_data = f.read()
            
            # Deserialize
            data = self._deserialize_data(compressed_data, data_type)
            
            logger.debug(f"Cache hit: {key} ({category})")
            return data
            
        except Exception as e:
            logger.debug(f"Failed to read cache {key}: {e}")
            return None
    
    def set(self, key: str, data: Any, category: str = "general", ttl: Optional[int] = None):
        """
        Store item in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            category: Cache category for organization
            ttl: Time to live in seconds (for metadata only)
        """
        cache_path = self._get_cache_path(key, category)
        
        try:
            # Serialize data
            serialized_data, data_type = self._serialize_data(data)
            
            # Compress and save
            with gzip.open(cache_path, 'wb') as f:
                f.write(serialized_data)
            
            # Save metadata
            self._save_metadata(cache_path, key, data_type)
            
            logger.debug(f"Cache set: {key} ({category}, {len(serialized_data)} bytes)")
            
        except Exception as e:
            logger.debug(f"Failed to cache {key}: {e}")
    
    def delete(self, key: str, category: str = "general"):
        """Delete item from cache."""
        cache_path = self._get_cache_path(key, category)
        metadata_path = self._get_metadata_path(cache_path)
        
        for path in [cache_path, metadata_path]:
            if path.exists():
                path.unlink()
        
        logger.debug(f"Cache deleted: {key} ({category})")
    
    def clear_category(self, category: str):
        """Clear all items in a category."""
        category_dir = self.cache_dir / category
        if category_dir.exists():
            for file_path in category_dir.iterdir():
                file_path.unlink()
        logger.info(f"Cleared cache category: {category}")
    
    def clear_expired(self, category: Optional[str] = None):
        """Clear all expired cache entries."""
        dirs_to_check = [self.cache_dir / category] if category else [
            d for d in self.cache_dir.iterdir() if d.is_dir()
        ]
        
        cleared_count = 0
        for cache_dir in dirs_to_check:
            if not cache_dir.exists():
                continue
                
            for cache_file in cache_dir.glob("*.cache"):
                metadata_file = cache_file.with_suffix('.meta')
                
                if not metadata_file.exists():
                    # Remove orphaned cache files
                    cache_file.unlink()
                    cleared_count += 1
                    continue
                
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    created_at = datetime.fromisoformat(metadata['created_at'])
                    # Handle timezone-aware vs timezone-naive comparison
                    now = datetime.now()
                    if created_at.tzinfo is not None and now.tzinfo is None:
                        # created_at is timezone-aware, make now timezone-aware
                        now = now.replace(tzinfo=created_at.tzinfo)
                    elif created_at.tzinfo is None and now.tzinfo is not None:
                        # now is timezone-aware, make created_at timezone-aware
                        created_at = created_at.replace(tzinfo=now.tzinfo)
                    if now - created_at > timedelta(seconds=self.default_ttl):
                        cache_file.unlink()
                        metadata_file.unlink()
                        cleared_count += 1
                        
                except Exception:
                    # Remove corrupted entries
                    cache_file.unlink()
                    if metadata_file.exists():
                        metadata_file.unlink()
                    cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'categories': {},
            'total_files': 0,
            'total_size_mb': 0
        }
        
        for category_dir in self.cache_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            cache_files = list(category_dir.glob("*.cache"))
            
            total_size = sum(f.stat().st_size for f in cache_files)
            
            stats['categories'][category_name] = {
                'files': len(cache_files),
                'size_mb': total_size / (1024 * 1024)
            }
            
            stats['total_files'] += len(cache_files)
            stats['total_size_mb'] += total_size / (1024 * 1024)
        
        return stats


# Global cache manager instance
_cache_manager = None


def get_cache() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# Convenience functions
def cache_get(key: str, category: str = "general", ttl: Optional[int] = None) -> Optional[Any]:
    """Get item from cache."""
    return get_cache().get(key, category, ttl)


def cache_set(key: str, data: Any, category: str = "general", ttl: Optional[int] = None):
    """Store item in cache."""
    get_cache().set(key, data, category, ttl)


def cache_delete(key: str, category: str = "general"):
    """Delete item from cache."""
    get_cache().delete(key, category)


# Compatibility class for existing code that expects 'Cache' class
class Cache:
    """
    Compatibility wrapper around CacheManager for existing code.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache compatibility wrapper.

        Args:
            cache_dir: Optional cache directory path
        """
        self.cache_manager = CacheManager(cache_dir)

    def get(self, key: str, category: str = "general", ttl: Optional[int] = None) -> Optional[Any]:
        """Get item from cache."""
        return self.cache_manager.get(key, category, ttl)

    def set(self, key: str, data: Any, category: str = "general", ttl: Optional[int] = None):
        """Store item in cache."""
        self.cache_manager.set(key, data, category, ttl)

    def delete(self, key: str, category: str = "general"):
        """Delete item from cache."""
        self.cache_manager.delete(key, category)

    def clear(self, category: Optional[str] = None):
        """Clear cache entries."""
        if category:
            self.cache_manager.clear_category(category)
        else:
            self.cache_manager.clear_all()
