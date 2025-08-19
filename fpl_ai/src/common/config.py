"""
Configuration management for FPL AI system.

Handles loading of settings from YAML files and environment variables,
with proper validation and type conversion.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for FPL AI system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. Defaults to settings.yaml in project root.
        """
        if config_path is None:
            # Find project root (contains pyproject.toml)
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / "pyproject.toml").exists():
                    self.project_root = current_dir
                    break
                current_dir = current_dir.parent
            else:
                raise RuntimeError("Could not find project root (pyproject.toml not found)")
            
            config_path = self.project_root / "settings.yaml"
        else:
            config_path = Path(config_path)
            self.project_root = config_path.parent
            
        self.config_path = config_path
        self._config = self._load_config()
        self._setup_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _setup_paths(self):
        """Setup and create necessary directories."""
        base_path = self.project_root / "fpl_ai"
        
        # Core directories
        self.cache_dir = base_path / self.get("io.cache_dir", "cache")
        self.models_dir = base_path / self.get("io.models_dir", "models") 
        self.artifacts_dir = base_path / self.get("io.out_dir", "artifacts")
        self.logs_dir = base_path / self.get("io.logs_dir", "logs")
        self.data_dir = base_path / self.get("io.data_dir", "data")
        
        # Create directories if they don't exist
        for directory in [self.cache_dir, self.models_dir, self.artifacts_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'training.seasons_back')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            value = self._config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        return os.getenv(key, default)
    
    def get_fpl_credentials(self) -> Dict[str, str]:
        """Get FPL login credentials from environment."""
        email = self.get_env("FPL_EMAIL")
        password = self.get_env("FPL_PASSWORD") 
        entry_id = self.get_env("FPL_ENTRY_ID")
        
        if not all([email, password, entry_id]):
            raise ValueError("FPL credentials not found in environment variables")
            
        return {
            "email": email,
            "password": password,
            "entry_id": entry_id
        }
    
    def get_fbr_api_key(self) -> str:
        """Get FBRef API key from environment."""
        api_key = self.get_env("FBR_API_KEY")
        if not api_key:
            raise ValueError("FBR_API_KEY not found in environment variables")
        return api_key
    
    def get_training_mode(self, current_gw: int) -> str:
        """
        Determine training mode based on current gameweek.
        
        Args:
            current_gw: Current gameweek
            
        Returns:
            Training mode: 'warm' or 'full'
        """
        staging_mode = self.get("training.staging.mode", "auto")
        
        if staging_mode == "auto":
            warm_until_gw = self.get("training.staging.warm_until_gw", 8)
            return "warm" if current_gw < warm_until_gw else "full"
        elif staging_mode in ["warm", "full"]:
            return staging_mode
        else:
            # Default to auto mode
            warm_until_gw = self.get("training.staging.warm_until_gw", 8)
            return "warm" if current_gw < warm_until_gw else "full"
    
    def get_positions(self) -> list[str]:
        """Get list of positions for modeling."""
        return self.get("modeling.per_position.positions", ["GK", "DEF", "MID", "FWD"])
    
    def get_rolling_windows(self) -> list[int]:
        """Get rolling window sizes for features."""
        return self.get("training.rolling_windows", [3, 5, 8])
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.get(f"feeds.{feature}", False)
    
    @property
    def project_path(self) -> Path:
        """Get project root path."""
        return self.project_root


# Global configuration instance
_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger
    """
    # Import here to avoid circular imports
    from .logging_setup import setup_logging
    
    # Setup logging on first call
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)


def save_settings_dict(cfg: dict, path: Path):
    """
    Save settings dictionary to YAML file.
    
    Args:
        cfg: Configuration dictionary to save
        path: Path to save the YAML file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def load_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load settings from YAML file.
    
    Args:
        config_path: Optional path to config file. If None, uses default settings.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default settings.yaml from project root
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:
            if (current_dir / "pyproject.toml").exists():
                config_path = str(current_dir / "settings.yaml")
                break
            current_dir = current_dir.parent
        else:
            raise RuntimeError("Could not find project root (pyproject.toml not found)")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
