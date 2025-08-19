"""
Smoke tests for FPL AI - Basic import and sanity checks.

These tests verify that all modules can be imported successfully
and basic functionality works without errors.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


def test_common_imports():
    """Test that common utilities can be imported."""
    try:
        from src.common import config, cache, logging_setup, timeutil, metrics
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import common modules: {e}")


def test_providers_imports():
    """Test that data providers can be imported."""
    try:
        from src.providers import (
            fpl_api, fpl_map, fpl_picks, fbrapi_client,
            injuries, fixtures, odds_input, setpieces_proxy,
            setpiece_roles
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import provider modules: {e}")


def test_features_imports():
    """Test that feature modules can be imported."""
    try:
        from src.features import builder, touches, team_form, h2h
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import feature modules: {e}")


def test_modeling_imports():
    """Test that modeling modules can be imported."""
    try:
        from src.modeling import (
            minutes_model, model_lgbm, calibration, mc_sim
        )
        assert True
    except ImportError as e:
        if 'lightgbm' in str(e) or 'sklearn' in str(e):
            pytest.skip(f"Skipping due to missing ML dependency: {e}")
        else:
            pytest.fail(f"Failed to import modeling modules: {e}")


def test_optimize_imports():
    """Test that optimization modules can be imported."""
    try:
        from src.optimize import formations, optimizer, chips_forward
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import optimization modules: {e}")


def test_plan_imports():
    """Test that planning modules can be imported."""
    try:
        from src.plan import utils_prices, multiweek_planner
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import planning modules: {e}")


def test_cli_imports():
    """Test that CLI modules can be imported."""
    try:
        from src.cli import pipeline, audit, preview, plan10
        assert True
    except ImportError as e:
        if 'lightgbm' in str(e) or 'sklearn' in str(e):
            pytest.skip(f"Skipping due to missing ML dependency: {e}")
        else:
            pytest.fail(f"Failed to import CLI modules: {e}")


def test_config_loading():
    """Test that configuration can be loaded."""
    try:
        from src.common.config import get_config
        
        # Test get_config works
        config = get_config()
        assert hasattr(config, 'get')
        
        # Test basic config access
        result = config.get('training.seasons_back', 4)
        assert isinstance(result, int)
        
    except Exception as e:
        pytest.fail(f"Failed to load configuration: {e}")


def test_logging_setup():
    """Test that logging can be set up."""
    try:
        from src.common.logging_setup import setup_logging
        
        # Should not raise exceptions
        setup_logging()
        setup_logging(level="DEBUG")
        setup_logging(level="INFO")
        
    except Exception as e:
        pytest.fail(f"Failed to setup logging: {e}")


def test_cache_functionality():
    """Test basic cache functionality."""
    try:
        from src.common.cache import get_cache, cache_get, cache_set
        
        # Basic test - should not crash
        cache = get_cache()
        assert hasattr(cache, 'get')
        
        # Test cache operations
        cache_set("test_key", "test_value")
        result = cache_get("test_key")
        assert result == "test_value"
        
    except Exception as e:
        pytest.fail(f"Failed cache functionality test: {e}")


def test_formation_validation():
    """Test formation validation functionality."""
    try:
        from src.optimize.formations import FormationValidator
        
        validator = FormationValidator()
        
        # Test that we can get valid formations
        formations = validator.get_valid_formations()
        assert len(formations) > 0
        assert (3, 4, 3) in formations
        
        # Test position requirements
        requirements = validator.get_position_requirements((3, 4, 3))
        assert requirements['DEF'] == 3
        assert requirements['MID'] == 4
        assert requirements['FWD'] == 3
        
    except Exception as e:
        pytest.fail(f"Failed formation validation test: {e}")


def test_fpl_pricing():
    """Test FPL pricing utilities."""
    try:
        from src.plan.utils_prices import fpl_sell_value
        
        # Test basic FPL selling rule
        # Player bought at 5.0, now worth 6.0, should sell for 5.5
        selling_price = fpl_sell_value(50, 60)  # In tenths
        assert selling_price == 55
        
        # Player bought at 5.0, now worth 4.0, should sell for 5.0
        selling_price = fpl_sell_value(50, 40)
        assert selling_price == 50
        
    except Exception as e:
        pytest.fail(f"Failed FPL pricing test: {e}")


def test_basic_dataframe_operations():
    """Test that pandas operations work as expected."""
    try:
        import pandas as pd
        import numpy as np
        
        # Create test dataframe
        df = pd.DataFrame({
            'element_id': [1, 2, 3],
            'position': ['GK', 'DEF', 'MID'],
            'proj_points': [4.0, 5.5, 6.2],
            'now_cost': [45, 55, 70]
        })
        
        # Basic operations
        assert len(df) == 3
        assert df['proj_points'].max() == 6.2
        assert 'GK' in df['position'].values
        
    except Exception as e:
        pytest.fail(f"Failed basic dataframe operations: {e}")


def test_file_structure():
    """Test that key files exist in the project structure."""
    # Files are in the parent directory of fpl_ai/
    project_root = Path(__file__).parent.parent.parent
    
    # Key files that should exist
    required_files = [
        "settings.yaml",
        "pyproject.toml",
        "Makefile",
        ".env.example",
        ".gitignore",
        "README.md"
    ]
    
    for file in required_files:
        file_path = project_root / file
        assert file_path.exists(), f"Required file missing: {file} (looking in {project_root})"


def test_directory_structure():
    """Test that key directories exist."""
    project_root = Path(__file__).parent.parent
    
    # Key directories that should exist
    required_dirs = [
        "src",
        "src/common",
        "src/providers", 
        "src/features",
        "src/modeling",
        "src/optimize",
        "src/plan",
        "src/cli",
        "app",
        "app/pages",
        "data",
        "cache",
        "models",
        "artifacts",
        "tests"
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Required directory missing: {dir_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
