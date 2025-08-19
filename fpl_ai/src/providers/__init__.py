"""
Data providers for FPL AI system.

This module contains all data providers for fetching information from various sources:
- FPL API (official fantasy football data)
- FBR API (comprehensive football statistics)
- Injury data providers
- Fixture and odds data
- Set piece and role data
"""

from .fpl_api import FPLAPIClient
from .fbrapi_client import FBRAPIClient
from .fpl_picks import FPLPicksClient
from .fixtures import FixturesProvider
from .injuries import InjuryProvider
from .setpieces_proxy import SetPieceProxy
from .setpiece_roles import SetPieceRolesManager

__all__ = [
    "FPLAPIClient",
    "FBRAPIClient", 
    "FPLPicksClient",
    "FixturesProvider",
    "InjuryProvider",
    "SetPieceProxy",
    "SetPieceRolesManager",
]
