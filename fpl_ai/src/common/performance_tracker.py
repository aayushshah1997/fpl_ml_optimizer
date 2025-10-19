"""
Performance tracking system for comparing predicted vs actual points.

This module tracks the performance of model predictions by comparing
predicted points against actual points scored by recommended players.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from .config import get_config, get_logger

logger = get_logger(__name__)


class ModelPerformanceTracker:
    """
    Tracks model performance by comparing predicted vs actual points
    for recommended teams and individual players.
    """
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.performance_dir = self.artifacts_dir / "performance"
        # Ensure parent directories exist (fixes FileNotFoundError when 'artifacts' is missing)
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking files
        self.team_performance_file = self.performance_dir / "team_performance.csv"
        self.player_performance_file = self.performance_dir / "player_performance.csv"
        self.summary_file = self.performance_dir / "performance_summary.json"
        
        logger.info(f"Performance tracker initialized - artifacts: {self.artifacts_dir}")
    
    def save_predicted_team(
        self, 
        gameweek: int, 
        squad: List[Dict], 
        starting_xi: List[Dict], 
        captain: Dict, 
        vice_captain: Dict,
        predictions_df: pd.DataFrame,
        formation: Tuple[int, int, int] = (0, 0, 0)
    ) -> None:
        """
        Save the predicted team for a gameweek for later performance comparison.
        
        Args:
            gameweek: Gameweek number
            squad: 15-man squad
            starting_xi: Starting XI players
            captain: Captain player info
            vice_captain: Vice-captain player info
            predictions_df: Full predictions DataFrame
            formation: Formation tuple (DEF, MID, FWD)
        """
        try:
            # Create prediction record
            prediction_record = {
                'gameweek': gameweek,
                'timestamp': datetime.now().isoformat(),
                'formation': f"{formation[0]}-{formation[1]}-{formation[2]}",
                'squad': self._extract_player_predictions(squad, predictions_df),
                'starting_xi': self._extract_player_predictions(starting_xi, predictions_df),
                'captain': self._extract_player_prediction(captain, predictions_df),
                'vice_captain': self._extract_player_prediction(vice_captain, predictions_df),
                'predicted_squad_points': sum(p.get('predicted_points', 0) for p in self._extract_player_predictions(squad, predictions_df)),
                'predicted_xi_points': sum(p.get('predicted_points', 0) for p in self._extract_player_predictions(starting_xi, predictions_df)),
                'predicted_captain_points': self._extract_player_prediction(captain, predictions_df).get('predicted_points', 0) * 2
            }
            
            # Save to file
            prediction_file = self.performance_dir / f"predicted_team_gw{gameweek}.json"
            with open(prediction_file, 'w') as f:
                json.dump(prediction_record, f, indent=2)
            
            logger.info(f"Saved predicted team for GW{gameweek} - Squad: {len(squad)}, XI: {len(starting_xi)}")
            
        except Exception as e:
            logger.error(f"Error saving predicted team for GW{gameweek}: {e}")
    
    def _extract_player_predictions(self, players: List[Dict], predictions_df: pd.DataFrame) -> List[Dict]:
        """Extract prediction data for a list of players."""
        result = []
        for player in players:
            result.append(self._extract_player_prediction(player, predictions_df))
        return result
    
    def _extract_player_prediction(self, player: Dict, predictions_df: pd.DataFrame) -> Dict:
        """Extract prediction data for a single player."""
        element_id = player.get('element_id')
        
        # Get prediction data
        pred_data = predictions_df[predictions_df['element_id'] == element_id].iloc[0] if not predictions_df[predictions_df['element_id'] == element_id].empty else {}
        
        return {
            'element_id': element_id,
            'web_name': player.get('web_name', pred_data.get('web_name', 'Unknown')),
            'position': player.get('position', pred_data.get('position', 'Unknown')),
            'team_name': player.get('team_name', pred_data.get('team_name', 'Unknown')),
            'predicted_points': player.get('proj_points', player.get('mean_points', pred_data.get('proj_points', 0))),
            'cost': player.get('now_cost', pred_data.get('now_cost', 0))
        }
    
    def update_actual_results(self, gameweek: int, vaastav_data_dir: str = None) -> None:
        """
        Update actual results for a completed gameweek.
        
        Args:
            gameweek: Completed gameweek number
            vaastav_data_dir: Path to Vaastav data directory
        """
        try:
            # Load predicted team for this gameweek
            prediction_file = self.performance_dir / f"predicted_team_gw{gameweek}.json"
            if not prediction_file.exists():
                logger.warning(f"No prediction file found for GW{gameweek}")
                return
            
            with open(prediction_file, 'r') as f:
                prediction_data = json.load(f)
            
            # Use default vaastav data directory if not provided
            if vaastav_data_dir is None:
                # Use current season dynamically
                from .timeutil import get_current_season
                current_season = get_current_season()
                vaastav_data_dir = self.artifacts_dir.parent / "data" / "vaastav" / "data" / current_season
            
            # Load actual results
            actual_results = self._load_actual_results(gameweek, str(vaastav_data_dir))
            if actual_results.empty:
                logger.warning(f"No actual results found for GW{gameweek}")
                return
            
            # Calculate performance metrics
            performance = self._calculate_performance(prediction_data, actual_results)
            
            # Save performance data
            self._save_performance_data(gameweek, performance)
            
            # Update summary statistics
            self._update_summary_stats()
            
            logger.info(f"Updated performance tracking for GW{gameweek}")
            
        except Exception as e:
            logger.error(f"Error updating actual results for GW{gameweek}: {e}")
    
    def _load_actual_results(self, gameweek: int, vaastav_data_dir: str) -> pd.DataFrame:
        """Load actual gameweek results from Vaastav data."""
        try:
            gw_file = Path(vaastav_data_dir) / "gws" / f"gw{gameweek}.csv"
            if not gw_file.exists():
                logger.warning(f"Actual results file not found: {gw_file}")
                return pd.DataFrame()
            
            df = pd.read_csv(gw_file)
            
            # Ensure we have the required columns
            required_cols = ['element', 'total_points', 'name']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in {gw_file}: {required_cols}")
                return pd.DataFrame()
            
            # Rename for consistency
            df = df.rename(columns={'element': 'element_id', 'total_points': 'actual_points'})
            
            logger.info(f"Loaded actual results for GW{gameweek}: {len(df)} players")
            return df[['element_id', 'actual_points', 'name', 'position', 'team']]
            
        except Exception as e:
            logger.error(f"Error loading actual results for GW{gameweek}: {e}")
            return pd.DataFrame()
    
    def _calculate_performance(self, prediction_data: Dict, actual_results: pd.DataFrame) -> Dict:
        """Calculate performance metrics comparing predicted vs actual."""
        performance = {
            'gameweek': prediction_data['gameweek'],
            'timestamp': datetime.now().isoformat(),
            'formation': prediction_data['formation'],
            'squad_performance': self._calculate_team_performance(prediction_data['squad'], actual_results),
            'xi_performance': self._calculate_team_performance(prediction_data['starting_xi'], actual_results),
            'captain_performance': self._calculate_player_performance(prediction_data['captain'], actual_results, is_captain=True),
            'vice_captain_performance': self._calculate_player_performance(prediction_data['vice_captain'], actual_results),
        }
        
        # Calculate overall metrics
        squad_perf = performance['squad_performance']
        xi_perf = performance['xi_performance']
        
        performance.update({
            'squad_accuracy': {
                'mae': squad_perf['mae'],
                'rmse': squad_perf['rmse'],
                'total_predicted': squad_perf['total_predicted'],
                'total_actual': squad_perf['total_actual'],
                'difference': squad_perf['total_actual'] - squad_perf['total_predicted']
            },
            'xi_accuracy': {
                'mae': xi_perf['mae'],
                'rmse': xi_perf['rmse'], 
                'total_predicted': xi_perf['total_predicted'],
                'total_actual': xi_perf['total_actual'],
                'difference': xi_perf['total_actual'] - xi_perf['total_predicted']
            }
        })
        
        return performance
    
    def _calculate_team_performance(self, predicted_team: List[Dict], actual_results: pd.DataFrame) -> Dict:
        """Calculate performance metrics for a team (squad or XI)."""
        total_predicted = 0
        total_actual = 0
        errors = []
        player_performances = []
        
        for player in predicted_team:
            element_id = player['element_id']
            predicted_points = player['predicted_points']
            
            # Find actual points - try by name first, fallback to element_id
            web_name = player['web_name']
            actual_row = actual_results[actual_results['name'] == web_name]
            if actual_row.empty:
                # Try partial name match for players with special characters
                actual_row = actual_results[actual_results['name'].str.contains(web_name, case=False, na=False)]
            if actual_row.empty:
                # Fallback to element_id match
                actual_row = actual_results[actual_results['element_id'] == element_id]
            
            if actual_row.empty:
                logger.warning(f"No actual results found for player {element_id} ({web_name})")
                actual_points = 0
            else:
                actual_points = actual_row.iloc[0]['actual_points']
                if len(actual_row) == 1:
                    logger.debug(f"Matched {web_name} -> {actual_points} points")
            
            # Calculate metrics
            error = actual_points - predicted_points
            total_predicted += predicted_points
            total_actual += actual_points
            errors.append(error)
            
            player_performances.append({
                'element_id': element_id,
                'web_name': player['web_name'],
                'position': player['position'],
                'predicted_points': predicted_points,
                'actual_points': actual_points,
                'error': error,
                'absolute_error': abs(error)
            })
        
        # Calculate team-level metrics
        mae = sum(abs(e) for e in errors) / len(errors) if errors else 0
        rmse = (sum(e**2 for e in errors) / len(errors))**0.5 if errors else 0
        
        return {
            'total_predicted': total_predicted,
            'total_actual': total_actual,
            'total_error': total_actual - total_predicted,
            'mae': mae,
            'rmse': rmse,
            'player_count': len(predicted_team),
            'players': player_performances
        }
    
    def _calculate_player_performance(self, player: Dict, actual_results: pd.DataFrame, is_captain: bool = False) -> Dict:
        """Calculate performance for individual player (captain/vice-captain)."""
        element_id = player['element_id']
        predicted_points = player['predicted_points']
        
        # Find actual points - try by name first, fallback to element_id
        web_name = player['web_name']
        actual_row = actual_results[actual_results['name'] == web_name]
        if actual_row.empty:
            # Try partial name match for players with special characters
            actual_row = actual_results[actual_results['name'].str.contains(web_name, case=False, na=False)]
        if actual_row.empty:
            # Fallback to element_id match
            actual_row = actual_results[actual_results['element_id'] == element_id]
        
        if actual_row.empty:
            actual_points = 0
        else:
            actual_points = actual_row.iloc[0]['actual_points']
        
        # For captain, multiply by 2
        if is_captain:
            predicted_points *= 2
            actual_points *= 2
        
        return {
            'element_id': element_id,
            'web_name': player['web_name'],
            'position': player['position'],
            'predicted_points': predicted_points,
            'actual_points': actual_points,
            'error': actual_points - predicted_points,
            'absolute_error': abs(actual_points - predicted_points),
            'is_captain': is_captain
        }
    
    def _save_performance_data(self, gameweek: int, performance: Dict) -> None:
        """Save performance data to CSV files."""
        try:
            # Save team-level performance
            team_record = {
                'gameweek': gameweek,
                'timestamp': performance['timestamp'],
                'formation': performance['formation'],
                'squad_predicted': performance['squad_accuracy']['total_predicted'],
                'squad_actual': performance['squad_accuracy']['total_actual'],
                'squad_difference': performance['squad_accuracy']['difference'],
                'squad_mae': performance['squad_accuracy']['mae'],
                'squad_rmse': performance['squad_accuracy']['rmse'],
                'xi_predicted': performance['xi_accuracy']['total_predicted'],
                'xi_actual': performance['xi_accuracy']['total_actual'],
                'xi_difference': performance['xi_accuracy']['difference'],
                'xi_mae': performance['xi_accuracy']['mae'],
                'xi_rmse': performance['xi_accuracy']['rmse'],
                'captain_predicted': performance['captain_performance']['predicted_points'],
                'captain_actual': performance['captain_performance']['actual_points'],
                'captain_error': performance['captain_performance']['error']
            }
            
            # Append to team performance file
            team_df = pd.DataFrame([team_record])
            if self.team_performance_file.exists():
                existing_df = pd.read_csv(self.team_performance_file)
                # Remove existing record for this gameweek if it exists
                existing_df = existing_df[existing_df['gameweek'] != gameweek]
                team_df = pd.concat([existing_df, team_df], ignore_index=True)
            
            team_df.to_csv(self.team_performance_file, index=False)
            
            # Save player-level performance
            player_records = []
            
            # Squad players
            for player in performance['squad_performance']['players']:
                player_records.append({
                    'gameweek': gameweek,
                    'element_id': player['element_id'],
                    'web_name': player['web_name'],
                    'position': player['position'],
                    'team_type': 'squad',
                    'predicted_points': player['predicted_points'],
                    'actual_points': player['actual_points'],
                    'error': player['error'],
                    'absolute_error': player['absolute_error'],
                    'is_captain': False,
                    'in_starting_xi': player['element_id'] in [p['element_id'] for p in performance['xi_performance']['players']]
                })
            
            # Add captain/vice-captain flags
            captain_id = performance['captain_performance']['element_id']
            vc_id = performance['vice_captain_performance']['element_id']
            
            for record in player_records:
                if record['element_id'] == captain_id:
                    record['is_captain'] = True
                elif record['element_id'] == vc_id:
                    record['is_vice_captain'] = True
                else:
                    record['is_vice_captain'] = False
            
            # Append to player performance file
            player_df = pd.DataFrame(player_records)
            if self.player_performance_file.exists():
                existing_df = pd.read_csv(self.player_performance_file)
                # Remove existing records for this gameweek if they exist
                existing_df = existing_df[existing_df['gameweek'] != gameweek]
                player_df = pd.concat([existing_df, player_df], ignore_index=True)
            
            player_df.to_csv(self.player_performance_file, index=False)
            
            logger.info(f"Saved performance data for GW{gameweek}")
            
        except Exception as e:
            logger.error(f"Error saving performance data for GW{gameweek}: {e}")
    
    def _update_summary_stats(self) -> None:
        """Update overall summary statistics."""
        try:
            if not self.team_performance_file.exists():
                return
            
            df = pd.read_csv(self.team_performance_file)
            
            if df.empty:
                return
            
            summary = {
                'last_updated': datetime.now().isoformat(),
                'gameweeks_tracked': len(df),
                'first_gameweek': int(df['gameweek'].min()),
                'latest_gameweek': int(df['gameweek'].max()),
                
                # Squad performance
                'squad_metrics': {
                    'avg_mae': float(df['squad_mae'].mean()),
                    'avg_rmse': float(df['squad_rmse'].mean()),
                    'total_predicted_points': float(df['squad_predicted'].sum()),
                    'total_actual_points': float(df['squad_actual'].sum()),
                    'total_difference': float(df['squad_difference'].sum()),
                    'avg_points_per_gw_predicted': float(df['squad_predicted'].mean()),
                    'avg_points_per_gw_actual': float(df['squad_actual'].mean())
                },
                
                # Starting XI performance
                'xi_metrics': {
                    'avg_mae': float(df['xi_mae'].mean()),
                    'avg_rmse': float(df['xi_rmse'].mean()),
                    'total_predicted_points': float(df['xi_predicted'].sum()),
                    'total_actual_points': float(df['xi_actual'].sum()),
                    'total_difference': float(df['xi_difference'].sum()),
                    'avg_points_per_gw_predicted': float(df['xi_predicted'].mean()),
                    'avg_points_per_gw_actual': float(df['xi_actual'].mean())
                },
                
                # Captain performance
                'captain_metrics': {
                    'total_predicted_points': float(df['captain_predicted'].sum()),
                    'total_actual_points': float(df['captain_actual'].sum()),
                    'total_error': float(df['captain_error'].sum()),
                    'avg_error': float(df['captain_error'].mean()),
                    'avg_points_per_gw_predicted': float(df['captain_predicted'].mean()),
                    'avg_points_per_gw_actual': float(df['captain_actual'].mean())
                }
            }
            
            # Save summary
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Updated summary statistics: {summary['gameweeks_tracked']} gameweeks tracked")
            
        except Exception as e:
            logger.error(f"Error updating summary statistics: {e}")
    
    def get_performance_summary(self) -> Optional[Dict]:
        """Get current performance summary."""
        try:
            if self.summary_file.exists():
                with open(self.summary_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading performance summary: {e}")
        return None
    
    def get_gameweek_performance(self, gameweek: int) -> Optional[Dict]:
        """Get performance data for a specific gameweek."""
        try:
            prediction_file = self.performance_dir / f"predicted_team_gw{gameweek}.json"
            if prediction_file.exists():
                with open(prediction_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading performance for GW{gameweek}: {e}")
        return None
    
    def generate_performance_report(self) -> str:
        """Generate a formatted performance report."""
        summary = self.get_performance_summary()
        if not summary:
            return "No performance data available."
        
        report = f"""
FPL Model Performance Report
============================

Tracking Period: GW{summary['first_gameweek']} - GW{summary['latest_gameweek']} ({summary['gameweeks_tracked']} gameweeks)
Last Updated: {summary['last_updated']}

SQUAD PERFORMANCE (15 players):
-------------------------------
• Average Points per GW: {summary['squad_metrics']['avg_points_per_gw_predicted']:.1f} predicted vs {summary['squad_metrics']['avg_points_per_gw_actual']:.1f} actual
• Total Points: {summary['squad_metrics']['total_predicted_points']:.1f} predicted vs {summary['squad_metrics']['total_actual_points']:.1f} actual
• Overall Difference: {summary['squad_metrics']['total_difference']:+.1f} points
• Mean Absolute Error: {summary['squad_metrics']['avg_mae']:.2f} points per player
• Root Mean Square Error: {summary['squad_metrics']['avg_rmse']:.2f} points per player

STARTING XI PERFORMANCE:
------------------------
• Average Points per GW: {summary['xi_metrics']['avg_points_per_gw_predicted']:.1f} predicted vs {summary['xi_metrics']['avg_points_per_gw_actual']:.1f} actual
• Total Points: {summary['xi_metrics']['total_predicted_points']:.1f} predicted vs {summary['xi_metrics']['total_actual_points']:.1f} actual
• Overall Difference: {summary['xi_metrics']['total_difference']:+.1f} points
• Mean Absolute Error: {summary['xi_metrics']['avg_mae']:.2f} points per player
• Root Mean Square Error: {summary['xi_metrics']['avg_rmse']:.2f} points per player

CAPTAIN PERFORMANCE:
-------------------
• Average Points per GW: {summary['captain_metrics']['avg_points_per_gw_predicted']:.1f} predicted vs {summary['captain_metrics']['avg_points_per_gw_actual']:.1f} actual
• Total Points: {summary['captain_metrics']['total_predicted_points']:.1f} predicted vs {summary['captain_metrics']['total_actual_points']:.1f} actual
• Average Error: {summary['captain_metrics']['avg_error']:+.1f} points per GW

ACCURACY ASSESSMENT:
-------------------
• Squad Prediction Accuracy: {(1 - summary['squad_metrics']['avg_mae'] / summary['squad_metrics']['avg_points_per_gw_actual']) * 100:.1f}%
• Starting XI Accuracy: {(1 - summary['xi_metrics']['avg_mae'] / summary['xi_metrics']['avg_points_per_gw_actual']) * 100:.1f}%
"""
        return report
    
    def save_predicted_team(
        self, 
        gameweek: int, 
        squad: List[Dict], 
        starting_xi: List[Dict], 
        captain: Dict, 
        vice_captain: Dict, 
        predictions_df: pd.DataFrame,
        formation: str
    ) -> None:
        """
        Save predicted team data for later performance comparison.
        
        Args:
            gameweek: Gameweek number
            squad: 15-player squad data
            starting_xi: 11-player starting XI
            captain: Captain data
            vice_captain: Vice-captain data
            predictions_df: Full predictions DataFrame
            formation: Formation string (e.g., "5-4-1")
        """
        try:
            # Extract player data for storage
            squad_data = [self._extract_player_prediction(player, predictions_df) for player in squad]
            xi_data = [self._extract_player_prediction(player, predictions_df) for player in starting_xi]
            captain_data = self._extract_player_prediction(captain, predictions_df) if captain else {}
            vc_data = self._extract_player_prediction(vice_captain, predictions_df) if vice_captain else {}
            
            prediction_data = {
                'gameweek': gameweek,
                'timestamp': datetime.now().isoformat(),
                'formation': formation,
                'squad': squad_data,
                'starting_xi': xi_data,
                'captain': captain_data,
                'vice_captain': vc_data
            }
            
            # Save to file
            prediction_file = self.performance_dir / f"predicted_team_gw{gameweek}.json"
            with open(prediction_file, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            
            logger.info(f"Saved predicted team for GW{gameweek} - Squad: {len(squad_data)}, XI: {len(xi_data)}")
            
        except Exception as e:
            logger.error(f"Error saving predicted team for GW{gameweek}: {e}")
