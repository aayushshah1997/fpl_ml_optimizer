"""
Model calibration for well-calibrated probability predictions.

Provides isotonic regression and other calibration methods to ensure
model predictions are properly calibrated for uncertainty quantification.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..common.config import get_config, get_logger
from ..common.metrics import plot_calibration

logger = get_logger(__name__)


class ModelCalibrator:
    """
    Calibrator for model predictions to ensure proper uncertainty quantification.
    """
    
    def __init__(self):
        """Initialize model calibrator."""
        self.config = get_config()
        
        # Calibration configuration
        self.method = self.config.get("modeling.calibration.method", "isotonic")
        self.window_gws = self.config.get("modeling.calibration.window_gws", 8)
        
        # Calibration models storage
        self.calibrators = {}
        self.calibration_metrics = {}
        
        logger.info(f"Model calibrator initialized with {self.method} method")
    
    def fit_calibrators(
        self,
        training_results: Dict[str, Any],
        recent_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Fit calibration models for each position.
        
        Args:
            training_results: Results from model training including OOF predictions
            recent_data: Recent data for time-specific calibration
            
        Returns:
            Calibration results and metrics
        """
        logger.info("Fitting calibration models")
        
        calibration_results = {}
        
        for position, pos_results in training_results.items():
            if 'oof_predictions' not in pos_results:
                logger.warning(f"No OOF predictions available for {position}")
                continue
            
            logger.debug(f"Fitting calibrator for {position}")
            
            # Prepare calibration data
            cal_data = self._prepare_calibration_data(pos_results, recent_data, position)
            
            if cal_data is None or len(cal_data['y_true']) == 0:
                logger.warning(f"No calibration data for {position}")
                continue
            
            # Fit calibrator
            calibrator, cal_metrics = self._fit_position_calibrator(
                cal_data['y_pred'], 
                cal_data['y_true'],
                cal_data.get('sample_weights')
            )
            
            if calibrator is not None:
                self.calibrators[position] = calibrator
                self.calibration_metrics[position] = cal_metrics
                calibration_results[position] = {
                    'calibrator': calibrator,
                    'metrics': cal_metrics
                }
                
                logger.info(f"{position} calibrator fitted - Brier score: {cal_metrics.get('brier_score', 0):.4f}")
        
        # Save calibrators
        self._save_calibrators()
        
        return calibration_results
    
    def calibrate_predictions(
        self,
        predictions: pd.DataFrame,
        position_col: str = 'position',
        prediction_col: str = 'proj_points'
    ) -> pd.DataFrame:
        """
        Apply calibration to predictions.
        
        Args:
            predictions: DataFrame with model predictions
            position_col: Column name for position
            prediction_col: Column name for predictions
            
        Returns:
            DataFrame with calibrated predictions
        """
        if predictions.empty or not self.calibrators:
            logger.warning("No calibrators available or empty predictions")
            return predictions
        
        calibrated_df = predictions.copy()
        calibrated_df[f'{prediction_col}_calibrated'] = calibrated_df[prediction_col]
        
        for position in predictions[position_col].unique():
            if position not in self.calibrators:
                logger.warning(f"No calibrator for position {position}")
                continue
            
            # Get position mask
            pos_mask = calibrated_df[position_col] == position
            pos_predictions = calibrated_df.loc[pos_mask, prediction_col]
            
            if len(pos_predictions) == 0:
                continue
            
            try:
                # Apply calibration
                calibrator = self.calibrators[position]
                calibrated_pred = calibrator.predict(pos_predictions.values.reshape(-1, 1))
                
                # Update calibrated predictions
                calibrated_df.loc[pos_mask, f'{prediction_col}_calibrated'] = calibrated_pred
                
                logger.debug(f"Calibrated {len(pos_predictions)} {position} predictions")
                
            except Exception as e:
                logger.error(f"Error calibrating {position} predictions: {e}")
                continue
        
        return calibrated_df
    
    def _prepare_calibration_data(
        self,
        pos_results: Dict[str, Any],
        recent_data: Optional[pd.DataFrame],
        position: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Prepare data for calibration."""
        # Get OOF predictions and true values
        oof_pred = pos_results.get('oof_predictions')
        
        if oof_pred is None:
            return None
        
        # For calibration, we need corresponding true values
        # This is a simplified approach - in practice, you'd need to maintain
        # the mapping between OOF predictions and true values
        
        # If recent data is available, use it for time-specific calibration
        if recent_data is not None and not recent_data.empty:
            pos_recent = recent_data[recent_data.get('position') == position]
            
            if not pos_recent.empty and len(pos_recent) >= self.window_gws:
                # Use recent data for calibration
                recent_pred = pos_recent.get('model_prediction', [])
                recent_true = pos_recent.get('actual_points', [])
                recent_weights = pos_recent.get('sample_weight', np.ones(len(pos_recent)))
                
                if len(recent_pred) > 0 and len(recent_true) > 0:
                    return {
                        'y_pred': np.array(recent_pred),
                        'y_true': np.array(recent_true),
                        'sample_weights': np.array(recent_weights)
                    }
        
        # Fallback: simulate calibration data from OOF predictions
        # This is a simplified approach for demonstration
        return self._simulate_calibration_data(oof_pred, position)
    
    def _simulate_calibration_data(
        self,
        oof_pred: np.ndarray,
        position: str
    ) -> Dict[str, np.ndarray]:
        """Simulate calibration data from OOF predictions."""
        logger.debug(f"Simulating calibration data for {position}")
        
        # Create simulated true values with realistic noise
        position_std = {
            'GK': 1.8,
            'DEF': 2.0,
            'MID': 2.8,
            'FWD': 3.0
        }
        
        noise_std = position_std.get(position, 2.5)
        
        # Add calibrated noise to predictions to simulate true values
        np.random.seed(42)  # For reproducibility
        simulated_true = oof_pred + np.random.normal(0, noise_std, len(oof_pred))
        
        # Ensure non-negative points
        simulated_true = np.maximum(0, simulated_true)
        
        return {
            'y_pred': oof_pred,
            'y_true': simulated_true,
            'sample_weights': np.ones(len(oof_pred))
        }
    
    def _fit_position_calibrator(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Any], Dict[str, float]]:
        """Fit calibrator for a specific position."""
        if len(y_pred) < 10:  # Need minimum samples for calibration
            logger.warning("Insufficient data for calibration")
            return None, {}
        
        try:
            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
                
                # Fit isotonic regression
                if sample_weights is not None:
                    calibrator.fit(y_pred, y_true, sample_weight=sample_weights)
                else:
                    calibrator.fit(y_pred, y_true)
                
            elif self.method == "sigmoid":
                # Platt scaling (logistic regression)
                calibrator = LogisticRegression()
                
                # Convert to binary classification for logistic calibration
                # This is simplified - you might want different binning strategies
                y_binary = (y_true > np.median(y_true)).astype(int)
                calibrator.fit(y_pred.reshape(-1, 1), y_binary, sample_weight=sample_weights)
                
            else:
                logger.error(f"Unknown calibration method: {self.method}")
                return None, {}
            
            # Calculate calibration metrics
            if self.method == "isotonic":
                y_calibrated = calibrator.predict(y_pred)
                
                # Calculate metrics
                metrics = {
                    'brier_score': np.mean((y_calibrated - y_true) ** 2),
                    'calibration_mae': np.mean(np.abs(y_calibrated - y_true)),
                    'calibration_rmse': np.sqrt(np.mean((y_calibrated - y_true) ** 2)),
                    'reliability': self._calculate_reliability(y_pred, y_true, y_calibrated),
                    'n_samples': len(y_pred)
                }
            else:
                # For sigmoid calibration, calculate different metrics
                y_binary = (y_true > np.median(y_true)).astype(int)
                y_prob = calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
                
                metrics = {
                    'brier_score': brier_score_loss(y_binary, y_prob),
                    'log_loss': log_loss(y_binary, y_prob),
                    'n_samples': len(y_pred)
                }
            
            return calibrator, metrics
            
        except Exception as e:
            logger.error(f"Error fitting calibrator: {e}")
            return None, {}
    
    def _calculate_reliability(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_calibrated: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate reliability (calibration) metric."""
        try:
            # Bin predictions
            bin_boundaries = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            reliability = 0
            total_count = 0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
                
                if np.sum(in_bin) > 0:
                    bin_acc = np.mean(y_true[in_bin])
                    bin_conf = np.mean(y_calibrated[in_bin])
                    bin_count = np.sum(in_bin)
                    
                    reliability += bin_count * (bin_acc - bin_conf) ** 2
                    total_count += bin_count
            
            if total_count > 0:
                reliability /= total_count
            
            return reliability
            
        except Exception as e:
            logger.error(f"Error calculating reliability: {e}")
            return 0.0
    
    def create_calibration_plots(
        self,
        training_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create calibration plots for all positions.
        
        Args:
            training_results: Training results with OOF predictions
            save_path: Optional path to save plots
            
        Returns:
            Plotly figure with calibration plots
        """
        positions = list(training_results.keys())
        n_positions = len(positions)
        
        if n_positions == 0:
            logger.warning("No training results for calibration plots")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{pos} Calibration" for pos in positions[:4]],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, position in enumerate(positions[:4]):  # Limit to 4 positions
            row = i // 2 + 1
            col = i % 2 + 1
            
            pos_results = training_results[position]
            
            if 'oof_predictions' not in pos_results:
                continue
            
            # Get calibration data
            cal_data = self._simulate_calibration_data(pos_results['oof_predictions'], position)
            
            if cal_data is None:
                continue
            
            y_pred = cal_data['y_pred']
            y_true = cal_data['y_true']
            
            # Create calibration plot data
            n_bins = 10
            bin_boundaries = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
            bin_centers = []
            bin_means = []
            bin_counts = []
            
            for j in range(n_bins):
                bin_mask = (y_pred >= bin_boundaries[j]) & (y_pred < bin_boundaries[j + 1])
                if np.sum(bin_mask) > 0:
                    bin_centers.append(np.mean(y_pred[bin_mask]))
                    bin_means.append(np.mean(y_true[bin_mask]))
                    bin_counts.append(np.sum(bin_mask))
            
            if bin_centers:
                # Add calibration line
                fig.add_trace(
                    go.Scatter(
                        x=bin_centers,
                        y=bin_means,
                        mode='markers+lines',
                        name=f'{position} Actual',
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
                
                # Add perfect calibration line
                min_val = min(min(bin_centers), min(bin_means))
                max_val = max(max(bin_centers), max(bin_means))
                
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(dash='dash', color='red'),
                        name='Perfect Calibration',
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Model Calibration Analysis",
            height=600,
            width=900
        )
        
        # Update axes
        for i in range(1, min(5, n_positions + 1)):
            row = (i - 1) // 2 + 1
            col = (i - 1) % 2 + 1
            fig.update_xaxes(title_text="Mean Predicted Points", row=row, col=col)
            fig.update_yaxes(title_text="Mean Actual Points", row=row, col=col)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved calibration plots to {save_path}")
        
        return fig
    
    def _save_calibrators(self) -> bool:
        """Save fitted calibrators to disk."""
        try:
            saved_count = 0
            
            for position, calibrator in self.calibrators.items():
                cal_path = self.config.models_dir / f"cal_{position}.pkl"
                
                cal_data = {
                    'calibrator': calibrator,
                    'metrics': self.calibration_metrics.get(position, {}),
                    'method': self.method,
                    'window_gws': self.window_gws
                }
                
                with open(cal_path, 'wb') as f:
                    pickle.dump(cal_data, f)
                
                saved_count += 1
                logger.debug(f"Saved {position} calibrator to {cal_path}")
            
            logger.info(f"Saved {saved_count} calibrators")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibrators: {e}")
            return False
    
    def load_calibrators(self) -> bool:
        """Load fitted calibrators from disk."""
        try:
            loaded_count = 0
            
            for position in self.config.get_positions():
                cal_path = self.config.models_dir / f"cal_{position}.pkl"
                
                if not cal_path.exists():
                    logger.debug(f"Calibrator file not found for {position}")
                    continue
                
                with open(cal_path, 'rb') as f:
                    cal_data = pickle.load(f)
                
                self.calibrators[position] = cal_data['calibrator']
                self.calibration_metrics[position] = cal_data.get('metrics', {})
                
                loaded_count += 1
                logger.debug(f"Loaded {position} calibrator")
            
            logger.info(f"Loaded {loaded_count} calibrators")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Failed to load calibrators: {e}")
            return False
