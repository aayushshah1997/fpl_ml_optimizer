"""
Expected minutes model for predicting player game time.

Predicts whether a player will start and how many minutes they'll play,
which is crucial for accurate per-90 projections.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from ..common.config import get_config, get_logger
from ..common.cache import get_cache
from ..common.metrics import calculate_metrics

logger = get_logger(__name__)


class MinutesModel:
    """
    Model for predicting expected minutes played.
    """
    
    def __init__(self):
        """Initialize minutes model."""
        self.config = get_config()
        self.cache = get_cache()
        
        # Model configuration
        self.lookback_windows = self.config.get("modeling.minutes.lookbacks", [3, 5, 8])
        self.min_cap = self.config.get("modeling.minutes.min_cap", 0)
        self.max_cap = self.config.get("modeling.minutes.max_cap", 90)
        
        # Model storage
        self.model = None
        self.feature_names = None
        self.model_metrics = {}
        
        logger.info("Minutes model initialized")
    
    def train(
        self,
        training_data: pd.DataFrame,
        mode: str = "full",
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train the minutes prediction model.
        
        Args:
            training_data: Training dataset
            mode: Training mode ('warm' or 'full')
            save_model: Whether to save the trained model
            
        Returns:
            Training metrics and model info
        """
        logger.info(f"Training minutes model in {mode} mode")
        
        if training_data.empty:
            logger.error("No training data provided")
            return {}
        
        # Prepare training data
        X, y, sample_weights = self._prepare_training_data(training_data)
        
        if len(X) == 0:
            logger.error("No valid training samples after preparation")
            return {}
        
        # Train model based on mode
        if mode == "warm":
            model_results = self._train_warm_model(X, y, sample_weights)
        else:
            model_results = self._train_full_model(X, y, sample_weights)
        
        self.model = model_results['model']
        self.feature_names = model_results['features']
        self.model_metrics = model_results['metrics']
        
        # Save model if requested
        if save_model:
            self._save_model()
        
        logger.info(f"Minutes model training completed. MAE: {self.model_metrics.get('mae', 0):.3f}")
        return model_results
    
    def predict(
        self,
        prediction_data: pd.DataFrame,
        load_model: bool = True
    ) -> pd.Series:
        """
        Predict expected minutes for players.
        
        Args:
            prediction_data: Data for prediction
            load_model: Whether to load saved model if not already loaded
            
        Returns:
            Series of predicted minutes
        """
        if prediction_data.empty:
            return pd.Series(dtype=float)
        
        # Load model if needed
        if self.model is None and load_model:
            if not self._load_model():
                logger.warning("No trained model available, using heuristic prediction")
                return self._heuristic_prediction(prediction_data)
        
        if self.model is None:
            logger.warning("No model available, using heuristic prediction")
            return self._heuristic_prediction(prediction_data)
        
        # Prepare prediction features
        X_pred = self._prepare_prediction_features(prediction_data)
        
        if len(X_pred) == 0:
            logger.warning("No valid prediction features, using heuristic")
            return self._heuristic_prediction(prediction_data)
        
        # Make predictions
        try:
            predictions = self.model.predict(X_pred)
            
            # Apply caps
            predictions = np.clip(predictions, self.min_cap, self.max_cap)
            
            # Convert to Series with original index
            pred_series = pd.Series(predictions, index=prediction_data.index)
            
            logger.debug(f"Predicted minutes for {len(pred_series)} players")
            return pred_series
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return self._heuristic_prediction(prediction_data)
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features and target for training."""
        # Target: next gameweek minutes
        if 'minutes_next' not in df.columns:
            # Create target by shifting minutes
            # Handle different column names for kickoff_time
            kickoff_col = 'kickoff_time' if 'kickoff_time' in df.columns else 'kickoff_time_x' if 'kickoff_time_x' in df.columns else 'kickoff_time_y'
            df = df.sort_values(['element_id', kickoff_col])
            df['minutes_next'] = df.groupby('element_id')['minutes'].shift(-1)
        
        # Remove rows without target
        df_clean = df.dropna(subset=['minutes_next']).copy()
        
        if df_clean.empty:
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
        
        # Feature selection
        features = self._get_feature_columns(df_clean)
        
        # Build feature matrix
        X = df_clean[features].copy()
        
        # Fill missing values
        X = self._fill_missing_features(X)
        
        # Target
        y = df_clean['minutes_next']
        
        # Sample weights
        sample_weights = df_clean.get('sample_weight', pd.Series(1.0, index=df_clean.index))
        
        logger.info(f"Prepared minutes training data: {len(X)} samples, {len(features)} features")
        return X, y, sample_weights
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for minutes prediction."""
        base_features = [
            'avail_prob',           # Availability probability
            'fixture_difficulty',   # Fixture difficulty
            'starts_pct',          # Historical start percentage
            'rotation_risk',       # Rotation risk score
        ]
        
        # Rolling minutes features
        rolling_features = []
        for window in self.lookback_windows:
            rolling_features.extend([
                f'minutes_r{window}',
                f'starts_r{window}',
            ])
        
        # Position features (one-hot encoded)
        position_features = []
        if 'position' in df.columns:
            positions = df['position'].unique()
            for pos in positions:
                pos_col = f'position_{pos}'
                df[pos_col] = (df['position'] == pos).astype(int)
                position_features.append(pos_col)
        
        # Team congestion features
        congestion_features = [
            'fixture_congestion',
            'days_since_last_game',
            'extra_rest'
        ]
        
        # Combine all features
        all_features = base_features + rolling_features + position_features + congestion_features
        
        # Filter to existing columns
        existing_features = [col for col in all_features if col in df.columns]
        
        logger.debug(f"Selected {len(existing_features)} features for minutes model")
        return existing_features
    
    def _fill_missing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in feature matrix."""
        # Default values by feature type
        defaults = {
            'avail_prob': 0.85,
            'fixture_difficulty': 3,
            'starts_pct': 0.5,
            'rotation_risk': 0.25,
            'fixture_congestion': 0,
            'days_since_last_game': 7,
            'extra_rest': 0
        }
        
        X_filled = X.copy()
        
        # Fill with defaults
        for col in X_filled.columns:
            if col in defaults:
                X_filled[col] = X_filled[col].fillna(defaults[col])
            elif col.startswith('minutes_r') or col.startswith('starts_r'):
                X_filled[col] = X_filled[col].fillna(0)
            elif col.startswith('position_'):
                X_filled[col] = X_filled[col].fillna(0)
            else:
                # Median fill for other columns
                X_filled[col] = X_filled[col].fillna(X_filled[col].median())
        
        return X_filled
    
    def _train_warm_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series
    ) -> Dict[str, Any]:
        """Train lightweight model for warm start mode."""
        logger.info("Training warm start minutes model")
        
        # Use Ridge regression for warm start (simpler, faster)
        model = Ridge(alpha=1.0, random_state=42)
        
        try:
            # Fit model
            model.fit(X, y, sample_weight=sample_weights)
            
            # Calculate OOF predictions for validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=3,  # Fewer folds for warm start
                scoring='neg_mean_absolute_error',
                fit_params={'sample_weight': sample_weights}
            )
            
            # Make predictions for metrics
            y_pred = model.predict(X)
            
            # Calculate metrics
            metrics = calculate_metrics(y.values, y_pred, sample_weights.values)
            metrics['cv_mae'] = -cv_scores.mean()
            metrics['cv_mae_std'] = cv_scores.std()
            
            return {
                'model': model,
                'features': list(X.columns),
                'metrics': metrics,
                'mode': 'warm'
            }
            
        except Exception as e:
            logger.error(f"Error training warm minutes model: {e}")
            return self._fallback_model(X, y, sample_weights)
    
    def _train_full_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series
    ) -> Dict[str, Any]:
        """Train full gradient boosting model."""
        logger.info("Training full minutes model")
        
        # LightGBM for full training
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'random_state': 42,
            'verbose': -1
        }
        
        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            oof_predictions = np.zeros(len(y))
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                w_train = sample_weights.iloc[train_idx]
                
                # Create datasets
                train_set = lgb.Dataset(X_train, label=y_train, weight=w_train)
                val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
                
                # Train model
                fold_model = lgb.train(
                    lgb_params,
                    train_set,
                    valid_sets=[val_set],
                    num_boost_round=200,
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
                
                # Predict on validation set
                val_pred = fold_model.predict(X_val, num_iteration=fold_model.best_iteration)
                oof_predictions[val_idx] = val_pred
                
                # Calculate fold score
                fold_mae = mean_absolute_error(y_val, val_pred)
                cv_scores.append(fold_mae)
                
                logger.debug(f"Fold {fold + 1} MAE: {fold_mae:.3f}")
            
            # Train final model on all data
            full_dataset = lgb.Dataset(X, label=y, weight=sample_weights)
            final_model = lgb.train(
                lgb_params,
                full_dataset,
                num_boost_round=int(np.mean([m.best_iteration for m in [fold_model]])),
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Calculate comprehensive metrics
            metrics = calculate_metrics(y.values, oof_predictions, sample_weights.values)
            metrics['cv_mae'] = np.mean(cv_scores)
            metrics['cv_mae_std'] = np.std(cv_scores)
            
            return {
                'model': final_model,
                'features': list(X.columns),
                'metrics': metrics,
                'oof_predictions': oof_predictions,
                'mode': 'full'
            }
            
        except Exception as e:
            logger.error(f"Error training full minutes model: {e}")
            return self._fallback_model(X, y, sample_weights)
    
    def _fallback_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series
    ) -> Dict[str, Any]:
        """Fallback to simple random forest if other models fail."""
        logger.warning("Using fallback Random Forest model")
        
        try:
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            model.fit(X, y, sample_weight=sample_weights)
            y_pred = model.predict(X)
            
            metrics = calculate_metrics(y.values, y_pred, sample_weights.values)
            
            return {
                'model': model,
                'features': list(X.columns),
                'metrics': metrics,
                'mode': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Even fallback model failed: {e}")
            return {
                'model': None,
                'features': [],
                'metrics': {},
                'mode': 'failed'
            }
    
    def _prepare_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        if self.feature_names is None:
            logger.error("No feature names available")
            return pd.DataFrame()
        
        # Get only the features used in training
        prediction_features = []
        
        for feature in self.feature_names:
            if feature in df.columns:
                prediction_features.append(feature)
            else:
                # Create missing features with defaults
                if feature.startswith('position_'):
                    # Position one-hot encoding
                    position = feature.replace('position_', '')
                    df[feature] = (df.get('position', '') == position).astype(int)
                    prediction_features.append(feature)
                else:
                    logger.warning(f"Missing feature for prediction: {feature}")
        
        if not prediction_features:
            return pd.DataFrame()
        
        X_pred = df[prediction_features].copy()
        X_pred = self._fill_missing_features(X_pred)
        
        return X_pred
    
    def _heuristic_prediction(self, df: pd.DataFrame) -> pd.Series:
        """Fallback heuristic minutes prediction."""
        logger.info("Using heuristic minutes prediction")
        
        # Simple heuristic based on availability and recent form
        base_minutes = 75  # Default expected minutes
        
        predictions = pd.Series(base_minutes, index=df.index)
        
        # Adjust for availability
        if 'avail_prob' in df.columns:
            predictions *= df['avail_prob']
        
        # Adjust for recent minutes if available
        for window in [3, 5, 8]:
            minutes_col = f'minutes_r{window}'
            if minutes_col in df.columns:
                recent_avg = df[minutes_col].fillna(base_minutes)
                predictions = (predictions + recent_avg) / 2
                break
        
        # Adjust for position
        if 'position' in df.columns:
            position_adjustments = {
                'GK': 0.95,   # Goalkeepers usually play full games
                'DEF': 0.85,  # Defenders rotated moderately
                'MID': 0.75,  # Midfielders rotated more
                'FWD': 0.80   # Forwards rotated moderately
            }
            
            for pos, adjustment in position_adjustments.items():
                mask = df['position'] == pos
                predictions.loc[mask] *= adjustment
        
        # Apply caps
        predictions = np.clip(predictions, self.min_cap, self.max_cap)
        
        return predictions
    
    def _save_model(self) -> bool:
        """Save trained model to disk."""
        try:
            model_path = self.config.models_dir / "model_minutes.pkl"
            
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'metrics': self.model_metrics,
                'config': {
                    'lookback_windows': self.lookback_windows,
                    'min_cap': self.min_cap,
                    'max_cap': self.max_cap
                }
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved minutes model to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save minutes model: {e}")
            return False
    
    def _load_model(self) -> bool:
        """Load trained model from disk."""
        try:
            model_path = self.config.models_dir / "model_minutes.pkl"
            
            if not model_path.exists():
                logger.warning(f"Minutes model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_metrics = model_data['metrics']
            
            # Update config if saved
            if 'config' in model_data:
                config = model_data['config']
                self.lookback_windows = config.get('lookback_windows', self.lookback_windows)
                self.min_cap = config.get('min_cap', self.min_cap)
                self.max_cap = config.get('max_cap', self.max_cap)
            
            logger.info(f"Loaded minutes model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load minutes model: {e}")
            return False


# Standalone functions for easy importing
def train_minutes_model(training_data: pd.DataFrame, mode: str = "full") -> Dict[str, Any]:
    """
    Standalone function to train minutes model.
    
    Args:
        training_data: Training dataset
        mode: Training mode ('warm' or 'full')
        
    Returns:
        Training results
    """
    model = MinutesModel()
    return model.train(training_data, mode=mode)


def predict_minutes(prediction_data: pd.DataFrame) -> pd.Series:
    """
    Standalone function to predict minutes.
    
    Args:
        prediction_data: Prediction dataset
        
    Returns:
        Predicted minutes
    """
    model = MinutesModel()
    return model.predict(prediction_data)
