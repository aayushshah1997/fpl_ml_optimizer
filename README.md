# FPL ML Optimizer ⚽

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

> **Production-grade machine learning system for Fantasy Premier League optimization with multi-league data integration, advanced planning, and automated tuning.**

## 🚀 Key Features

- **🎯 Multi-League Data Integration**: Combines FPL data with FBRef API for comprehensive player history
- **🧠 Intelligent Training Modes**: Warm Start vs Full ML based on gameweek progression  
- **📊 4-Year Walk-Forward Backtesting**: Temporal validation with EWMA weighting and no data leakage
- **⚙️ Auto-Tuning with Optuna**: Hyperparameter optimization for LightGBM + Monte Carlo + captain policies
- **🎪 Position-Specific Models**: Separate parameter tuning for GK/DEF/MID/FWD with position-aware optimization
- **🔄 Advanced Transfer Strategy**: Auto-optimization of when to roll vs make transfers with hit cost analysis
- **👑 Intelligent Captain Policy**: Mean/CVaR/mixed captaincy selection using Monte Carlo scenarios
- **📈 10-Week Planning**: Multi-week transfer planner with risk optimization and chip timing
- **📱 Beautiful Dashboard**: Streamlit interface with performance analytics and strategy controls
- **🔒 Production Ready**: Comprehensive logging, caching, error handling, and monitoring

## 📊 Data Sources & Coverage

### Historical Data
- **Primary**: Last 4 full Premier League seasons + current season-to-date
- **Multi-League Priors**: Up to 2 seasons from La Liga, Serie A, Bundesliga, Ligue 1 for new signings
- **Recency Weighting**: Recent games weighted exponentially higher with season and league strength adjustments

### New Signings Support
- Automatic integration of FBRef API history from previous leagues
- Per-90 metrics adjusted by league strength coefficients
- Cold-start shrinkage to global means when minute data is scarce
- Manual player mapping via `data/fbr_player_map.csv`

## ⚡ Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/aayushshah1997/fpl_ml_optimizer.git
cd fpl_ml_optimizer
cp .env.example .env
```

### 2. Configure Environment
Add your credentials to `.env`:
```bash
# FBRef API (get from: curl -X POST https://fbrapi.com/generate_api_key)
FBR_API_KEY=your_key_here

# FPL Account
FPL_EMAIL=your_email@example.com
FPL_PASSWORD=your_password
FPL_ENTRY_ID=your_entry_id
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Bootstrap System
```bash
make bootstrap
```
This single command will:
- Fetch all player and FBRef data
- Build manager mappings and rotation priors  
- Train all ML models with fresh data
- Display cache creation summary

### 5. Launch Dashboard
```bash
make dash
```

## 🎯 Training Modes

### Warm Start (GW < 8)
- Lighter GBM models with fewer trees
- Higher shrinkage for non-Premier League data
- Reduced cross-validation folds for faster training
- Ideal for early season when data is limited

### Full ML (GW ≥ 8)  
- Full per-position gradient boosting models
- Complete 5-fold time series cross-validation
- Isotonic calibration on recent 8-game windows
- Maximum model complexity and accuracy

Mode is automatically selected based on current gameweek and configuration.

## 🔧 Available Commands

```bash
# Core Pipeline
make train          # Full training + prediction
make predict        # Prediction only (requires trained models)
make audit          # Feature coverage analysis
make plan10         # 10-week transfer planning

# Dashboard & Analysis  
make dash           # Launch Streamlit dashboard
make cache-stats    # Check cache utilization

# Health Check
make doctor         # Repository audit and health check

# Development
make test           # Run test suite
make lint           # Code quality checks
make format         # Auto-format code
make clean          # Clear cache and artifacts
```

## 🔬 Auto-Tuning & Backtesting

Run Optuna-powered hyperparameter optimization with 4-year walk-forward backtesting:

```bash
# Run full hyperparameter search (60 trials, 60min timeout)
make tune

# Quick tune for faster iteration (20 trials, 30min timeout)
make tune-quick

# Retrain models with best found parameters
make retrain

# Run tuning then retrain automatically
make tune_and_retrain

# View tuning results and leaderboard
make tune-results
```

### Walk-Forward Backtesting
- **4-Season Window**: Tests on 2021-2025 with realistic data availability
- **Temporal Validation**: Uses only data available before each gameweek
- **EWMA Weighting**: Recent training samples weighted exponentially higher
- **Position-Specific Models**: Separate parameter tuning for GK/DEF/MID/FWD
- **Captain Policy Testing**: Evaluates mean/CVaR/mixed captain selection strategies

### Hyperparameter Optimization
- **Optuna Integration**: State-of-the-art Bayesian optimization
- **Multi-Objective**: Optimizes for total points, risk-adjusted points, or Sharpe-like ratios
- **Pruning**: MedianPruner stops unpromising trials early
- **Parallel-Safe**: Configurable trial parallelism for faster search

## 📱 Dashboard Features

### 1. Predicted Team
- **Best XI Optimization**: Risk-adjusted team selection with captain choice
- **Confidence Intervals**: P10/P50/P90 projections with Monte Carlo simulation
- **Recent Form Display**: Rolling 3/5 game form alongside projections  
- **Training Mode Indicator**: Shows whether using Warm Start or Full ML

### 2. Model Performance
- **Cross-Validation Metrics**: MAE/RMSE by position from recent training
- **Feature Importance**: Top predictive features by position
- **Calibration Quality**: Reliability diagrams and residual analysis
- **Coverage Statistics**: Feature availability across player universe

### 3. 10-Week Planner
- **GW1 Baseline Integration**: Load your actual starting team
- **Auto-Transfer Strategy**: Backtests and tunes when to roll vs make transfers
- **Captaincy Policy**: Optimizes (C)/(VC) via mean/CVaR/mixed strategies
- **Risk Management**: Balance expected returns vs uncertainty with CVaR parameters
- **Interactive Controls**: Adjust bank, transfers, starting gameweek, strategy settings

### 4. Managers Audit
- **Rotation Risk Analysis**: Manager-specific rotation patterns and priors
- **Data Quality Metrics**: Coverage and reliability of rotation data
- **Team-by-Team Breakdown**: Current manager assignments and historical analysis

### 5. Backtest & Auto-Tuning
- **Study Overview**: Optuna optimization session results and metrics
- **Trials Leaderboard**: All hyperparameter trials ranked by performance
- **Parameter Analysis**: Sensitivity analysis and parameter importance
- **Walk-Forward Results**: Temporal validation performance by season/gameweek

## 🏗️ Architecture

```
fpl_ml_optimizer/
├── fpl_ai/
│   ├── src/
│   │   ├── common/          # Config, caching, logging, utilities
│   │   ├── providers/       # Data APIs (FPL, FBRef, injuries, fixtures)
│   │   ├── features/        # Feature engineering pipeline
│   │   ├── plan/            # Transfer strategy optimization and planning
│   │   ├── tuning/          # Walk-forward backtesting and captain policies
│   │   ├── modeling/        # ML models, calibration, Monte Carlo
│   │   ├── optimize/        # Team optimization and formations
│   │   └── cli/             # Command-line interfaces
│   ├── app/                 # Streamlit dashboard
│   ├── data/                # Static data files and mappings
│   ├── cache/               # API response caching
│   ├── models/              # Trained model artifacts
│   └── artifacts/           # Training outputs and logs
├── docs/                    # Documentation
├── settings.yaml            # Configuration
├── pyproject.toml          # Dependencies and project metadata
└── Makefile                # Build commands
```

## 🎛️ Configuration

Key settings in `settings.yaml`:

```yaml
training:
  seasons_back: 4                    # How many seasons to include
  extra_leagues: [12,11,20,13,10]   # Additional leagues for priors
  staging:
    warm_until_gw: 8                # When to switch to full ML

modeling:
  per_position:
    enabled: true                   # Enable position-specific models
  gbm:
    n_estimators: 1200              # Model complexity
  gbm_by_pos:                       # Position-specific overrides
    GK: {}                          # Goalkeeper parameters
    DEF: {}                         # Defender parameters
    MID: {}                         # Midfielder parameters  
    FWD: {}                         # Forward parameters

backtest:
  seasons: ["2021-2022","2022-2023","2023-2024","2024-2025"]
  start_gw: 2                       # First GW to backtest
  end_gw: 38                        # Last GW to backtest
  objective: "risk_adj_points"      # Optimization target

autotune:
  enabled: true                     # Enable hyperparameter tuning
  n_trials: 60                      # Number of Optuna trials
  timeout_min: 60                   # Study timeout

captain:
  policy: "mix"                     # Captain selection strategy
  mix_lambda: 0.60                  # Mean vs CVaR weight

mc:
  num_scenarios: 2000               # Monte Carlo scenarios
  lambda_risk: 0.20                 # Risk penalty weight
```

## 🔍 Model Interpretability

- **Feature Importance**: Understand what drives predictions
- **Residual Analysis**: Identify model strengths and weaknesses
- **Calibration Plots**: Verify prediction reliability
- **Cross-Validation**: Time series splits for realistic evaluation

## 📊 Performance Tracking

- **Out-of-Sample Testing**: Never train on future data
- **Live Tracking**: Compare predictions vs actual points weekly
- **Feature Drift**: Monitor data quality and feature stability
- **Weekly Retraining**: Models update with latest data

## 🛡️ Security & Privacy

- **Environment Variables**: All credentials stored in `.env` file only
- **No Hardcoded Secrets**: API keys and passwords never printed or logged
- **Local Processing**: All data processing happens locally on your machine

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes. Fantasy football involves risk, and past performance does not guarantee future results. Always do your own research and make informed decisions.

## 🙏 Acknowledgments

- Fantasy Premier League for providing the official API
- FBRef for comprehensive football statistics
- The FPL community for insights and data sharing
- Open source libraries that power this system

---

Built with ❤️ for the FPL community

[🔗 GitHub Repository](https://github.com/aayushshah1997/fpl_ml_optimizer) | [📖 Documentation](docs/) | [🐛 Report Issues](https://github.com/aayushshah1997/fpl_ml_optimizer/issues)