# SHL Forecast - CLI Quick Start

## 🚀 Get Started in 3 Commands

```bash
# 1. Run the full pipeline (this does everything)
./forecast pipeline

# 2. View the analysis output
cat reports/simulations/analysis_*.json | tail -50

# 3. View fixture predictions
cat reports/simulations/fixture_predictions.csv
```

## 📋 Common Commands

```bash
# Get help
./forecast --help

# Scrape latest data
./forecast scrape

# Clean data
./forecast clean

# Train model (fast)
./forecast train

# Train model (advanced - more accurate)
./forecast train --advanced

# Run simulation (10,000 iterations - default)
./forecast simulate

# Run simulation (custom iterations)
./forecast simulate -n 50000

# Analyze results
./forecast analyze

# Predict upcoming fixtures
./forecast predict

# Run entire workflow
./forecast pipeline
```

## 🎯 Quick Workflows

### First Time Setup
```bash
./forecast pipeline
```

### Update Forecasts (daily)
```bash
./forecast scrape
./forecast clean
./forecast train --advanced
./forecast simulate -n 25000
./forecast analyze
./forecast predict
```

### Just Get Predictions
```bash
./forecast predict
```

## 📁 Where Are My Results?

```
reports/simulations/
├── sim_results_*.csv           # Raw simulation data
├── analysis_*.json             # Championship/relegation probabilities
└── fixture_predictions.csv     # Match predictions

data/clean/
├── results.csv                 # Historical match results
└── fixtures.csv                # Upcoming matches

data/processed/
└── team_stats.csv             # Team strength ratings

models/
└── poisson_params.pkl         # Trained model
```

## 💡 Tips

- **Fast testing**: Use `-n 100` for quick tests
- **Production runs**: Use `-n 50000` or higher for accurate probabilities
- **Advanced model**: Add `--advanced` for better accuracy (takes longer)
- **Pipeline**: The `pipeline` command runs everything in the right order

## 🔧 Alternative Usage

If you prefer to use Python directly:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run CLI directly
python cli.py --help
python cli.py pipeline
```

## 📖 Full Documentation

See `CLI_GUIDE.md` for complete documentation with examples and advanced usage.

## 🏒 Web Interface

To use the Streamlit web interface instead:

```bash
# First, restore the full app
cp app.py.backup app.py

# Then run Streamlit
streamlit run app.py --server.port 5000
```

The web interface provides:
- Interactive visualizations
- Dashboard with charts
- Database management
- Odds integration (with API key)
