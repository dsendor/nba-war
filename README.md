# 🏀 NBA Wins Above Replacement (WAR)

A father-son data science project to calculate NBA player value using Adjusted Plus-Minus regression.

## Overview

This project builds a Jupyter Notebook that calculates NBA **Wins Above Replacement (WAR)** for the 2024-25 season. It uses play-by-play data from the official NBA API, parses on-court lineups into "stints," and runs a Ridge regression to isolate each player's individual contribution.

The notebook is written with a **teaching tone** — designed so a 13-year-old who knows math (but not data science) can follow along and understand every step.

## What We're Building

| Section | Topic |
|---------|-------|
| 1 | Setup & Data Collection (NBA API) |
| 2 | Parse Play-by-Play into Stints |
| 3 | Build the Stint Matrix |
| 4 | Tiny Example (Teaching Moment) |
| 5 | Run the Ridge Regression |
| 6 | Convert Ratings to WAR |
| 7 | Validation & Visualization |
| 8 | Win Probability Leverage (Bonus) |
| 9 | Summary & Next Steps |

## Quick Start

**Recommended: using [uv](https://docs.astral.sh/uv/)**

```bash
# Install uv (once, if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/dsendor/nba-war.git
cd nba-war

# Create venv and install all dependencies
uv sync

# Launch Jupyter inside the project environment
uv run jupyter notebook nba_war.ipynb
```

**Alternative: using pip**

```bash
git clone https://github.com/dsendor/nba-war.git
cd nba-war
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook nba_war.ipynb
```

## Key Configuration

At the top of the notebook, you can adjust:

```python
SEASON = "2024-25"
N_GAMES = None           # Set to e.g. 200 to limit data pull for testing
REPLACEMENT_LEVEL = -2.0 # Per 100 possessions
CACHE_DIR = "./nba_data_cache"
FORCE_REFRESH = False    # Set True to re-pull data even if cache exists
```

## Tech Stack

- **Python 3.10+**
- `nba_api` — play-by-play and player data
- `pandas` / `numpy` — data wrangling
- `scikit-learn` — Ridge regression
- `scipy` — sparse matrix operations
- `matplotlib` / `plotly` — charts
- `tqdm` — progress bars

## Outputs

- `nba_war_2024_25.csv` — full WAR leaderboard
- `nba_data_cache/` — cached API data (parquet files, auto-generated)

## How WAR is Calculated

1. Pull every play-by-play event for every regular season game
2. Parse which 10 players were on court during each "stint" (uninterrupted stretch)
3. Build a sparse matrix: each row = one stint, each column = one player (+1 home, -1 away)
4. Run weighted Ridge regression to isolate each player's contribution in points per 100 possessions
5. Convert to WAR: `marginal_value × (minutes / 240) × (82 / 30.89)`

## Full Build Instructions

See [INSTRUCTIONS.md](./INSTRUCTIONS.md) for the complete specification.

---

*A learning project. Built with love for the game and for teaching.*
