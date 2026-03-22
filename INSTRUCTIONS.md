# 🏀 Claude Code Instructions: Build NBA WAR Notebook

> Source: [Notion page](https://www.notion.so/c96d5acc8b7e4119a4b557cbe22ba94c) — saved 2026-03-22

---

## Goal

Build a complete, self-contained Jupyter Notebook (.ipynb) that calculates NBA Wins Above Replacement (WAR) using Adjusted Plus-Minus regression. The notebook should be fully runnable end-to-end so I can explore the results before doing this project with my 13-year-old son.

Output: A single `.ipynb` file saved to the working directory.

---

## Technical Stack

- Python 3.10+
- Libraries: `nba_api`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `plotly` (for interactive charts), `tqdm` (for progress bars)
- Format: Jupyter Notebook (`.ipynb`)

At the top of the notebook, include a cell that installs all dependencies:

```python
!pip install nba_api pandas numpy scikit-learn matplotlib plotly tqdm
```

---

## Notebook Structure

Organize the notebook into clearly labeled sections with markdown headers. Each section should have explanatory markdown cells written in a **teaching tone** — as if explaining to a smart 13-year-old who knows math but not data science. This is a father-son learning project.

---

### Section 1: Setup & Data Collection

**What to build:**

1. Import all libraries
2. Pull **play-by-play data** for the **2024-25 NBA regular season** using `nba_api`
   - Use `playbyplayv2` endpoint from `nba_api.stats.endpoints`
   - First get the list of all games for the season using `leaguegamefinder` or `leaguegamelog`
   - Loop through each game and pull play-by-play
   - Add a `time.sleep(0.6)` between API calls to avoid rate limiting
   - Cache results to a local CSV/parquet after first pull so we don't have to re-pull
3. Pull **player info** (player ID to name mapping) using `commonallplayers` endpoint
4. Pull **team standings / win totals** for validation later

**Important implementation notes:**

- The NBA API can be flaky. Wrap API calls in try/except with retries (max 3 retries with exponential backoff)
- If a full season pull is too slow (1200+ games), provide an option to limit to a subset (e.g., first 200 games) for faster iteration, with a clear variable at the top: `N_GAMES = None  # Set to an integer to limit, None for full season`
- Print progress: "Pulling game 47/1230..."
- After pulling, save to `nba_pbp_2024_25.parquet` so subsequent runs skip the pull
- At the start of the data pull section, check if the cached file exists and load it if so

---

### Section 2: Parse Play-by-Play into Stints

**What to build:**

A stint = a continuous stretch of game time where the same 10 players are on the court. A new stint begins whenever a substitution happens.

1. **Track substitutions** in the play-by-play data to determine which 5 players are on court for each team at every moment
   - The play-by-play has event types for substitutions (EVENTMSGTYPE == 8)
   - You need to figure out the **starting lineups** for each period. The play-by-play doesn't always make this explicit. A reliable approach:
     - For each period, look at the first events. Players who appear in events (shots, rebounds, fouls, etc.) before any substitution are starters.
     - If that doesn't give you 5 per team, fall back to the `boxscoretraditionalv2` endpoint filtered by period, or use the `gamerotation` endpoint to get exact rotation data.
     - **Alternatively and preferably**, use the `gamerotation` endpoint (`nba_api.stats.endpoints.gamerotation`) which gives exact IN/OUT times for every player in every game. This is the most reliable method.
2. **Build stint records**: For each stint, record:
   - `game_id`
   - `period`
   - `home_player_1` through `home_player_5` (player IDs)
   - `away_player_1` through `away_player_5` (player IDs)
   - `home_points` scored during this stint
   - `away_points` scored during this stint
   - `duration_seconds`
   - `possessions` (estimate: use the formula `possessions ≈ FGA + 0.44 * FTA - ORB + TOV` for each team during the stint, averaged across both teams. If this is too complex to calculate per-stint, approximate as `duration_seconds / 14` which gives roughly 1 possession per 14 seconds of game time)
   - `point_differential` = home_points - away_points
   - `points_per_100_poss` = (point_differential / possessions) * 100
3. **Filter out** stints shorter than 10 seconds or with 0 possessions (garbage transitions)
4. **Validation check**: Print summary stats
   - Total number of stints
   - Average stint length
   - Distribution of stint lengths (histogram)
   - Total points should roughly match season totals

---

### Section 3: Build the Stint Matrix

**What to build:**

1. Create a **player index**: map each unique player ID to a column index (0 to N_players-1)
2. Build a **sparse matrix X** where:
   - Each row = one stint
   - Each column = one player
   - Value = `+1` if player is on court for the **home** team
   - Value = `-1` if player is on court for the **away** team
   - Value = `0` if player is not on court
3. Build **target vector y** = points_per_100_poss for each stint
4. Build **weight vector w** = possessions for each stint (so longer stints count more)
5. Use `scipy.sparse.csr_matrix` for efficiency — the matrix will be ~50,000 stints × ~500 players but very sparse

**Markdown explanation cell before this section:**

> "Here's the key idea: we're setting up a big system of equations. Each stint tells us: 'When these 5 home players were on court against these 5 away players, the home team outscored the away team by X points per 100 possessions.' The regression will solve for each player's individual contribution."

---

### Section 4: The Tiny Example (Teaching Moment)

**What to build:**

Before running the full model, create a **toy example** with 3-4 players and 5-6 made-up stints. Walk through:

1. Setting up the matrix by hand
2. Solving with ordinary least squares (`np.linalg.lstsq`)
3. Showing how the solution gives each player a rating
4. Then showing what happens when players are correlated (always play together) — the solution becomes unstable
5. Then showing how Ridge regression stabilizes it

This section should be heavily annotated with markdown. It's the conceptual core.

---

### Section 5: Run the Ridge Regression

**What to build:**

```python
from sklearn.linear_model import RidgeCV

alphas = [1, 10, 50, 100, 500, 1000, 5000, 10000]
model = RidgeCV(alphas=alphas, store_cv_results=True)
model.fit(X, y, sample_weight=w)

print(f"Best alpha: {model.alpha_}")
```

1. Fit the model
2. Extract coefficients and map back to player names
3. Create a DataFrame: `player_name`, `player_id`, `team`, `minutes_played`, `rating` (coefficient from model)
4. Sort by rating and display top 20 and bottom 20
5. **Experiment cell**: Show what happens with different alpha values. Plot the top player's rating as alpha changes. Explain: high alpha = more shrinkage toward zero = more conservative estimates.

---

### Section 6: Convert to WAR

**What to build:**

1. Define replacement level: `-2.0` per 100 possessions (adjustable variable at top)
2. For each player, calculate:

```python
marginal_value = player_rating - replacement_level
war = marginal_value * (minutes_played / (48 * 5)) * (82 / 30.89)
# 48*5 = total player-minutes per game (48 min × 5 players)
# 30.89 ≈ point differential per win (can also use ~30.5)
```

3. Create final WAR leaderboard DataFrame
4. Display top 30 by WAR

**Markdown explanation:**

> "A player's rating tells us how many points per 100 possessions they add above replacement level. To turn this into wins, we need to account for (a) how many minutes they actually played, and (b) how many points of margin equals one win."

---

### Section 7: Validation & Visualization

**What to build:**

1. **Top 20 WAR bar chart** (horizontal bar, sorted, with team colors if feasible, otherwise a clean single-color scheme)
2. **Team WAR vs. Actual Wins scatter plot**
   - Sum WAR for all players on each team
   - Plot against actual team wins
   - Add a regression line and R² value
   - Label each point with team abbreviation
   - Markdown: "If our model is any good, teams with more total WAR should have more actual wins."
3. **Comparison to published metrics**
   - Try to pull the FiveThirtyEight RAPTOR historical data from GitHub: `https://raw.githubusercontent.com/fivethirtyeight/nba-player-advanced-metrics/master/nba-data-historical.csv`
   - If available for the season, scatter plot: our WAR vs. RAPTOR WAR
   - If not available (RAPTOR stopped updating), note this and just show our rankings with a "sanity check" discussion
4. **Distribution of player ratings** — histogram showing most players near zero (average), with tails
5. **Position breakdown** — box plot of ratings by position
6. **On/Off comparison for top 10 players** — bar chart showing raw on/off differential vs. our adjusted rating. The difference between these is the "teammate adjustment" which is the whole point of the model.

---

### Section 8: Win Probability Leverage (Bonus)

**What to build:**

1. **Calculate win probability** for each moment in each game:
   - Use a simple logistic model: `win_prob = f(score_margin, time_remaining, home_away)`
   - You can fit this from the data itself, or use a simple known formula. A reasonable approximation:

```python
def win_prob(margin, seconds_remaining, is_home):
    home_advantage = 0.03 if is_home else -0.03
    if seconds_remaining == 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    z = (margin / (math.sqrt(seconds_remaining) * 0.4)) + home_advantage
    return norm.cdf(z)
```

2. **Calculate leverage for each stint**: absolute change in win probability from start to end of stint
3. **Re-run the Ridge regression** with `sample_weight = w * leverage` (possessions × leverage)
4. **Compare**: Who gains the most WAR when we leverage-weight? Who loses?
   - Create a DataFrame with both standard WAR and leverage-weighted WAR
   - Sort by the difference
   - "Clutch performers" = players who gain the most
   - "Garbage time heroes" = players who lose the most
5. Scatter plot: Standard WAR vs. Leverage-Weighted WAR, with players labeled

---

### Section 9: Summary & Next Steps

Markdown cell summarizing:

- What we built
- Key findings
- What a 13-year-old should focus on learning (the concepts in Sections 4-6)
- Ideas for extensions: defensive vs. offensive breakdown, multi-year trends, playoff vs. regular season

---

## Code Quality Requirements

- **Every code cell** should have a markdown cell above it explaining what it does and why, in plain language
- Use clear variable names (no single-letter variables except in the toy example)
- Include inline comments for any non-obvious logic
- Add `%%time` magic to any cell that takes more than a few seconds
- Use `tqdm` progress bars for any loop over games
- Print shape/head of DataFrames after major transformations so the reader can see what's happening
- Handle errors gracefully — if the API is down or data is missing, print a clear message and continue with available data
- The notebook should be **fully runnable top-to-bottom** without any manual intervention (except possibly the initial data pull which takes time)

---

## Key Variables (Configurable at Top of Notebook)

Put these in the very first code cell after imports:

```python
# === CONFIGURATION ===
SEASON = "2024-25"
N_GAMES = None           # Set to int to limit games pulled (e.g., 200 for testing)
REPLACEMENT_LEVEL = -2.0 # Per 100 possessions, relative to average
CACHE_DIR = "./nba_data_cache"
FORCE_REFRESH = False    # Set True to re-pull data even if cache exists
```

---

## Known Pitfalls to Handle

1. **nba_api rate limiting**: Sleep 0.6s between calls. If you get a 429 or connection error, wait 5s and retry.
2. **Starting lineup detection**: This is the hardest data engineering problem. The `gamerotation` endpoint is the most reliable source. Fall back to parsing early play-by-play events if needed.
3. **Players traded mid-season**: They'll have entries for multiple teams. That's fine — they get one coefficient. But track their total minutes across all teams.
4. **Very short stints** (< 10 seconds): Filter these out. They add noise.
5. **Overtime**: Include OT stints. They're legitimate game time.
6. **All-Star game / exhibition**: Exclude these. Filter to regular season only.
7. **Sparse matrix**: Use `scipy.sparse` — the dense matrix would be ~500 players × 50,000 stints and waste memory.
8. **Season data availability**: The 2024-25 season may be incomplete depending on when this runs. The code should handle partial seasons gracefully — just use whatever games are available.

---

## Final Output

At the end of the notebook, save key outputs:

```python
war_leaderboard.to_csv("nba_war_2024_25.csv", index=False)
print("WAR leaderboard saved to nba_war_2024_25.csv")
```

Also display a final "headline" summary:

```
=== 2024-25 NBA WAR Leaders ===
1. [Player Name] ([Team]) — 18.4 WAR
2. [Player Name] ([Team]) — 15.2 WAR
...
```
