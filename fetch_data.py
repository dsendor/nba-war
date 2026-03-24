"""
NBA WAR Data Fetcher
====================
Downloads all raw data from the NBA stats API and saves to the cache directory.

Usage:
    python fetch_data.py

Options (edit config.py or override via env):
    FORCE_REFRESH=true python fetch_data.py   # Re-download even if cached
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import (
    leaguegamefinder,
    playbyplayv3,
    gamerotation,
    commonallplayers,
    leaguestandings,
)

from config import SEASON, N_GAMES, CACHE_DIR, FORCE_REFRESH


def api_call_with_retry(endpoint_class, max_retries=3, sleep_base=0.6, **kwargs):
    """Call an nba_api endpoint with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            time.sleep(sleep_base)
            return endpoint_class(**kwargs)
        except Exception as exc:
            wait_time = sleep_base * (2 ** attempt)
            if attempt < max_retries - 1:
                print(f"  API error (attempt {attempt+1}/{max_retries}): {exc}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"  API call failed after {max_retries} attempts: {exc}")
                return None
    return None


def fetch_players(cache_dir, season, force_refresh=False):
    cache_path = os.path.join(cache_dir, "players.parquet")
    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        print(f"Loaded player info from cache: {len(df)} players")
        return df

    print("Pulling player info from NBA API...")
    result = api_call_with_retry(
        commonallplayers.CommonAllPlayers,
        is_only_current_season=1,
        league_id="00",
        season=season,
    )
    if result is None:
        print("Could not pull player info.")
        return pd.DataFrame(columns=["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ABBREVIATION"])

    df = result.get_data_frames()[0]
    df.to_parquet(cache_path, index=False)
    print(f"Pulled {len(df)} players")
    return df


def fetch_standings(cache_dir, season, force_refresh=False):
    cache_path = os.path.join(cache_dir, "standings.parquet")
    if not force_refresh and os.path.exists(cache_path):
        print("Loaded standings from cache.")
        return pd.read_parquet(cache_path)

    print("Pulling team standings...")
    result = api_call_with_retry(
        leaguestandings.LeagueStandings,
        season=season,
        season_type="Regular Season",
        league_id="00",
    )
    if result is None:
        print("Could not pull standings.")
        return pd.DataFrame()

    df = result.get_data_frames()[0]
    df.to_parquet(cache_path, index=False)
    print(f"Pulled standings for {len(df)} teams")
    return df


def fetch_games(cache_dir, season, n_games=None, force_refresh=False):
    cache_path = os.path.join(cache_dir, "games_list.parquet")
    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        print(f"Loaded game list from cache: {len(df)} records")
    else:
        print("Pulling game list from NBA API...")
        result = api_call_with_retry(
            leaguegamefinder.LeagueGameFinder,
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        )
        if result is None:
            print("Could not pull game list.")
            return pd.DataFrame(), []
        df = result.get_data_frames()[0]
        df.to_parquet(cache_path, index=False)
        print(f"Pulled {len(df)} game records")

    unique_game_ids = df["GAME_ID"].unique().tolist()
    if n_games is not None:
        unique_game_ids = unique_game_ids[:n_games]
        print(f"Limited to {n_games} games for testing")
    print(f"Total unique games: {len(unique_game_ids)}")
    return df, unique_game_ids


def fetch_rotations(cache_dir, season, unique_game_ids, force_refresh=False, workers=3):
    cache_path = os.path.join(cache_dir, f"rotations_{season.replace('-', '_')}.parquet")
    parts_dir = os.path.join(cache_dir, "rotations_parts")

    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        print(f"Loaded rotation data from cache: {len(df):,} records across {df['GAME_ID'].nunique()} games")
        return df

    os.makedirs(parts_dir, exist_ok=True)
    if force_refresh:
        for f in os.listdir(parts_dir):
            os.remove(os.path.join(parts_dir, f))

    already_done = {f.replace(".parquet", "") for f in os.listdir(parts_dir) if f.endswith(".parquet")}
    todo = [g for g in unique_game_ids if g not in already_done]
    print(f"Rotations — already cached: {len(already_done)}, to download: {len(todo)}")

    rate_lock = threading.Lock()
    last_req = [0.0]

    def rate_limited_sleep():
        with rate_lock:
            wait = 0.7 - (time.time() - last_req[0])
            if wait > 0:
                time.sleep(wait)
            last_req[0] = time.time()

    failed = []

    def fetch_one(game_id):
        rate_limited_sleep()
        result = api_call_with_retry(gamerotation.GameRotation, game_id=game_id, sleep_base=0.3)
        if result is None:
            return game_id, False
        try:
            frames = result.get_data_frames()
            records = []
            for team_idx, team_label in enumerate(["AWAY", "HOME"]):
                if team_idx < len(frames) and len(frames[team_idx]) > 0:
                    frame = frames[team_idx].copy()
                    frame["GAME_ID"] = game_id
                    frame["TEAM_SIDE"] = team_label
                    records.append(frame)
            if records:
                pd.concat(records, ignore_index=True).to_parquet(
                    os.path.join(parts_dir, f"{game_id}.parquet"), index=False
                )
            return game_id, True
        except Exception as e:
            print(f"  Parse error {game_id}: {e}")
            return game_id, False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, g): g for g in todo}
        for future in tqdm(as_completed(futures), total=len(todo), desc="Downloading rotations"):
            game_id, ok = future.result()
            if not ok:
                failed.append(game_id)

    parts = [
        pd.read_parquet(os.path.join(parts_dir, f))
        for f in os.listdir(parts_dir)
        if f.endswith(".parquet")
    ]
    if not parts:
        print("No rotation data downloaded.")
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(cache_path, index=False)
    print(f"Rotations complete: {len(df):,} records from {df['GAME_ID'].nunique()} games")
    if failed:
        print(f"Failed ({len(failed)} games): {failed[:10]}")
    return df


def fetch_pbp(cache_dir, season, unique_game_ids, force_refresh=False, workers=3):
    cache_path = os.path.join(cache_dir, f"pbp_{season.replace('-', '_')}.parquet")
    parts_dir = os.path.join(cache_dir, "pbp_parts")

    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        print(f"Loaded PBP from cache: {len(df):,} events across {df['GAME_ID'].nunique()} games")
        return df

    os.makedirs(parts_dir, exist_ok=True)
    if force_refresh:
        for f in os.listdir(parts_dir):
            os.remove(os.path.join(parts_dir, f))

    already_done = {f.replace(".parquet", "") for f in os.listdir(parts_dir) if f.endswith(".parquet")}
    todo = [g for g in unique_game_ids if g not in already_done]
    print(f"PBP — already cached: {len(already_done)}, to download: {len(todo)}")

    rate_lock = threading.Lock()
    last_req = [0.0]

    def rate_limited_sleep():
        with rate_lock:
            wait = 0.7 - (time.time() - last_req[0])
            if wait > 0:
                time.sleep(wait)
            last_req[0] = time.time()

    failed = []

    def fetch_one(game_id):
        rate_limited_sleep()
        result = api_call_with_retry(playbyplayv3.PlayByPlayV3, game_id=game_id, sleep_base=0.3)
        if result is None:
            return game_id, False
        try:
            df = result.get_data_frames()[0]
            df["GAME_ID"] = game_id
            df.to_parquet(os.path.join(parts_dir, f"{game_id}.parquet"), index=False)
            return game_id, True
        except Exception as e:
            print(f"  Parse error {game_id}: {e}")
            return game_id, False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, g): g for g in todo}
        for future in tqdm(as_completed(futures), total=len(todo), desc="Downloading PBP"):
            game_id, ok = future.result()
            if not ok:
                failed.append(game_id)

    parts = [
        pd.read_parquet(os.path.join(parts_dir, f))
        for f in os.listdir(parts_dir)
        if f.endswith(".parquet")
    ]
    if not parts:
        print("No PBP data downloaded.")
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(cache_path, index=False)
    print(f"PBP complete: {len(df):,} events from {df['GAME_ID'].nunique()} games")
    if failed:
        print(f"Failed ({len(failed)} games): {failed[:10]}")
    return df


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"=== NBA WAR Data Fetcher ===")
    print(f"Season:        {SEASON}")
    print(f"Cache dir:     {CACHE_DIR}")
    print(f"Force refresh: {FORCE_REFRESH}")
    print(f"Game limit:    {N_GAMES if N_GAMES else 'full season'}")
    print()

    fetch_players(CACHE_DIR, SEASON, FORCE_REFRESH)
    print()

    fetch_standings(CACHE_DIR, SEASON, FORCE_REFRESH)
    print()

    _, unique_game_ids = fetch_games(CACHE_DIR, SEASON, N_GAMES, FORCE_REFRESH)
    print()

    fetch_rotations(CACHE_DIR, SEASON, unique_game_ids, FORCE_REFRESH)
    print()

    fetch_pbp(CACHE_DIR, SEASON, unique_game_ids, FORCE_REFRESH)
    print()

    print("=== All data fetched. Run the notebook to compute WAR. ===")


if __name__ == "__main__":
    main()
