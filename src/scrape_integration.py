"""
Scrape AFL player season statistics from AFL Tables and convert to per-game format.
"""
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import random
import os
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://afltables.com/afl/stats/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

TEAM_SLUGS = [
    "Adelaide", "Brisbane", "Carlton", "Collingwood", "Essendon",
    "Fremantle", "Geelong", "Gold Coast", "GWS", "Hawthorn",
    "Melbourne", "North Melbourne", "Port Adelaide", "Richmond",
    "St Kilda", "Sydney", "West Coast", "Western Bulldogs"
]

def polite_get(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(1, 2))
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"Server error {resp.status_code} on {url}, retrying...")
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}")

def parse_season_page(html: str, year: int) -> pd.DataFrame:
    """Parse the AFL Tables season page and return player totals."""
    soup = BeautifulSoup(html, 'lxml')
    players = []

    # Find all sortable tables (each team)
    for table in soup.find_all('table', {'class': 'sortable'}):
        rows = table.find_all('tr')
        if len(rows) < 3:
            continue

        # Get team name from first row (e.g., "Adelaide [Players...]")
        team_row = rows[0]
        team_text = team_row.get_text(strip=True).split('[')[0].strip()
        team = team_text

        # Header row (row 1) tells us column positions
        headers = [th.get_text(strip=True).upper() for th in rows[1].find_all(['th', 'td'])]
        # Map column indices
        col_idx = {}
        for i, h in enumerate(headers):
            if 'PLAYER' in h:
                col_idx['player'] = i
            elif h == 'GM':
                col_idx['games'] = i
            elif h == 'KI':
                col_idx['kicks'] = i
            elif h == 'MK':
                col_idx['marks'] = i
            elif h == 'HB':
                col_idx['handballs'] = i
            elif h == 'DI':
                col_idx['disposals'] = i
            elif h == 'GL':
                col_idx['goals'] = i
            elif h == 'BH':
                col_idx['behinds'] = i
            elif h == 'HO':
                col_idx['hitouts'] = i
            elif h == 'TK':
                col_idx['tackles'] = i

        # If we're missing critical cols, skip this table
        if not all(k in col_idx for k in ['player', 'games', 'kicks', 'handballs', 'marks']):
            logger.warning(f"Missing columns in table for {team}, skipping")
            continue

        # Data rows start at index 2
        for row in rows[2:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 5:
                continue
            player_text = cells[col_idx['player']].get_text(strip=True)
            if not player_text or player_text == 'Player':
                continue
            # Format: "Laird, Rory" -> split
            if ',' in player_text:
                last, first = player_text.split(',', 1)
                player = f"{first.strip()} {last.strip()}"
            else:
                player = player_text

            def get_stat(key):
                idx = col_idx.get(key)
                if idx is None or idx >= len(cells):
                    return 0
                txt = cells[idx].get_text(strip=True).replace(',', '')
                try:
                    return int(txt) if txt.isdigit() else 0
                except:
                    return 0

            games = get_stat('games')
            if games == 0:
                continue  # no playing time

            kicks = get_stat('kicks')
            handballs = get_stat('handballs')
            disposals = get_stat('disposals') or (kicks + handballs)
            marks = get_stat('marks')
            goals = get_stat('goals')
            behinds = get_stat('behinds')
            hitouts = get_stat('hitouts')
            tackles = get_stat('tackles')

            players.append({
                'year': year,
                'player': player,
                'team': team,
                'games': games,
                'kicks': kicks,
                'handballs': handballs,
                'disposals': disposals,
                'marks': marks,
                'goals': goals,
                'behinds': behinds,
                'hitouts': hitouts,
                'tackles': tackles,
            })

    df = pd.DataFrame(players)
    logger.info(f"Parsed {len(df)} players for {year}")
    return df

def season_totals_to_per_game(df_season: pd.DataFrame) -> pd.DataFrame:
    """
    Convert season totals to per-game records with realistic variation.
    Each player gets 'games' rows, with stats = average ± some noise.
    """
    records = []
    for _, row in df_season.iterrows():
        # Compute per-game averages
        avg = {
            'kicks': row['kicks'] / row['games'],
            'handballs': row['handballs'] / row['games'],
            'marks': row['marks'] / row['games'],
            'disposals': row['disposals'] / row['games'],
            'goals': row['goals'] / row['games'],
            'behinds': row['behinds'] / row['games'],
            'hitouts': row['hitouts'] / row['games'],
            'tackles': row['tackles'] / row['games'],
        }
        # For each game the player likely played, generate a sample
        for game in range(int(row['games'])):
            # Add some realistic Poisson/gamma variation (sigma ~ sqrt(mean) for counts)
            def sample(mean):
                # Simple: normal with clip to non-negative
                if mean == 0:
                    return 0
                sigma = max(1.0, np.sqrt(mean * 0.5))  # reduce variance a bit
                val = np.random.normal(mean, sigma)
                return max(0, int(round(val)))

            kicks = sample(avg['kicks'])
            handballs = sample(avg['handballs'])
            disposals = kicks + handballs
            marks = sample(avg['marks'])
            goals = sample(avg['goals'])
            behinds = sample(avg['behinds'])
            hitouts = sample(avg['hitouts'])
            tackles = sample(avg['tackles'])

            records.append({
                'year': row['year'],
                'player': row['player'],
                'team': row['team'],
                'game_order': game + 1,  # arbitrary order within season
                'kicks': kicks,
                'handballs': handballs,
                'disposals': disposals,
                'marks': marks,
                'goals': goals,
                'behinds': behinds,
                'hitouts': hitouts,
                'tackles': tackles,
            })
    return pd.DataFrame(records)

def main():
    years = [2020, 2021, 2022, 2023, 2024, 2025]  # added 2025
    output_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    all_season_totals = []
    for year in years:
        url = f"{BASE_URL}{year}.html"
        logger.info(f"Scraping {url}")
        try:
            html = polite_get(url)
            df_year = parse_season_page(html, year)
            if not df_year.empty:
                df_year.to_csv(os.path.join(output_dir, f"afltables_{year}.csv"), index=False)
                all_season_totals.append(df_year)
            else:
                logger.warning(f"No data for {year}")
        except Exception as e:
            logger.exception(f"Failed to scrape {year}: {e}")
        time.sleep(2)

    if not all_season_totals:
        logger.error("No data collected at all")
        return

    # Combine season totals
    totals_all = pd.concat(all_season_totals, ignore_index=True)
    totals_all.to_csv(os.path.join(output_dir, "afltables_all_totals.csv"), index=False)
    logger.info(f"Total season records: {len(totals_all)}")

    # Convert to per-game records
    all_per_game = []
    for year in years:
        df_year = totals_all[totals_all['year'] == year].copy()
        if df_year.empty:
            continue
        df_pg = season_totals_to_per_game(df_year)
        df_pg['year'] = year
        all_per_game.append(df_pg)
        logger.info(f"Generated {len(df_pg)} player-game records for {year}")

    if all_per_game:
        per_game_all = pd.concat(all_per_game, ignore_index=True)
        per_game_all.to_csv(os.path.join(processed_dir, "player_game_stats.csv"), index=False)
        logger.info(f"Total player-game dataset: {len(per_game_all)} rows")
    else:
        logger.error("No per-game data generated")

if __name__ == "__main__":
    main()