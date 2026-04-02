"""
Scrape AFL player statistics from AFL Tables (afltables.com).
Provides per-player season totals with a clean, consistent table format.
"""
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import random
import os
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://afltables.com"
TEAMS = [
    "Adelaide", "Brisbane", "Carlton", "Collingwood", "Essendon",
    "Fremantle", "Geelong", "Gold Coast", "GWS", "Hawthorn",
    "Melbourne", "North Melbourne", "Port Adelaide", "Richmond",
    "St Kilda", "Sydney", "West Coast", "Western Bulldogs"
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

def polite_get(url: str, retries: int = 3, delay_range=(1, 3)) -> str:
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(*delay_range))
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"Error {resp.status_code} on {url}, retrying...")
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")

def scrape_team_season(team: str, year: int) -> pd.DataFrame:
    """
    Scrape player statistics for a given team and year from AFL Tables.
    URL pattern: https://afltables.com/afl/stats/teams/<Team>/<year>.html
    Returns DataFrame with player-season stats.
    """
    url = f"{BASE_URL}/afl/stats/teams/{team}/{year}.html"
    logger.info(f"Fetching {url}")
    try:
        html = polite_get(url)
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(html, 'lxml')
    # Find the main stats table - it's the first sortable table with player data
    tables = soup.find_all('table', {'class': 'sortable'})
    if not tables:
        logger.warning(f"No sortable table found for {team} {year}")
        return pd.DataFrame()

    # The player stats table is usually the first one after the team header
    table = tables[0]
    rows = table.find_all('tr')
    if len(rows) < 2:
        return pd.DataFrame()

    # Header is in first row; columns: #, Player, GM, KI, MK, HB, DI, DA, GL, BH, HO, TK, etc.
    headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(['th', 'td'])]
    # Map expected columns
    col_idx = {}
    for idx, h in enumerate(headers):
        if 'player' in h:
            col_idx['player'] = idx
        elif h == 'gm':
            col_idx['games'] = idx
        elif h == 'ki':
            col_idx['kicks'] = idx
        elif h == 'mk':
            col_idx['marks'] = idx
        elif h == 'hb':
            col_idx['handballs'] = idx
        elif h == 'di' or h == 'disposals':
            col_idx['disposals'] = idx
        elif h == 'da':
            col_idx['disposal_avg'] = idx  # not used
        elif h == 'gl':
            col_idx['goals'] = idx
        elif h == 'bh':
            col_idx['behinds'] = idx
        elif h == 'ho':
            col_idx['hitouts'] = idx
        elif h == 'tk':
            col_idx['tackles'] = idx

    # Some pages may not have all columns; ensure we have at least basics
    required = ['player', 'games', 'kicks', 'handballs', 'marks']
    if not all(c in col_idx for c in required):
        logger.warning(f"Missing columns for {team} {year}. Found: {list(col_idx.keys())}")
        return pd.DataFrame()

    records = []
    for row in rows[1:]:
        cells = row.find_all(['td', 'th'])
        if len(cells) < 5:
            continue
        # Extract values
        def get(col, default=0):
            idx = col_idx.get(col)
            if idx is None or idx >= len(cells):
                return default
            txt = cells[idx].get_text(strip=True).replace(',', '')
            try:
                return int(txt) if txt.isdigit() else default
            except:
                return default

        player_name = cells[col_idx['player']].get_text(strip=True)
        # Remove any footyzone affiliation in parentheses
        if '(' in player_name:
            player_name = player_name.split('(')[0].strip()

        games = get('games')
        kicks = get('kicks')
        handballs = get('handballs')
        marks = get('marks')
        goals = get('goals')
        behinds = get('behinds')
        hitouts = get('hitouts')
        tackles = get('tackles')
        disposals = get('disposals') or (kicks + handballs)

        records.append({
            'year': year,
            'player': player_name,
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

    df = pd.DataFrame(records)
    logger.info(f"Scraped {len(df)} players for {team} {year}")
    return df

def main():
    years = [2020, 2021, 2022, 2023, 2024]
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for year in years:
        year_data = []
        for team in TEAMS:
            try:
                df = scrape_team_season(team, year)
                if not df.empty:
                    year_data.append(df)
                time.sleep(random.uniform(1, 2))
            except Exception as e:
                logger.exception(f"Error scraping {team} {year}: {e}")
        if year_data:
            combined_year = pd.concat(year_data, ignore_index=True)
            combined_year.to_csv(os.path.join(output_dir, f"afltables_{year}.csv"), index=False)
            all_data.append(combined_year)
        logger.info(f"Completed {year}, total rows: {len(combined_year) if 'combined_year' in locals() else 0}")

    if all_data:
        all_years = pd.concat(all_data, ignore_index=True)
        all_years.to_csv(os.path.join(output_dir, "afltables_all.csv"), index=False)
        logger.info(f"Total collected: {len(all_years)} player-season records")
    else:
        logger.error("No data collected")

if __name__ == "__main__":
    main()
