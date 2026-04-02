"""
Scrape AFL player statistics from Footywire and AFL Tables.
Adapted from the AFL Fantasy League project (TypeScript → Python).
"""
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import random
import os
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Team name mapping (from AFL Fantasy League constants)
TEAM_NAME_MAPPINGS = {
    "adelaide": "Adelaide Crows",
    "brisbane": "Brisbane Lions",
    "carlton": "Carlton Blues",
    "collingwood": "Collingwood Magpies",
    "essendon": "Essendon Bombers",
    "fremantle": "Fremantle Dockers",
    "geelong": "Geelong Cats",
    "gold coast": "Gold Coast Suns",
    "greater western sydney": "GWS Giants",
    "gws": "GWS Giants",
    "hawthorn": "Hawthorn Hawks",
    "melbourne": "Melbourne Demons",
    "north melbourne": "North Melbourne Kangaroos",
    "port adelaide": "Port Adelaide Power",
    "richmond": "Richmond Tigers",
    "st kilda": "St Kilda Saints",
    "sydney": "Sydney Swans",
    "west coast": "West Coast Eagles",
    "western bulldogs": "Western Bulldogs",
    "bulldogs": "Western Bulldogs",
    "crows": "Adelaide Crows",
    "lions": "Brisbane Lions",
    "blues": "Carlton Blues",
    "magpies": "Collingwood Magpies",
    "bombers": "Essendon Bombers",
    "dockers": "Fremantle Dockers",
    "cats": "Geelong Cats",
    "suns": "Gold Coast Suns",
    "giants": "GWS Giants",
    "hawks": "Hawthorn Hawks",
    "demons": "Melbourne Demons",
    "kangaroos": "North Melbourne Kangaroos",
    "power": "Port Adelaide Power",
    "tigers": "Richmond Tigers",
    "saints": "St Kilda Saints",
    "swans": "Sydney Swans",
    "eagles": "West Coast Eagles",
}

def normalize_team_name(name: str) -> str:
    key = name.lower().strip()
    return TEAM_NAME_MAPPINGS.get(key, name)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def polite_get(url: str, retries: int = 3, delay_range=(2, 5)) -> str:
    """Fetch with retries and random delays."""
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(*delay_range))
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
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")

BASE_URL = "https://www.footywire.com"

def scrape_season_players(year: int, max_players: int = 500) -> pd.DataFrame:
    """
    Scrape player list and season totals for a given year from Footywire.
    Returns DataFrame with: player, team, games, kicks, handballs, disposals, marks, goals, behinds, hitouts, tackles
    """
    url = f"{BASE_URL}/afl/footy/players?year={year}"
    logger.info(f"Scraping {url}")
    html = polite_get(url)
    soup = BeautifulSoup(html, 'lxml')

    players = []
    # Find player links in the main table
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/afl/player/' in href:
            player_name = a.get_text(strip=True)
            if not player_name:
                continue
            # The team is often in the following span with class playerflag
            team_span = a.find_next('span', class_='playerflag')
            team_name = team_span.get('title', '') if team_span else ''
            if team_name:
                team_name = normalize_team_name(team_name)
            players.append({
                'player': player_name,
                'profile_url': href if href.startswith('http') else BASE_URL + href,
                'team': team_name,
                'year': year,
            })

    logger.info(f"Found {len(players)} player profiles for {year}, limiting to {max_players}")
    players = players[:max_players]

    all_stats = []
    for p in players:
        try:
            profile_html = polite_get(p['profile_url'])
            profile_soup = BeautifulSoup(profile_html, 'lxml')
            # Find the career stats table; look for "Career Averages" or "Supercoach"
            tables = profile_soup.find_all('table')
            career_table = None
            for table in tables:
                if table.find('th') and 'Kicks' in table.find('th').get_text():
                    career_table = table
                    break
            if not career_table:
                # Skip if no stats table found
                logger.warning(f"No stats table for {p['player']}")
                continue

            # Parse career totals row (usually first data row)
            rows = career_table.find_all('tr')
            # Look for row with "Career" or the most recent season row
            stats_row = None
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 10:
                    first_cell = cells[0].get_text(strip=True).lower()
                    if 'career' in first_cell or str(year) in first_cell:
                        stats_row = cells
                        break
            if not stats_row and len(rows) > 2:
                stats_row = rows[-1].find_all('td')  # fallback to last row

            if not stats_row or len(stats_row) < 10:
                logger.warning(f"Could not parse stats for {p['player']}")
                continue

            def get_stat(idx):
                try:
                    val = stats_row[idx].get_text(strip=True).replace(',', '')
                    return int(val) if val.isdigit() else 0
                except:
                    return 0

            kicks = get_stat(3)  # Kicks column (adjust based on actual table)
            handballs = get_stat(4)  # Handballs
            disposals = get_stat(5)  # Disposals (sometimes sum of kicks+handballs)
            marks = get_stat(6)  # Marks
            goals = get_stat(7) if len(stats_row) > 7 else 0
            behinds = get_stat(8) if len(stats_row) > 8 else 0
            hitouts = get_stat(9) if len(stats_row) > 9 else 0
            tackles = get_stat(10) if len(stats_row) > 10 else 0
            games = get_stat(2) if len(stats_row) > 2 else 0

            all_stats.append({
                'year': year,
                'player': p['player'],
                'team': p['team'],
                'games': games,
                'kicks': kicks,
                'handballs': handballs,
                'disposals': disposals or (kicks + handballs),
                'marks': marks,
                'goals': goals,
                'behinds': behinds,
                'hitouts': hitouts,
                'tackles': tackles,
            })

        except Exception as e:
            logger.error(f"Error parsing {p['player']}: {e}")
            continue
        time.sleep(random.uniform(0.5, 1.5))

    df = pd.DataFrame(all_stats)
    logger.info(f"Scraped {len(df)} player season totals for {year}")
    return df

def main():
    years = [2020, 2021, 2022, 2023, 2024]
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for year in years:
        try:
            df = scrape_season_players(year, max_players=100)  # limit for testing
            if not df.empty:
                df.to_csv(os.path.join(output_dir, f"player_season_{year}.csv"), index=False)
                all_data.append(df)
        except Exception as e:
            logger.exception(f"Scrape for {year} failed: {e}")
        time.sleep(2)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, "player_season_all.csv"), index=False)
        logger.info(f"Saved combined season data: {len(combined)} rows")
    else:
        logger.error("No data collected")

if __name__ == "__main__":
    main()
