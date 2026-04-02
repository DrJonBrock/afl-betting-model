"""
Scrape missing 2026 AFL team game-by-game data: Port Adelaide (power), Sydney (swans), Western Bulldogs (dogs).
Appends to existing combined dataset.
"""
import requests
import pandas as pd
import os
import logging
import re
import time
import random
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://afltables.com/afl/stats/"

# Only the missing teams
MISSING_TEAMS = ["Port Adelaide", "Sydney", "Western Bulldogs"]
TEAM_SLUGS = {
    "Port Adelaide": "padelaide",
    "Sydney": "swans",
    "Western Bulldogs": "bullldogs",
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

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

def parse_disposals_table(html: str, team: str, year: int) -> pd.DataFrame:
    soup = BeautifulSoup(html, 'lxml')
    tabs = soup.find_all('div', class_='simpleTabsContent')
    if not tabs:
        logger.warning(f"No simpleTabsContent found for {team}")
        return pd.DataFrame()
    disposals_tab = tabs[0]
    table = disposals_tab.find('table', class_='sortable')
    if not table:
        logger.warning(f"No sortable table in first tab for {team}")
        return pd.DataFrame()

    rows = table.find_all('tr')
    if len(rows) < 3:
        return pd.DataFrame()

    header_idx = None
    for i, row in enumerate(rows):
        if row.find(['th', 'td'], string=re.compile(r'Player', re.I)):
            header_idx = i
            break
    if header_idx is None:
        logger.warning(f"Could not find header row for {team}")
        return pd.DataFrame()

    header_cells = rows[header_idx].find_all(['th', 'td'])
    player_col_idx = None
    round_indices = []
    for idx, cell in enumerate(header_cells):
        txt = cell.get_text(strip=True).upper()
        if 'PLAYER' in txt:
            player_col_idx = idx
        m = re.match(r'R(\d+)', txt)
        if m:
            round_num = int(m.group(1))
            round_indices.append((idx, round_num))
    if player_col_idx is None or not round_indices:
        logger.warning(f"Missing player or round columns for {team}")
        return pd.DataFrame()

    records = []
    for row in rows[header_idx+1:]:
        cells = row.find_all(['td', 'th'])
        if len(cells) <= player_col_idx:
            continue
        player_cell = cells[player_col_idx]
        player_text = player_cell.get_text(strip=True)
        if not player_text or player_text.lower() in ('player', 'totals', 'opponent'):
            continue
        if ',' in player_text:
            last, first = player_text.split(',', 1)
            player = f"{first.strip()} {last.strip()}"
        else:
            player = player_text

        for col_idx, round_num in round_indices:
            if col_idx >= len(cells):
                continue
            val_txt = cells[col_idx].get_text(strip=True)
            if not val_txt or val_txt == '&nbsp;' or val_txt == '-':
                continue
            try:
                disposals = int(val_txt)
            except:
                continue
            records.append({
                'year': year,
                'player': player,
                'team': team,
                'round': round_num,
                'disposals': disposals,
            })

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} player-round disposals for {team}")
    return df

def main():
    year = 2026
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for team in MISSING_TEAMS:
        slug = TEAM_SLUGS[team]
        url = f"{BASE_URL}teams/{slug}/{year}_gbg.html"
        logger.info(f"Fetching {url}")
        try:
            html = polite_get(url)
            df_team = parse_disposals_table(html, team, year)
            if not df_team.empty:
                df_team.to_csv(os.path.join(output_dir, f"{slug}_{year}_gbg.csv"), index=False)
                all_data.append(df_team)
            else:
                logger.warning(f"No data returned for {team}")
        except Exception as e:
            logger.exception(f"Failed to scrape {team}: {e}")
        time.sleep(1)

    if all_data:
        # Append to existing combined file
        combined_path = os.path.join(output_dir, f"afltables_{year}_gbg.csv")
        if os.path.exists(combined_path):
            existing = pd.read_csv(combined_path)
            combined = pd.concat([existing] + all_data, ignore_index=True)
        else:
            combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(combined_path, index=False)
        logger.info(f"Updated combined file: total {len(combined)} player-round records")
        logger.info(f"Unique players: {combined['player'].nunique()}")
        logger.info(f"Rounds covered: {sorted(combined['round'].unique())}")
        logger.info(f"Teams covered: {sorted(combined['team'].unique())}")
    else:
        logger.error("No data collected for missing teams")

if __name__ == "__main__":
    main()
