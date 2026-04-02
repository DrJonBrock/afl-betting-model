"""
Scrape 2026 AFL player game-by-game stats (full set) from per-team pages.
Returns DataFrame: year, player, team, round, disposals, kicks, handballs, marks, goals, tackles.
"""
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import random
import os
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://afltables.com/afl/stats/"
TEAMS = [
    "Adelaide", "Brisbane", "Carlton", "Collingwood", "Essendon",
    "Fremantle", "Geelong", "Gold Coast", "GWS", "Hawthorn",
    "Melbourne", "North Melbourne", "Port Adelaide", "Richmond",
    "St Kilda", "Sydney", "West Coast", "Western Bulldogs"
]

TEAM_SLUGS = {
    "Adelaide": "adelaide",
    "Brisbane": "brisbanel",
    "Carlton": "carlton",
    "Collingwood": "collingwood",
    "Essendon": "essendon",
    "Fremantle": "fremantle",
    "Geelong": "geelong",
    "Gold Coast": "goldcoast",
    "GWS": "gws",
    "Hawthorn": "hawthorn",
    "Melbourne": "melbourne",
    "North Melbourne": "kangaroos",
    "Port Adelaide": "portadelaide",
    "Richmond": "richmond",
    "St Kilda": "stkilda",
    "Sydney": "sydney",
    "West Coast": "westcoast",
    "Western Bulldogs": "westernbulldogs",
}

STATS_NEEDED = ['disposals', 'kicks', 'handballs', 'marks', 'goals', 'tackles']
# Map how the stat names appear in table headers (they are the same but case-sensitive: "Disposals", "Kicks", etc.)
STAT_HEADER_MAP = {
    'disposals': 'Disposals',
    'kicks': 'Kicks',
    'handballs': 'Handballs',
    'marks': 'Marks',
    'goals': 'Goals',
    'tackles': 'Tackles',
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

def find_header_row(rows):
    """Find the index of the header row that contains 'Player' and round columns."""
    for i, row in enumerate(rows):
        if row.find(['th', 'td'], string=re.compile(r'Player', re.I)):
            return i
    return None

def parse_table_for_stat(html: str, stat_name: str, team: str, year: int) -> pd.DataFrame:
    """
    Parse the table corresponding to a given stat from the game-by-game page.
    Returns DataFrame with columns: year, player, team, round, stat_value (named as stat_name)
    """
    soup = BeautifulSoup(html, 'lxml')
    tabs = soup.find_all('div', class_='simpleTabsContent')
    if not tabs:
        return pd.DataFrame()

    # Find the tab whose header matches the stat name (case-sensitive as displayed)
    target_tab = None
    for tab in tabs:
        header = tab.find_previous('ul', class_='simpleTabsNavigation')
        # The tab navigation links have text equal to stat abbreviation? Actually the tabs are "DI","KI","MK", etc.
        # But we can also check the first row of the table inside the tab: it says the full stat name.
        table = tab.find('table', class_='sortable')
        if table:
            # The first row of the table's thead likely has <th colspan="5">Disposals</th> etc.
            first_th = table.find('th')
            if first_th:
                title = first_th.get_text(strip=True)
                if title.lower() == stat_name.lower():
                    target_tab = tab
                    break
    if not target_tab:
        return pd.DataFrame()

    table = target_tab.find('table', class_='sortable')
    if not table:
        return pd.DataFrame()
    rows = table.find_all('tr')
    if len(rows) < 3:
        return pd.DataFrame()

    header_idx = find_header_row(rows)
    if header_idx is None:
        return pd.DataFrame()
    header_cells = rows[header_idx].find_all(['th', 'td'])
    player_col_idx = None
    round_indices = []  # list of (col_idx, round_num)
    for idx, cell in enumerate(header_cells):
        txt = cell.get_text(strip=True).upper()
        if 'PLAYER' in txt:
            player_col_idx = idx
        m = re.match(r'R(\d+)', txt)
        if m:
            round_num = int(m.group(1))
            round_indices.append((idx, round_num))
    if player_col_idx is None or not round_indices:
        return pd.DataFrame()

    records = []
    for row in rows[header_idx+1:]:
        cells = row.find_all(['td', 'th'])
        if len(cells) <= player_col_idx:
            continue
        player_text = cells[player_col_idx].get_text(strip=True)
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
            if not val_txt or val_txt in ('&nbsp;', '-'):
                continue
            try:
                val = int(val_txt)
            except:
                continue
            records.append({
                'year': year,
                'player': player,
                'team': team,
                'round': round_num,
                stat_name: val,
            })
    return pd.DataFrame(records)

def main():
    year = 2026
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for team in TEAMS:
        slug = TEAM_SLUGS[team]
        url = f"{BASE_URL}teams/{slug}/{year}_gbg.html"
        logger.info(f"Fetching {url}")
        try:
            html = polite_get(url)
            # Scrape all needed stats for this team
            team_dfs = []
            for stat in STATS_NEEDED:
                df_stat = parse_table_for_stat(html, stat, team, year)
                if not df_stat.empty:
                    team_dfs.append(df_stat)
            if team_dfs:
                # Merge all stats on common keys (year, player, team, round) using inner joins
                team_df = team_dfs[0]
                for df in team_dfs[1:]:
                    team_df = team_df.merge(df, on=['year','player','team','round'], how='inner')
                # Save per-team file
                team_df.to_csv(os.path.join(output_dir, f"{slug}_{year}_full.csv"), index=False)
                all_data.append(team_df)
                logger.info(f"Parsed {len(team_df)} records for {team}")
            else:
                logger.warning(f"No data for {team}")
        except Exception as e:
            logger.exception(f"Failed to scrape {team}: {e}")
        time.sleep(1)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, f"afltables_{year}_full.csv"), index=False)
        logger.info(f"Total 2026 player-round records with full stats: {len(combined)}")
        logger.info(f"Unique players: {combined['player'].nunique()}")
        logger.info(f"Rounds covered: {sorted(combined['round'].unique())}")
    else:
        logger.error("No data collected")

if __name__ == "__main__":
    main()
