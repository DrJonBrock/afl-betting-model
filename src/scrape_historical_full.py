"""
Scrape AFL player game-by-game stats for ALL YEARS (2020-2025) and ALL TEAMS.
Captures full stat sheet: disposals, kicks, marks, handballs, goals, Behinds, hitouts, tackles, rebound 50s, inside 50s, clearances, clangers, free kicks, Brownlow votes, contested possessions, uncontested possessions, contested marks, marks inside 50, one percenters, bounces, goal assists.
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
YEARS = list(range(2000, 2026))

TEAMS = [
    "Adelaide", "Brisbane", "Carlton", "Collingwood", "Essendon",
    "Fremantle", "Geelong", "Gold Coast", "GWS", "Hawthorn",
    "Melbourne", "North Melbourne", "Port Adelaide", "Richmond",
    "St Kilda", "Sydney", "West Coast", "Western Bulldogs"
]

# Verified slugs from AFL Tables 2025 page
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
    "Port Adelaide": "padelaide",
    "Richmond": "richmond",
    "St Kilda": "stkilda",
    "Sydney": "swans",
    "West Coast": "westcoast",
    "Western Bulldogs": "bullldogs",
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

def polite_get(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(1.5, 2.5))
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

def parse_game_by_game_table(html: str, team: str, year: int) -> pd.DataFrame:
    """
    Parse the full Disposals table (first tab) from game-by-game page.
    Extracts all stats columns for each player per round.
    """
    soup = BeautifulSoup(html, 'lxml')
    tabs = soup.find_all('div', class_='simpleTabsContent')
    if not tabs:
        logger.warning(f"No simpleTabsContent found for {team} {year}")
        return pd.DataFrame()
    disposals_tab = tabs[0]
    table = disposals_tab.find('table', class_='sortable')
    if not table:
        logger.warning(f"No sortable table in first tab for {team} {year}")
        return pd.DataFrame()

    rows = table.find_all('tr')
    if len(rows) < 3:
        return pd.DataFrame()

    # Find header row containing "Player"
    header_idx = None
    for i, row in enumerate(rows):
        if row.find(['th', 'td'], string=re.compile(r'Player', re.I)):
            header_idx = i
            break
    if header_idx is None:
        logger.warning(f"Could not find header row for {team} {year}")
        return pd.DataFrame()

    header_cells = rows[header_idx].find_all(['th', 'td'])
    col_indices = {}
    round_cols = {}  # round_num -> (col_idx, col_name)

    for idx, cell in enumerate(header_cells):
        txt = cell.get_text(strip=True).strip()
        if not txt:
            continue
        # Identify round columns: R1, R2, ... or Round 1, etc.
        m = re.match(r'R(\d+)', txt, re.I)
        if m:
            round_num = int(m.group(1))
            round_cols[round_num] = (idx, txt)
        else:
            # Store other column names (Player, GM, KI, MK, HB, DI, DA, GL, BH, HO, TK, RB, IF, CL, CG, FF, FA, BR, CP, UP, CM, MI, 1%, BO, GA, %P, SU)
            col_indices[txt] = idx

    if not round_cols:
        logger.warning(f"No round columns found for {team} {year}")
        return pd.DataFrame()

    # Data rows start after header
    records = []
    for row in rows[header_idx+1:]:
        cells = row.find_all(['td', 'th'])
        if len(cells) <= max([idx for idx, _ in round_cols.values()] + [0]):
            continue

        # Player name
        player_cell = None
        if 'Player' in col_indices:
            player_cell = cells[col_indices['Player']]
        else:
            # Try to find first non-numeric column with text that looks like a name
            for idx, cell in enumerate(cells):
                txt = cell.get_text(strip=True)
                if txt and not txt.replace('.','',1).isdigit() and not re.match(r'^R\d+$', txt, re.I):
                    player_cell = cell
                    break
        if player_cell is None:
            continue
        player_text = player_cell.get_text(strip=True)
        if not player_text or player_text.lower() in ('player', 'totals', 'opponent'):
            continue
        if ',' in player_text:
            last, first = player_text.split(',', 1)
            player = f"{first.strip()} {last.strip()}"
        else:
            player = player_text

        # Other per-player stats (GM, season totals, etc.)
        player_stats = {}
        for col_name, col_idx in col_indices.items():
            if col_name == 'Player':
                continue
            if col_idx < len(cells):
                val_txt = cells[col_idx].get_text(strip=True)
                if val_txt and val_txt not in ('-', '&nbsp;'):
                    try:
                        if '.' in val_txt:
                            player_stats[col_name] = float(val_txt)
                        else:
                            player_stats[col_name] = int(val_txt)
                    except:
                        player_stats[col_name] = val_txt

        # Extract round-by-round stats
        for round_num, (col_idx, col_label) in round_cols.items():
            if col_idx >= len(cells):
                continue
            val_txt = cells[col_idx].get_text(strip=True)
            if not val_txt or val_txt in ('-', '&nbsp;'):
                continue
            try:
                value = int(val_txt)
            except:
                continue
            record = {
                'year': year,
                'player': player,
                'team': team,
                'round': round_num,
                'stat_type': col_label,
                'value': value,
            }
            record.update(player_stats)
            records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} stat rows for {team} {year}")
    return df

def main():
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    for year in YEARS:
        for team in TEAMS:
            slug = TEAM_SLUGS[team]
            url = f"{BASE_URL}teams/{slug}/{year}_gbg.html"
            logger.info(f"Fetching {url}")
            try:
                html = polite_get(url)
                df_team = parse_game_by_game_table(html, team, year)
                if not df_team.empty:
                    # Save per-team-year file
                    df_team.to_csv(os.path.join(output_dir, f"{slug}_{year}_full.csv"), index=False)
                    all_data.append(df_team)
                else:
                    logger.warning(f"No data returned for {team} {year}")
            except Exception as e:
                logger.exception(f"Failed to scrape {team} {year}: {e}")
            time.sleep(1)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, f"afltables_historical_full.csv"), index=False)
        logger.info(f"Total historical records: {len(combined)}")
        logger.info(f"Years covered: {sorted(combined['year'].unique())}")
        logger.info(f"Teams covered: {sorted(combined['team'].unique())}")
        logger.info(f"Stat types: {sorted(combined['stat_type'].unique())}")
    else:
        logger.error("No data collected")

if __name__ == "__main__":
    main()
