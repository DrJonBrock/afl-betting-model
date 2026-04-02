"""
Robust AFL data scraper with retries, delays.
Primary source: Footywire (footywire.com) – player game logs per year.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://www.footywire.com"
OUTPUT_RAW_DIR = "data/raw"
OUTPUT_PROCESSED_DIR = "data/processed"
os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def polite_get(url: str, retries: int = 3, delay_range=(2, 5)) -> str:
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(*delay_range))
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning(f"Server error {resp.status_code} on {url}, retrying...")
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")

def fetch_footywire_year_page(year: int) -> str:
    url = f"{BASE_URL}/afl/footy/players?year={year}"
    logger.info(f"Fetching Footywire year page: {url}")
    html = polite_get(url)
    with open(os.path.join(OUTPUT_RAW_DIR, f"footywire_{year}_players.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return html

def parse_footywire_player_links(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, 'lxml')
    players = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/afl/player/' in href:
            player_name = a.get_text(strip=True)
            if player_name:
                players.append({
                    'player': player_name,
                    'profile_url': href if href.startswith('http') else BASE_URL + href
                })
    logger.info(f"Found {len(players)} player profiles")
    return players

def parse_footywire_game_log(html: str, player_name: str, year: int) -> List[Dict]:
    soup = BeautifulSoup(html, 'lxml')
    records = []
    for table in soup.find_all('table'):
        headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
        if not headers or 'round' not in headers:
            continue
        try:
            idx_round = headers.index('round')
            idx_opp = next((i for i, h in enumerate(headers) if 'opp' in h), None)
            idx_k = next((i for i, h in enumerate(headers) if h == 'k'), None)
            idx_h = next((i for i, h in enumerate(headers) if h == 'h'), None)
            idx_d = next((i for i, h in enumerate(headers) if h == 'd'), None)
            idx_m = next((i for i, h in enumerate(headers) if h == 'm'), None)
        except ValueError:
            continue
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) < 5:
                continue
            try:
                round_num = cells[idx_round].get_text(strip=True)
                opponent = cells[idx_opp].get_text(strip=True) if idx_opp is not None else None
                kicks = int(cells[idx_k].get_text(strip=True)) if idx_k is not None and cells[idx_k].get_text(strip=True).isdigit() else 0
                handballs = int(cells[idx_h].get_text(strip=True)) if idx_h is not None and cells[idx_h].get_text(strip=True).isdigit() else 0
                if idx_d is not None and cells[idx_d].get_text(strip=True).isdigit():
                    disposals = int(cells[idx_d].get_text(strip=True))
                else:
                    disposals = kicks + handballs
                marks = int(cells[idx_m].get_text(strip=True)) if idx_m is not None and cells[idx_m].get_text(strip=True).isdigit() else 0
                records.append({
                    'player': player_name,
                    'year': year,
                    'round': round_num,
                    'opponent': opponent,
                    'kicks': kicks,
                    'handballs': handballs,
                    'disposals': disposals,
                    'marks': marks,
                })
            except (ValueError, IndexError):
                continue
    return records

def scrape_year(year: int, max_players: int = 200) -> pd.DataFrame:
    try:
        index_html = fetch_footywire_year_page(year)
        players = parse_footywire_player_links(index_html)
    except Exception as e:
        logger.error(f"Failed to parse year {year} from Footywire: {e}")
        return pd.DataFrame()
    all_games = []
    for player in players[:max_players]:
        try:
            profile_html = polite_get(player['profile_url'])
            safe_name = player['player'].replace(' ', '_').replace('/', '_')
            with open(os.path.join(OUTPUT_RAW_DIR, f"footywire_{year}_{safe_name}.html"), "w", encoding="utf-8") as f:
                f.write(profile_html)
            games = parse_footywire_game_log(profile_html, player['player'], year)
            all_games.extend(games)
            logger.info(f"Parsed {len(games)} games for {player['player']}")
        except Exception as e:
            logger.error(f"Failed to parse player {player['player']}: {e}")
        time.sleep(random.uniform(1, 3))
    if all_games:
        df = pd.DataFrame(all_games)
        df['game_id'] = (df['year'].astype(str) + '_' + df['player'].str.replace(' ', '_') + '_' + df['round'].astype(str))
        return df
    else:
        return pd.DataFrame()

def main():
    years = list(range(2020, 2024))
    all_players = []
    for year in years:
        try:
            df = scrape_year(year, max_players=50)  # limit for testing
            if not df.empty:
                all_players.append(df)
        except Exception as e:
            logger.exception(f"Scrape for {year} failed: {e}")
        time.sleep(1)
    if all_players:
        players_total = pd.concat(all_players, ignore_index=True)
        players_total.to_csv(os.path.join(OUTPUT_PROCESSED_DIR, "player_game_stats.csv"), index=False)
        logger.info(f"Saved {len(players_total)} player-game rows")
    else:
        logger.error("No data collected")

if __name__ == "__main__":
    main()