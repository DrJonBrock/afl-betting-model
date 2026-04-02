"""
Simplified 2026 match scraper: fetch season page, extract all match stat links with round numbers by parsing the anchor structure.
"""
import requests
import pandas as pd
import re
import logging
from bs4 import BeautifulSoup
import time
import random
import os
from urllib.parse import urljoin

BASE = "https://afltables.com"
SEASON_URL = "https://afltables.com/afl/seas/2026.html"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def get(url):
    time.sleep(random.uniform(0.5,1.0))
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def extract_matches_by_round(html):
    soup = BeautifulSoup(html, 'lxml')
    rounds_data = {}
    # Find all <a name="X"> where X is a round number
    anchors = soup.find_all('a', {'name': re.compile(r'^\d+$')})
    for anchor in anchors:
        rn = int(anchor['name'])
        # Find the next <table> after this anchor (contains matches for that round)
        next_table = anchor.find_next('table')
        if not next_table:
            continue
        # Within that table, find all rows with match stats links
        matches = []
        for a in next_table.find_all('a', href=True):
            if 'Match stats' in a.get_text(strip=True):
                href = urljoin(BASE, a['href'])
                # Get home/away from row cells
                tr = a.find_parent('tr')
                if tr:
                    tds = tr.find_all('td')
                    if len(tds) >= 2:
                        away = tds[0].get_text(strip=True)
                        home = tds[1].get_text(strip=True)
                        matches.append((away, home, href))
        rounds_data[rn] = matches
    return rounds_data

def parse_match(html, round_num):
    soup = BeautifulSoup(html, 'lxml')
    records = []
    # Find all sortable tables (team stats)
    for table in soup.find_all('table', class_='sortable'):
        # Determine team from preceding heading
        team = None
        prev = table.find_previous(['h4','b','strong'])
        if prev:
            team = prev.get_text(strip=True)
        rows = table.find_all('tr')
        if len(rows)<2: continue
        # Find header row index
        hdr_idx = None
        for i,row in enumerate(rows):
            if row.find(['th','td'], string=re.compile(r'Player', re.I)):
                hdr_idx = i
                break
        if hdr_idx is None: continue
        hdr = rows[hdr_idx].find_all(['th','td'])
        col_index = {}
        for idx, cell in enumerate(hdr):
            t = cell.get_text(strip=True).upper()
            if 'PLAYER' in t: col_index['player'] = idx
            elif t == 'K': col_index['kicks'] = idx
            elif t == 'HB': col_index['handballs'] = idx
            elif t == 'M': col_index['marks'] = idx
            elif t in ('D','DI'): col_index['disposals'] = idx
            elif t == 'GL': col_index['goals'] = idx
            elif t == 'TK': col_index['tackles'] = idx
        if 'player' not in col_index: continue
        for row in rows[hdr_idx+1:]:
            cells = row.find_all(['td','th'])
            if len(cells) <= col_index['player']: continue
            pname = cells[col_index['player']].get_text(strip=True)
            if not pname or pname.lower() in ('player','totals','opponent'): continue
            if ',' in pname:
                last, first = pname.split(',',1)
                player = f"{first.strip()} {last.strip()}"
            else:
                player = pname
            def stat(key):
                idx = col_index.get(key)
                if idx is None or idx>=len(cells): return 0
                try: return int(cells[idx].get_text(strip=True))
                except: return 0
            kicks = stat('kicks')
            handballs = stat('handballs')
            disposals = stat('disposals') or (kicks+handballs)
            marks = stat('marks')
            goals = stat('goals')
            tackles = stat('tackles')
            records.append({
                'round': round_num,
                'player': player,
                'team': team,
                'kicks': kicks,
                'handballs': handballs,
                'disposals': disposals,
                'marks': marks,
                'goals': goals,
                'tackles': tackles,
            })
    return pd.DataFrame(records)

def main():
    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.info("Fetching season page...")
    html = get(SEASON_URL)
    rounds = extract_matches_by_round(html)
    logger.info(f"Found rounds: {list(rounds.keys())}")
    all_data = []
    for rn, matches in rounds.items():
        # Only include rounds <=3 (or maybe 4 if you confirm it's finished)
        if rn > 4:
            logger.info(f"Skipping round {rn} (future)")
            continue
        logger.info(f"Round {rn} has {len(matches)} matches")
        for away, home, url in matches:
            logger.info(f"  {away} vs {home}")
            try:
                mhtml = get(url)
                df = parse_match(mhtml, rn)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.exception(f"Failed match {url}: {e}")
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        out_path = os.path.join(out_dir, "afltables_2026_matches.csv")
        combined.to_csv(out_path, index=False)
        logger.info(f"Saved {len(combined)} player-round records to {out_path}")
    else:
        logger.error("No data collected")

if __name__ == "__main__":
    main()
