"""
Scrape 2026 AFL match-by-match player stats (rounds 1-3).
Collects: year, round, player, team, opponent, venue, kicks, handballs, marks, disposals, goals, tackles.
"""
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import time
import random
import os
from urllib.parse import urljoin

BASE = "https://afltables.com"
SEASON_URL = "https://afltables.com/afl/seas/2026.html"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

def get(url):
    time.sleep(random.uniform(0.5, 1.0))
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def extract_fixture(html):
    """Return list of (round, home_team, away_team, match_url) for all matches."""
    soup = BeautifulSoup(html, 'lxml')
    fixtures = []
    # Find all tables that contain match rows (they have "Match stats" links)
    for table in soup.find_all('table'):
        # Look for rows with Match stats link
        for a in table.find_all('a', href=True):
            if 'Match stats' in a.get_text(strip=True):
                href = urljoin(BASE, a['href'])
                # Find the row and extract away/home teams
                tr = a.find_parent('tr')
                if not tr:
                    continue
                tds = tr.find_all('td')
                if len(tds) >= 2:
                    away = tds[0].get_text(strip=True)
                    home = tds[1].get_text(strip=True)
                    # Determine round from preceding heading
                    # Walk up to find preceding element that contains "Round X"
                    round_num = None
                    prev = tr.find_previous(['b','strong','h2','h3'])
                    while prev:
                        txt = prev.get_text(strip=True)
                        m = re.search(r'Round\s+(\d+)', txt, re.I)
                        if m:
                            round_num = int(m.group(1))
                            break
                        prev = prev.find_previous(['b','strong','h2','h3'])
                    if round_num is None:
                        continue
                    fixtures.append((round_num, away, home, href))
    # Deduplicate by match URL (some matches may appear twice)
    seen = set()
    unique = []
    for rn, away, home, url in fixtures:
        if url not in seen:
            seen.add(url)
            unique.append((rn, away, home, url))
    return unique

def parse_match_page(html, round_num, home_team, away_team):
    """Parse player stats tables; return DataFrame with records for both teams."""
    soup = BeautifulSoup(html, 'lxml')
    records = []
    # Find all sortable tables (player stats)
    for table in soup.find_all('table', class_='sortable'):
        # Determine which team this table belongs to
        # Usually there's a preceding <b> or <h4> with team name
        team = None
        prev = table.find_previous(['b','strong','h4'])
        if prev:
            team = prev.get_text(strip=True)
        # If not found, maybe the table is directly after a team row; fallback: infer from player team abbreviations? Not reliable.
        if not team:
            continue
        # Determine opponent
        opponent = away_team if team == home_team else (home_team if team == away_team else None)
        # Parse table
        rows = table.find_all('tr')
        if len(rows) < 2:
            continue
        # Find header row
        hdr_idx = None
        for i, row in enumerate(rows):
            if row.find(['th','td'], string=re.compile(r'Player', re.I)):
                hdr_idx = i
                break
        if hdr_idx is None:
            continue
        hdr_cells = rows[hdr_idx].find_all(['th','td'])
        col_idx = {}
        for idx, cell in enumerate(hdr_cells):
            txt = cell.get_text(strip=True).upper()
            if 'PLAYER' in txt:
                col_idx['player'] = idx
            elif txt == 'K':
                col_idx['kicks'] = idx
            elif txt == 'HB':
                col_idx['handballs'] = idx
            elif txt == 'M':
                col_idx['marks'] = idx
            elif txt in ('D','DI'):
                col_idx['disposals'] = idx
            elif txt == 'GL':
                col_idx['goals'] = idx
            elif txt == 'TK':
                col_idx['tackles'] = idx
        if 'player' not in col_idx:
            continue
        # Data rows
        for row in rows[hdr_idx+1:]:
            cells = row.find_all(['td','th'])
            if len(cells) <= col_idx['player']:
                continue
            ptext = cells[col_idx['player']].get_text(strip=True)
            if not ptext or ptext.lower() in ('player','totals','opponent'):
                continue
            if ',' in ptext:
                last, first = ptext.split(',',1)
                player = f"{first.strip()} {last.strip()}"
            else:
                player = ptext
            def get_stat(key):
                idx = col_idx.get(key)
                if idx is None or idx >= len(cells):
                    return 0
                try:
                    return int(cells[idx].get_text(strip=True))
                except:
                    return 0
            kicks = get_stat('kicks')
            handballs = get_stat('handballs')
            marks = get_stat('marks')
            goals = get_stat('goals')
            tackles = get_stat('tackles')
            disposals = get_stat('disposals') or (kicks + handballs)
            records.append({
                'year': 2026,
                'round': round_num,
                'player': player,
                'team': team,
                'opponent': opponent,
                'venue': None,  # we'll add venue later from match summary
                'kicks': kicks,
                'handballs': handballs,
                'marks': marks,
                'disposals': disposals,
                'goals': goals,
                'tackles': tackles,
            })
    return pd.DataFrame(records)

def get_venue_from_match_page(html):
    """Extract venue name from match page."""
    soup = BeautifulSoup(html, 'lxml')
    # Venue often appears after "Venue:" or in a link to venues
    venue_link = soup.find('a', href=re.compile(r'/venues/'))
    if venue_link:
        return venue_link.get_text(strip=True)
    # Fallback: search for Venue: pattern
    text = soup.get_text()
    m = re.search(r'Venue:\s*([^\n\r]+)', text)
    if m:
        return m.group(1).strip()
    return None

def main():
    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)
    print("Fetching season page...")
    season_html = get(SEASON_URL)
    fixtures = extract_fixture(season_html)
    print(f"Found {len(fixtures)} matches. Filtering to rounds 1-3...")
    fixtures = [(rn, aw, hm, url) for (rn, aw, hm, url) in fixtures if 1 <= rn <= 3]
    print(f"Processing {len(fixtures)} matches from rounds 1-3")
    all_records = []
    for round_num, away, home, url in fixtures:
        print(f"Round {round_num}: {away} vs {home}")
        try:
            html = get(url)
            # Get venue
            venue = get_venue_from_match_page(html)
            df_match = parse_match_page(html, round_num, home, away)
            if not df_match.empty:
                df_match['venue'] = venue
                all_records.append(df_match)
            else:
                print(f"  Warning: no player data parsed from {url}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)
    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        out_path = os.path.join(out_dir, "afl2026_matches_rounds1-3.csv")
        combined.to_csv(out_path, index=False)
        print(f"Saved {len(combined)} player-match records to {out_path}")
        print("Unique teams:", combined['team'].nunique())
        print("Rounds:", sorted(combined['round'].unique()))
    else:
        print("No data collected")

if __name__ == "__main__":
    main()
