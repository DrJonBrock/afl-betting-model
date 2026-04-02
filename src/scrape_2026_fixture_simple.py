"""
Simple 2026 fixture builder: parse season page to get (round, team) pairings.
We'll build a mapping of (round, team) -> opponent by scanning for tables with team links and venue links.
"""
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import time
import random

BASE = "https://afltables.com"
SEASON_URL = "https://afltables.com/afl/seas/2026.html"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get(url):
    time.sleep(random.uniform(0.3,0.8))
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def extract_team_games(html):
    """Parse round tables and produce list of (round, team, is_home, opponent, venue)"""
    soup = BeautifulSoup(html, 'lxml')
    records = []
    current_round = None
    # Traverse tables in document order
    for table in soup.find_all('table'):
        # Detect round header table
        txt = table.get_text(strip=True)
        m = re.match(r'Round\s+(\d+)', txt, re.I)
        if m:
            current_round = int(m.group(1))
            continue
        # If not a round header, check if this table contains a match (venue link)
        if current_round is not None and table.find('a', href=re.compile(r'/venues/')):
            # This table represents one or more matches? Usually a table contains two rows (away then home) for one match.
            # We'll extract team links from this table: find all <a> with /teams/
            team_links = table.find_all('a', href=re.compile(r'/teams/'))
            team_names = [a.get_text(strip=True) for a in team_links]
            # Also get venue from the venue link
            ven_link = table.find('a', href=re.compile(r'/venues/'))
            venue = ven_link.get_text(strip=True) if ven_link else None
            if len(team_names) >= 2:
                # Usually exactly 2 teams per match (away then home)
                away = team_names[0]
                home = team_names[1]
                records.append((current_round, away, home, venue))
                # Also record the reverse mapping for home team later
    return records

def main():
    print("Fetching season page...")
    html = get(SEASON_URL)
    recs = extract_team_games(html)
    df = pd.DataFrame(recs, columns=['round','away','home','venue'])
    print(f"Found {len(df)} match entries")
    print(df.head(10))
    # Build expanded mapping: for each team, row with round, team, opponent, venue, is_home
    rows = []
    for _, r in df.iterrows():
        round_num = r['round']
        venue = r['venue']
        away = r['away']
        home = r['home']
        rows.append({'round': round_num, 'team': away, 'opponent': home, 'venue': venue, 'is_home': 0})
        rows.append({'round': round_num, 'team': home, 'opponent': away, 'venue': venue, 'is_home': 1})
    mapping = pd.DataFrame(rows)
    mapping.to_csv('data/raw/2026_fixture_mapping.csv', index=False)
    print("Saved fixture mapping to data/raw/2026_fixture_mapping.csv")
    print("Unique teams:", mapping['team'].nunique())
    print("Rounds covered:", sorted(mapping['round'].unique()))

if __name__ == "__main__":
    main()
