"""
Scrape AFL fixtures for multiple years (2020-2026) from AFL Tables.
For each year, produces:
  - data/raw/fixture_{year}.csv: (round, away, home, venue)
  - data/raw/fixture_mapping_{year}.csv: (round, team, opponent, venue, is_home)
"""
import requests
import pandas as pd
import re
import sys
from bs4 import BeautifulSoup
import time
import random

BASE = "https://afltables.com"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get(url):
    time.sleep(random.uniform(0.3,0.8))
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def extract_fixture_tables(html):
    """Return list of (round, away, home, venue) from season page."""
    soup = BeautifulSoup(html, 'lxml')
    records = []
    current_round = None
    for table in soup.find_all('table'):
        txt = table.get_text(strip=True)
        m = re.match(r'Round\s+(\d+)', txt, re.I)
        if m:
            current_round = int(m.group(1))
            continue
        if current_round is not None and table.find('a', href=re.compile(r'/venues/')):
            team_links = table.find_all('a', href=re.compile(r'/teams/'))
            team_names = [a.get_text(strip=True) for a in team_links]
            ven_link = table.find('a', href=re.compile(r'/venues/'))
            venue = ven_link.get_text(strip=True) if ven_link else None
            if len(team_names) >= 2:
                away = team_names[0]
                home = team_names[1]
                records.append((current_round, away, home, venue))
    return records

def build_mapping(df_fixture):
    rows = []
    for _, r in df_fixture.iterrows():
        rnd = r['round']; venue = r['venue']; away = r['away']; home = r['home']
        rows.append({'round': rnd, 'team': away, 'opponent': home, 'venue': venue, 'is_home': 0})
        rows.append({'round': rnd, 'team': home, 'opponent': away, 'venue': venue, 'is_home': 1})
    return pd.DataFrame(rows)

def scrape_year(year):
    url = f"{BASE}/afl/seas/{year}.html"
    print(f"Scraping {year} from {url}")
    html = get(url)
    recs = extract_fixture_tables(html)
    df_fix = pd.DataFrame(recs, columns=['round','away','home','venue'])
    out_fix = f"data/raw/fixture_{year}.csv"
    df_fix.to_csv(out_fix, index=False)
    print(f"  Saved fixture: {len(df_fix)} matches")
    mapping = build_mapping(df_fix)
    out_map = f"data/raw/fixture_mapping_{year}.csv"
    mapping.to_csv(out_map, index=False)
    print(f"  Saved mapping: {len(mapping)} rows, teams {mapping['team'].nunique()}, rounds {sorted(mapping['round'].unique())[:5]}-{sorted(mapping['round'].unique())[-5:]}")
    return df_fix, mapping

def main():
    years = list(range(2010, 2027))  # 2010-2026 inclusive
    for year in years:
        try:
            scrape_year(year)
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    main()
