"""
Robust 2026 fixture parser: iterate tables in order, track round, extract matches.
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
    time.sleep(random.uniform(0.3, 0.8))
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def extract_matches_from_table(table):
    matches = []
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) < 4:
            continue
        # Away team: first td with link
        away_link = tds[0].find('a')
        if not away_link:
            continue
        away = away_link.get_text(strip=True)
        # Home team: second td with link
        home_link = tds[1].find('a')
        if not home_link:
            continue
        home = home_link.get_text(strip=True)
        # Venue: look for link to /venues/ in any later td
        venue = None
        for td in tds[2:]:
            ven_link = td.find('a', href=re.compile(r'/venues/'))
            if ven_link:
                venue = ven_link.get_text(strip=True)
                break
        if venue:
            matches.append((away, home, venue))
    return matches

def parse_fixture(html):
    soup = BeautifulSoup(html, 'lxml')
    tables = soup.find_all('table')
    current_round = None
    fixtures = []
    for table in tables:
        # Check if this table is a round header
        header_text = table.get_text(strip=True)
        m = re.match(r'Round\s+(\d+)', header_text, re.I)
        if m:
            current_round = int(m.group(1))
            continue
        # If we have a round set, and table contains match rows (with venue links), parse
        if current_round is not None:
            # Quick check: does table have any rows with at least 4 cells and a venue link?
            if table.find('a', href=re.compile(r'/venues/')):
                matches = extract_matches_from_table(table)
                for away, home, venue in matches:
                    fixtures.append((current_round, away, home, venue))
    df = pd.DataFrame(fixtures, columns=['round','away_team','home_team','venue'])
    return df

def main():
    print("Fetching 2026 fixture...")
    html = get(SEASON_URL)
    df = parse_fixture(html)
    print(f"Found {len(df)} matches across rounds {df['round'].min()}-{df['round'].max()}")
    print(df.head(10).to_string(index=False))
    df.to_csv('data/raw/2026_fixture.csv', index=False)
    print("Saved fixture to data/raw/2026_fixture.csv")

if __name__ == "__main__":
    main()
