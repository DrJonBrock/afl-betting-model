"""
Generic fixture scraper for a given AFL year.
"""
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import time
import random
import sys

BASE = "https://afltables.com"
SEASON_URL_TEMPLATE = "https://afltables.com/afl/seas/{year}.html"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get(url):
    time.sleep(random.uniform(0.3,0.8))
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def extract_matches_from_table(table):
    matches = []
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) < 4:
            continue
        away_link = tds[0].find('a', href=re.compile(r'/teams/'))
        if not away_link:
            continue
        away = away_link.get_text(strip=True)
        home_link = tds[1].find('a', href=re.compile(r'/teams/'))
        if not home_link:
            continue
        home = home_link.get_text(strip=True)
        venue_link = tr.find('a', href=re.compile(r'/venues/'))
        venue = venue_link.get_text(strip=True) if venue_link else None
        matches.append((away, home, venue))
    return matches

def parse_fixture(html):
    soup = BeautifulSoup(html, 'lxml')
    tables = soup.find_all('table')
    current_round = None
    fixtures = []
    for table in tables:
        txt = table.get_text(strip=True)
        m = re.match(r'Round\s+(\d+)', txt, re.I)
        if m:
            current_round = int(m.group(1))
            continue
        if current_round is not None and table.find('a', href=re.compile(r'/venues/')):
            matches = extract_matches_from_table(table)
            for away, home, venue in matches:
                fixtures.append((current_round, away, home, venue))
    return pd.DataFrame(fixtures, columns=['round','away','home','venue'])

def main():
    if len(sys.argv) < 2:
        print("Usage: python scrape_fixture.py <year>")
        sys.exit(1)
    year = int(sys.argv[1])
    url = SEASON_URL_TEMPLATE.format(year=year)
    print(f"Fetching {url}...")
    html = get(url)
    df = parse_fixture(html)
    print(f"Found {len(df)} matches for {year}")
    outpath = f"data/raw/fixture_{year}.csv"
    df.to_csv(outpath, index=False)
    print(f"Saved to {outpath}")

if __name__ == "__main__":
    main()
