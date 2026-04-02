"""
Robust AFL fixture scraper for any year (2020-2026).
Traverss document in order, tracking current round from any heading, then assigns that round to subsequent match tables.
"""
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import time
import random
import sys
from urllib.parse import urljoin

BASE = "https://afltables.com"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get(url):
    time.sleep(random.uniform(0.3,0.8))
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def parse_fixture(html):
    """Return DataFrame with columns: round, away, home, venue."""
    soup = BeautifulSoup(html, 'lxml')
    # Get all elements in document order: we'll iterate over tables and also headings
    # Use .descendants? Simpler: collect all tags we care about in order with their positions using .find_all with recursion
    # We'll gather a list of (tag, text) in order, then process sequentially
    elements = list(soup.find_all(['table','b','strong','h2','h3','center']))
    # But we need to also capture text that might be inside other tags like <font>. Instead, we can traverse using .find_all with recursive=True, but that's not in order by tag types.
    # Alternative: iterate over all tags but that's too many. Simpler: iterate over tables but before each table, look at its preceding siblings for a round heading.
    fixtures = []
    for table in soup.find_all('table'):
        # Find preceding heading by walking back in the DOM tree
        round_num = None
        prev = table.find_previous()
        while prev:
            txt = prev.get_text(strip=True)
            m = re.search(r'Round\s+(\d+)', txt, re.I)
            if m:
                round_num = int(m.group(1))
                break
            prev = prev.find_previous()
        if round_num is None:
            continue
        # Check if this table contains match info (team links + venue link)
        team_links = table.find_all('a', href=re.compile(r'/teams/'))
        if len(team_links) < 2:
            continue
        venue_link = table.find('a', href=re.compile(r'/venues/'))
        if not venue_link:
            continue
        away = team_links[0].get_text(strip=True)
        home = team_links[1].get_text(strip=True)
        venue = venue_link.get_text(strip=True)
        fixtures.append((round_num, away, home, venue))
    df = pd.DataFrame(fixtures, columns=['round','away','home','venue'])
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python scrape_fixture_robust.py <year>")
        sys.exit(1)
    year = int(sys.argv[1])
    url = f"{BASE}/afl/seas/{year}.html"
    print(f"Fetching {url}...")
    html = get(url)
    df = parse_fixture(html)
    print(f"Found {len(df)} matches for {year}")
    if len(df) > 0:
        print("Round distribution:")
        print(df['round'].value_counts().sort_index().head(10))
        print("Sample teams:", df['away'].unique()[:5])
    out_fix = f"data/raw/fixture_{year}.csv"
    df.to_csv(out_fix, index=False)
    print(f"Saved to {out_fix}")
    # Also mapping
    mapping = pd.DataFrame([
        {'round': r, 'team': away, 'opponent': home, 'venue': venue, 'is_home': 0}
        for r, away, home, venue in zip(df['round'], df['away'], df['home'], df['venue'])
    ] + [
        {'round': r, 'team': home, 'opponent': away, 'venue': venue, 'is_home': 1}
        for r, away, home, venue in zip(df['round'], df['away'], df['home'], df['venue'])
    ])
    out_map = f"data/raw/fixture_mapping_{year}.csv"
    mapping.to_csv(out_map, index=False)
    print(f"Saved mapping to {out_map}")

if __name__ == "__main__":
    main()
