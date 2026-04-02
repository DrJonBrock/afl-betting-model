"""
Debug: print tables that have team links and their preceding round.
"""
import requests
from bs4 import BeautifulSoup
import re

url = "https://afltables.com/afl/seas/2025.html"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
html = requests.get(url, headers=HEADERS).text
soup = BeautifulSoup(html, 'lxml')
tables = soup.find_all('table')
print(f"Total tables: {len(tables)}")

def get_round_before(table):
    prev = table.find_previous()
    while prev:
        txt = prev.get_text(strip=True)
        m = re.search(r'Round\s+(\d+)', txt, re.I)
        if m:
            return int(m.group(1))
        prev = prev.find_previous()
    return None

count = 0
for table in tables[:100]:
    rnd = get_round_before(table)
    team_links = table.find_all('a', href=re.compile(r'/teams/'))
    venue_link = table.find('a', href=re.compile(r'/venues/'))
    if team_links and venue_link and rnd:
        away = team_links[0].get_text(strip=True)
        home = team_links[1].get_text(strip=True) if len(team_links)>1 else ''
        venue = venue_link.get_text(strip=True)
        print(f"Round {rnd}: {away} vs {home} at {venue}")
        count += 1
        if count >= 10:
            break

if count == 0:
    print("No qualifying tables found. Printing some table snippets instead:")
    for i, table in enumerate(tables[:10]):
        team_links = table.find_all('a', href=re.compile(r'/teams/'))
        venue_link = table.find('a', href=re.compile(r'/venues/'))
        txt = table.get_text(strip=True)[:200]
        print(f"\nTable {i}: team_links={len(team_links)}, venue_link={'yes' if venue_link else 'no'}")
        print("Snippet:", txt[:200])
