"""
Debug 2025 with corrected patterns.
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
for table in tables[:200]:
    rnd = get_round_before(table)
    team_links = table.find_all('a', href=re.compile(r'/team/'))
    venue_link = table.find('a', href=re.compile(r'/venues/'))
    if team_links and venue_link and rnd:
        # Expect at least 2 team links (away, home)
        if len(team_links) >= 2:
            away = team_links[0].get_text(strip=True)
            home = team_links[1].get_text(strip=True)
            venue = venue_link.get_text(strip=True)
            print(f"Round {rnd}: {away} vs {home} at {venue}")
            count += 1
            if count >= 20:
                break

if count == 0:
    print("No qualifying tables found. Sample team links:")
    for table in tables[:5]:
        team_links = table.find_all('a', href=re.compile(r'/team/'))
        print(f"  table with {len(team_links)} team links")
        for a in team_links[:2]:
            print(f"    {a.get('href')}: {a.get_text(strip=True)}")
    print("Sample venue links:")
    for table in tables[:5]:
        venue_links = table.find_all('a', href=re.compile(r'/venues/'))
        for v in venue_links[:2]:
            print(f"    {v.get('href')}: {v.get_text(strip=True)}")
