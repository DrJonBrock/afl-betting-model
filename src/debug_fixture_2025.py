"""
Debug 2025 fixture page structure.
"""
import requests
from bs4 import BeautifulSoup
import re

BASE = "https://afltables.com"
url = "https://afltables.com/afl/stats/2025.html"  # note: stats subfolder? Actually seas/2025.html is main, but maybe tables are under /stats/?
# Wait: the season page might be /afl/stats/2025.html? Let's check: we used seas; but maybe the tables are in /afl/stats/2025.html
# Actually AFL Tables season page is /afl/seas/2025.html, and it contains links to stats pages. The match tables we need are probably on that page itself.
HEADERS = {'User-Agent': 'Mozilla/5.0'}
resp = requests.get(url, headers=HEADERS)
resp.raise_for_status()
html = resp.text
soup = BeautifulSoup(html, 'lxml')
tables = soup.find_all('table')
print(f"Total tables: {len(tables)}")

def find_round_before(table):
    prev = table.find_previous()
    while prev:
        txt = prev.get_text(strip=True)
        m = re.search(r'Round\s+(\d+)', txt, re.I)
        if m:
            return int(m.group(1))
        prev = prev.find_previous()
    return None

# Print first 10 tables' info
for i, table in enumerate(tables[:20]):
    rnd = find_round_before(table)
    team_links = table.find_all('a', href=re.compile(r'/teams/'))
    venue_link = table.find('a', href=re.compile(r'/venues/'))
    print(f"Table {i}: round={rnd}, team_links={len(team_links)}, venue_link={'yes' if venue_link else 'no'}")
    if rnd is None:
        # Print a snippet of the table's text to see what it is
        txt = table.get_text(strip=True)[:100]
        print(f"   preview: {txt}")
