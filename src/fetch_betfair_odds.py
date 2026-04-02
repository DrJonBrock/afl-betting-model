"""
Fetch AFL 2026 total score lines from Betfair Exchange API.
Requires environment variables:
  BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY
Optional: BETFAIR_CERT_PATH, BETFAIR_KEY_PATH for certificate auth.
Outputs CSV with columns: year,round,home_team,away_team,total_score_close
"""
import os
import re
import sys
import datetime
import argparse
import csv
from typing import Dict, Tuple, List, Optional

import pandas as pd
from betfairlightweight import APIClient, filters

# Team name mapping from Betfair to our canonical names
BETFAIR_TO_OUR = {
    "Sydney Swans": "Sydney",
    "Collingwood Magpies": "Collingwood",
    "West Coast Eagles": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
    "Greater Western Sydney Giants": "GWS",
    "Gold Coast SUNS": "Gold Coast",
    "North Melbourne Kangaroos": "North Melbourne",
    "Geelong Cats": "Geelong",
    "Essendon Bombers": "Essendon",
    "Carlton Blues": "Carlton",
    "Melbourne Demons": "Mel Melbourne",  # careful: "Melbourne Demons" -> "Melbourne"
    "Hawthorn Hawks": "Hawthorn",
    "Richmond Tigers": "Richmond",
    "St Kilda Saints": "St Kilda",
    "Adelaide Crows": "Adelaide",
    "Brisbane Lions": "Brisbane Lions",
    "Fremantle Dockers": "Fremantle",
    "Port Adelaide Power": "Port Adelaide",
}

# Reverse mapping for sanity check (optional)
OUR_TEAMS = [
    "Adelaide", "Brisbane Lions", "Carlton", "Collingwood", "Essendon",
    "Fremantle", "Geelong", "Gold Coast", "GWS", "Hawthorn",
    "Melbourne", "North Melbourne", "Port Adelaide", "Richmond",
    "St Kilda", "Sydney", "West Coast", "Western Bulldogs"
]

# Round 1 start date (Thursday) per AFL Tables 2026
ROUND1_START = datetime.date(2026, 3, 5)  # 2026-03-05

def round_from_date(event_date: datetime.date) -> int:
    """Compute round number given event date based on weekly cycle starting R1 start."""
    delta = (event_date - ROUND1_START).days
    if delta < 0:
        return 0
    return delta // 7 + 1

def normalize_team(name: str) -> str:
    """Map Betfair team name to our canonical name."""
    # Clean whitespace
    name = name.strip()
    # Direct mapping
    if name in BETFAIR_TO_OUR:
        return BETFAIR_TO_OUR[name]
    # Some names may be without nickname, e.g., "Collingwood" might appear as "Collingwood" already.
    # If name is in OUR_TEAMS directly, return as is.
    if name in OUR_TEAMS:
        return name
    # Try partial match: if any known team name is a substring of the Betfair name or vice versa
    for our in OUR_TEAMS:
        if our.lower() in name.lower() or name.lower() in our.lower():
            return our
    # Fallback: return original but log warning later
    return name

def build_home_away_mapping(stats_path: str) -> Dict[Tuple[int, Tuple[str, str]], Tuple[str, str]]:
    """
    Build a mapping from (round, sorted_team_pair) -> (home_team, away_team)
    using player_game_stats.csv which contains is_home flags.
    """
    mapping = {}
    df = pd.read_csv(stats_path)
    # Filter 2026
    df = df[df['year'] == 2026]
    for row in df.itertuples():
        r = int(row.round)
        team = row.team
        opp = row.opponent
        is_home = bool(row.is_home)
        key = (r, tuple(sorted([team, opp])))
        if is_home:
            mapping[key] = (team, opp)
        else:
            # We'll fill when we see the home row; if we see away first, skip unless we already have home then flip?
            # Actually we only store when we see home. So we ignore away rows.
            pass
    return mapping

def parse_event_name(event_name: str) -> Tuple[str, str]:
    """Extract two team names from Betfair event name like 'Sydney Swans v Collingwood Magpies'."""
    # Split on 'v' or 'vs' with spaces
    parts = re.split(r'\s+v\s+|\s+vs\s+', event_name, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # Sometimes it might be 'Team A / Team B' or other separators
    # Fallback: try splitting on ' - ' or just take first two quoted parts
    raise ValueError(f"Cannot parse event name: {event_name}")

def main():
    parser = argparse.ArgumentParser(description="Fetch 2026 AFL total odds from Betfair")
    parser.add_argument('--rounds', default='1-4', help='Round range to fetch, e.g., 1-4 or 1-: for all')
    parser.add_argument('--output', default='data/raw/betfair_odds_2026.csv', help='Output CSV path')
    parser.add_argument('--stats', default='data/processed/player_game_stats.csv', help='Path to player_game_stats.csv for home/away mapping')
    args = parser.parse_args()

    # Parse rounds
    if '-' in args.rounds:
        start_r, end_r = map(int, args.rounds.split('-'))
        target_rounds = list(range(start_r, end_r+1))
    else:
        target_rounds = [int(args.rounds)]

    # Build home/away mapping from existing stats (for rounds 1-4 present)
    home_away_map = build_home_away_mapping(args.stats)
    print(f"Loaded home/away mapping for {len(home_away_map)} matches from {args.stats}")

    # Get credentials from environment
    username = os.getenv('BETFAIR_USERNAME')
    password = os.getenv('BETFAIR_PASSWORD')
    app_key = os.getenv('BETFAIR_APP_KEY')
    if not all([username, password, app_key]):
        print("ERROR: Please set BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY environment variables.")
        sys.exit(1)

    cert_path = os.getenv('BETFAIR_CERT_PATH')
    key_path = os.getenv('BETFAIR_KEY_PATH')
    certs = None
    if cert_path and key_path:
        certs = (cert_path, key_path)

    # Login
    client = APIClient(username=username, password=password, app_key=app_key, certs=certs)
    client.login()
    print("Logged in to Betfair")

    # Find AFL competition ID
    comps = client.betting.list_competitions()
    afl_comp = None
    for comp in comps:
        if 'AFL' in comp.name or 'Australian Football League' in comp.name:
            afl_comp = comp
            break
    if not afl_comp:
        print("ERROR: Could not find AFL competition. Listing competitions:")
        for comp in comps:
            print(f"  {comp.name} (ID: {comp.id})")
        sys.exit(1)
    comp_id = afl_comp.id
    print(f"AFL competition ID: {comp_id}")

    # Determine date range for target rounds (or entire season up to today)
    # We'll fetch events from season start to today, then filter by round.
    date_from = datetime.datetime(2026, 3, 1).strftime('%Y-%m-%d')
    date_to = datetime.datetime.today().strftime('%Y-%m-%d')  # up to today
    # But we can also just fetch up to latest round
    event_filter = filters.lightweight_event_filter(competition_ids=[comp_id], date_from=date_from, date_to=date_to)
    events = client.betting.list_events(filter=event_filter)
    print(f"Fetched {len(events)} events from Betfair for competition {comp_id} between {date_from} and {date_to}")

    # Process events
    odds_rows = []
    for e in events:
        try:
            team_a, team_b = parse_event_name(e.name)
        except ValueError:
            continue
        # Normalize names
        team_a = normalize_team(team_a)
        team_b = normalize_team(team_b)
        # If either unknown, warn
        if team_a not in OUR_TEAMS or team_b not in OUR_TEAMS:
            # Try alternative mapping: maybe team names are swapped; try each individually with fuzzy
            # For now, attempt to map using substring
            if team_a not in OUR_TEAMS:
                for our in OUR_TEAMS:
                    if our.lower() in team_a.lower():
                        team_a = our
                        break
            if team_b not in OUR_TEAMS:
                for our in OUR_TEAMS:
                    if our.lower() in team_b.lower():
                        team_b = our
                        break
            if team_a not in OUR_TEAMS or team_b not in OUR_TEAMS:
                # print(f"Warning: unknown team(s) in event {e.name}: {team_a}, {team_b}")
                continue
        # Compute round from date
        event_date = e.date.date()
        round_num = round_from_date(event_date)
        if round_num not in target_rounds:
            continue
        # Determine home and away based on our mapping
        key = (round_num, tuple(sorted([team_a, team_b])))
        if key not in home_away_map:
            # print(f"Warning: match not found in player_game_stats for round {round_num}, teams {team_a}/{team_b}")
            continue
        home_team, away_team = home_away_map[key]
        # Fetch markets to get total line
        try:
            mcat = client.betting.list_market_catalogue(
                filter=filters.lightweight_market_filter(
                    event_ids=[e.id],
                    market_type_codes=['OVER_UNDER'],
                    market_betting_type='ODDS',
                    turn_in_play=False
                ),
                max_results=50
            )
        except Exception as ex:
            print(f"Error fetching markets for event {e.id}: {ex}")
            continue
        total_line = None
        for market in mcat:
            mname = market.market_name
            if 'Total Points' in mname and 'Over/Under' in mname:
                m = re.search(r'Over/Under ([\d.]+)', mname)
                if m:
                    total_line = float(m.group(1))
                    break
        if total_line is None:
            continue
        # Append row
        odds_rows.append({
            'year': 2026,
            'round': round_num,
            'home_team': home_team,
            'away_team': away_team,
            'total_score_close': total_line
        })
    # Write CSV
    if not odds_rows:
        print("No odds rows collected. Check filters.")
        sys.exit(1)
    out_df = pd.DataFrame(odds_rows, columns=['year','round','home_team','away_team','total_score_close'])
    out_df.to_csv(args.output, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Wrote {len(out_df)} rows to {args.output}")
    # Summary
    print("Rounds covered:", sorted(out_df['round'].unique()))
    print(out_df.head().to_string(index=False))

if __name__ == "__main__":
    main()
