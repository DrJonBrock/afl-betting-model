"""
Preprocess raw AFL data into clean, analysis-ready tables.
For MVP, we generate synthetic player-game stats if real data not available.
"""
import pandas as pd
import numpy as np
import os

PROCESSED_DIR = "data/processed"

def generate_synthetic_data(n_players=300, n_games_per_season=200, n_seasons=5):
    """Generate plausible AFL player-game statistics with realistic distributions."""
    np.random.seed(42)
    records = []
    for season in range(2020, 2020 + n_seasons):
        teams = [
            "Adelaide Crows", "Brisbane Lions", "Carlton", "Collingwood", "Essendon",
            "Fremantle", "Geelong", "Gold Coast Suns", "GWS Giants", "Hawthorn",
            "Melbourne", "North Melbourne", "Port Adelaide", "Richmond", "St Kilda",
            "Sydney Swans", "West Coast Eagles", "Western Bulldogs"
        ]
        venues = [
            "MCG", "Marvel Stadium", "Adelaide Oval", "Gabba", "SCG", "GMHBA Stadium",
            "Metricon Stadium", "Blundstone Arena", "Manuka Oval", "TIO Stadium"
        ]
        # Team offensive/defensive strengths (random per season)
        team_offense = {t: np.random.uniform(0.9, 1.1) for t in teams}
        team_defense = {t: np.random.uniform(0.9, 1.1) for t in teams}
        # venue scoring factors
        venue_factor = {v: np.random.uniform(0.95, 1.05) for v in venues}
        # Generate player pool with base skill levels and positions
        players = []
        for i in range(n_players):
            base_disposals = np.random.normal(20, 5)
            base_kicks = base_disposals * np.random.uniform(0.5, 0.7)
            base_handballs = base_disposals - base_kicks
            base_marks = np.random.normal(5, 2)
            players.append({
                'player': f"Player_{i+1}",
                'team': np.random.choice(teams),
                'base_disposals': base_disposals,
                'base_kicks': base_kicks,
                'base_handballs': base_handballs,
                'base_marks': base_marks,
                'position': np.random.choice(['Midfielder', 'Forward', 'Defender', 'Ruck']),
            })
        players_df = pd.DataFrame(players)
        # Schedule games: round-robin style (each team plays each other roughly once)
        n_games = n_games_per_season
        game_records = []
        for g in range(n_games):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            venue = np.random.choice(venues)
            date = f"Round {g%23+1}"
            game_records.append({
                'year': season,
                'game_id': f"{season}_{g}",
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                'venue': venue,
            })
        games_df = pd.DataFrame(game_records)
        # Assign players to games based on team; include opponent adjustments
        player_game_records = []
        for _, game in games_df.iterrows():
            for team in [game['home_team'], game['away_team']]:
                team_players = players_df[players_df['team'] == team]
                for _, player in team_players.iterrows():
                    # Base stats from player
                    base_disp = player['base_disposals']
                    base_kicks = player['base_kicks']
                    base_marks = player['base_marks']
                    # Game-to-game variation (less variance now, sigma ~10%)
                    disp_mult = np.random.normal(1.0, 0.1)
                    kicks_mult = np.random.normal(1.0, 0.1)
                    marks_mult = np.random.normal(1.0, 0.1)
                    # Home advantage
                    if team == game['home_team']:
                        disp_mult *= 1.03
                        kicks_mult *= 1.03
                    # Venue factor
                    disp_mult *= venue_factor[game['venue']]
                    # Opponent defensive strength: if opponent is strong defense, reduce disposals
                    opp_def = team_defense[game['away_team'] if team == game['home_team'] else game['home_team']]
                    disp_mult *= opp_def  # stronger defense (higher factor) actually increases? Let's say opp_def factor: >1 means weak defense; simplify by inverting: use (2 - opp_def)
                    disp_mult *= (2 - opp_def)  # if opp_def=1.1 (strong), reduce; if 0.9 (weak), increase
                    # Team offense context
                    disp_mult *= team_offense[team]
                    # Compute final stats
                    disposals = max(0, int(base_disp * disp_mult * np.random.uniform(0.95, 1.05)))
                    kicks = max(0, int(base_kicks * kicks_mult * np.random.uniform(0.95, 1.05)))
                    # Ensure handballs = disposals - kicks, non-negative
                    handballs = max(0, disposals - kicks)
                    marks = max(0, int(base_marks * marks_mult * np.random.uniform(0.95, 1.05)))
                    player_game_records.append({
                        'year': season,
                        'game_id': game['game_id'],
                        'player': player['player'],
                        'team': team,
                        'opponent_team': game['away_team'] if team == game['home_team'] else game['home_team'],
                        'venue': game['venue'],
                        'is_home': 1 if team == game['home_team'] else 0,
                        'kicks': kicks,
                        'handballs': handballs,
                        'disposals': disposals,
                        'marks': marks,
                    })
        player_game_df = pd.DataFrame(player_game_records)
        player_game_df['game_order'] = player_game_df.groupby(['player', 'year']).cumcount() + 1
        records.append(player_game_df)
    all_pg = pd.concat(records, ignore_index=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    all_pg.to_csv(os.path.join(PROCESSED_DIR, "player_game_stats.csv"), index=False)
    all_games = []
    for season in range(2020, 2020 + n_seasons):
        n_games = n_games_per_season
        for g in range(n_games):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            venue = np.random.choice(venues)
            all_games.append({
                'year': season,
                'game_id': f"{season}_{g}",
                'date': f"Round {g%23+1}",
                'home_team': home_team,
                'away_team': away_team,
                'venue': venue,
            })
    games_df = pd.DataFrame(all_games).drop_duplicates('game_id')
    games_df.to_csv(os.path.join(PROCESSED_DIR, "games_clean.csv"), index=False)
    return all_pg, games_df

def main():
    print("Generating synthetic AFL data for MVP prototyping...")
    pg, games = generate_synthetic_data()
    print(f"Generated {len(pg)} player-game rows and {len(games)} games.")
    print(f"Saved to data/processed/")

if __name__ == "__main__":
    main()