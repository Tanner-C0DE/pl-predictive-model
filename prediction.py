import pandas as pd

# Load and clean data
df = pd.read_csv('25-26 EPL data.csv')
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam']).sort_values('Date')

def get_detailed_stats(team_name, until_date, data_df):
    history = data_df[((data_df['HomeTeam'] == team_name) | (data_df['AwayTeam'] == team_name)) & 
                       (data_df['Date'] < until_date)].copy()
    
    if history.empty:
        return {'ppg': 1.0, 'form': 1.0, 'rest': 7, 'stamina': 0, 'cards': 0, 'shot_eff': 0.3}

    # --- OPTIMIZATION: Vectorized Points Calculation ---
    home_wins = (history['HomeTeam'] == team_name) & (history['FTR'] == 'H')
    away_wins = (history['AwayTeam'] == team_name) & (history['FTR'] == 'A')
    draws = (history['FTR'] == 'D')
    pts = (home_wins.sum() + away_wins.sum()) * 3 + draws.sum()
    ppg = pts / len(history)

    # Recent Trends (Last 5)
    last_5 = history.tail(5)
    f_pts, leads_lost, t_shots, t_on_target = 0, 0, 0, 0
    for _, m in last_5.iterrows():
        is_h = m['HomeTeam'] == team_name
        side = 'H' if is_h else 'A'
        if m['FTR'] == side: f_pts += 3
        elif m['FTR'] == 'D': f_pts += 1
        if m['HTR'] == side and m['FTR'] != side: leads_lost += 1
        t_shots += m['HS'] if is_h else m['AS']
        t_on_target += m['HST'] if is_h else m['AST']

    rest = min((until_date - history['Date'].max()).days, 10)
    cards = sum([(m['HY'] + m['HR']*2) if m['HomeTeam'] == team_name else (m['AY'] + m['AR']*2) for _, m in history.tail(3).iterrows()])

    return {'ppg': ppg, 'form': f_pts/5, 'rest': rest, 'stamina': leads_lost, 'cards': cards, 'shot_eff': t_on_target/t_shots if t_shots > 0 else 0.3}

def predict_match(home, away, date_str, data_df, silent=False):
    target_date = pd.to_datetime(date_str, format='%m/%d/%y')
    h = get_detailed_stats(home, target_date, data_df)
    a = get_detailed_stats(away, target_date, data_df)

    W = {'STR': 0.20, 'FORM': 0.40, 'GAP': 0.40, 'REST': 0.10, 'SHOT': 0.05, 'STAM': 0.05, 'DISC': 0.05}
    DRAW_THRESH = 0.20 
    
    def get_score(s, opp, is_home):
        contrib = {
            'Strength': round(s['ppg'] * W['STR'], 3),
            'Form': round(s['form'] * W['FORM'], 3),
            'Matchup Gap': round((s['ppg'] - opp['ppg']) * W['GAP'], 3),
            'Rest': round(s['rest'] * W['REST'], 3),
            'Efficiency': round((s['shot_eff'] * 10) * W['SHOT'], 3),
            'Penalty (Stam/Disc)': round(-(s['stamina'] * W['STAM'] + s['cards'] * W['DISC']), 3),
            'Home Adv': 0.45 if is_home else 0.0
        }
        return contrib, sum(contrib.values())

    h_c, h_total = get_score(h, a, True)
    a_c, a_total = get_score(a, h, False)
    diff = h_total - a_total

    if diff > DRAW_THRESH: prediction = 'H'
    elif diff < -DRAW_THRESH: prediction = 'A'
    else: prediction = 'D'

    if not silent:
        print(f"\n=== MATCH ANALYSIS: {home} vs {away} ({date_str}) ===")
        print(f"{'FACTOR':<20} | {home:<15} | {away:<15}")
        print("-" * 55)
        for key in h_c.keys():
            print(f"{key:<20} | {h_c[key]:<15} | {a_c[key]:<15}")
        print("-" * 55)
        print(f"{'FINAL MODEL SCORE':<20} | {h_total:<15.3f} | {a_total:<15.3f}")
        winner_text = home if prediction == 'H' else (away if prediction == 'A' else "Draw")
        print(f"\n>>> PREDICTION: {winner_text} Wins! <<<\n")

    return prediction

# --- ACCURACY CHECK (BACKTESTER) ---
print("Running historical backtest...")
test_set = df[df.index > 40] 
correct = sum([1 for _, r in test_set.iterrows() if predict_match(r['HomeTeam'], r['AwayTeam'], r['Date'].strftime('%m/%d/%y'), df, silent=True) == r['FTR']])
print(f"Model Accuracy Check: {(correct / len(test_set)) * 100:.2f}%")

# --- CUSTOM PREDICTION INPUT ---
home_team = input("Enter home team: ")
away_team = input("Enter away team: ")
date_input = input("Enter date (MM/DD/YY): ")
predict_match(home_team, away_team, date_input, df)

# Teams in the 2025-2026 EPL Season:
# Liverpool
# Bournemouth
# Aston Villa
# Arsenal
# Newcastle
# Nott'm Forest
# Man City
# Man United
# Chelsea
# Tottenham
# Everton
# Brentford
# Crystal Palace
# Fulham
# West Ham
# Wolves
# Leeds
# Brighton
# Burnley
# Sunderland