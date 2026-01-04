# pl-predictive-model

Premier League Heuristic Scoring Engine

This project implements a custom-weighted "Grid Model" to predict English Premier League match outcomes. Unlike "black-box" machine learning models, this engine uses explicit mathematical weights assigned to key performance indicators, providing total transparency into why a specific team is favored.

The Logic:
The model calculates a Match Strength Score for each team based on:

Points Per Game (PPG): Base strength of the season.

Momentum (Form): A rolling average of the last 5 matches.

Fatigue (Rest Days): Penalties applied for short turnaround times between games.

Shot Efficiency: A calculated ratio of Shots on Target vs. Total Shots.

Disciplinary Record: Negative weights applied for yellow and red card accumulation.

Home Advantage: A flat +0.45 boost to the home side to account for historical variance.

Key Features:
Custom Dashboard: Outputs a side-by-side "Match Analysis" grid comparing the two teams across all factors.

Built-in Backtester: Includes a script to run the model against completed fixtures to calculate historical accuracy.

Zero Warm-up: Uses vectorized pandas operations to calculate stats on the fly.

Sample Output:

Model Accuracy Check: 50.34%
Enter home team: Aston Villa
Enter away team: Nott'm Forest
Enter date (MM/DD/YY): 1/3/26

=== MATCH ANALYSIS: Aston Villa vs Nott'm Forest (1/3/26) ===
FACTOR               | Aston Villa     | Nott'm Forest
-------------------------------------------------------
Strength             | 0.411           | 0.189
Form                 | 0.96            | 0.24
Matchup Gap          | 0.442           | -0.442
Rest                 | 0.4             | 0.4
Efficiency           | 0.223           | 0.153
Penalty (Stam/Disc)  | -0.35           | -0.3
Home Adv             | 0.45            | 0.0
-------------------------------------------------------
FINAL MODEL SCORE    | 2.536           | 0.240

>>> PREDICTION: Aston Villa Wins! <<<

Data was collected from https://www.football-data.co.uk/englandm.php
