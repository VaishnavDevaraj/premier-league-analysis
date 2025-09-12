# Train match prediction model

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime

print("Training match prediction model...")

# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Load data
print("Loading data...")
data_path = os.path.join(DATA_DIR, "fbref_PL_2024-25.csv")
if not os.path.exists(data_path):
    # Try parent directory
    data_path = os.path.join(PROJECT_DIR, "..", "fbref_PL_2024-25.csv")
    if not os.path.exists(data_path):
        print("Error: Dataset not found!")
        exit(1)

df = pd.read_csv(data_path)

# Prepare team statistics
print("Preparing team statistics...")
team_stats = {}
for team in df['Squad'].unique():
    team_players = df[df['Squad'] == team]
    
    # Calculate team strength metrics
    goals = team_players['Gls'].sum()
    xg = team_players['xG'].sum() if 'xG' in team_players.columns else goals * 0.9
    assists = team_players['Ast'].sum() if 'Ast' in team_players.columns else goals * 0.7
    minutes = team_players['Min'].sum() if 'Min' in team_players.columns else team_players.shape[0] * 90
    
    # Use average age as proxy for experience
    avg_age = team_players['Age'].mean() if 'Age' in team_players.columns else 25
    
    # Calculate attack strength from actual goals and xG
    attack_strength = (goals + xg) / 2
    
    # Calculate defense strength from goals against (GA) if available
    # If not available, calculate from defensive stats or estimate
    if 'GA' in team_players.columns:
        goals_against = team_players['GA'].sum()
        defense_strength = 2.0 - (goals_against / max(1, minutes/90))  # Invert so higher is better
    else:
        # Use defensive actions if available
        tackles = team_players['Tkl'].sum() if 'Tkl' in team_players.columns else 0
        interceptions = team_players['Int'].sum() if 'Int' in team_players.columns else 0
        blocks = team_players['Blocks'].sum() if 'Blocks' in team_players.columns else 0
        defensive_actions = tackles + interceptions + blocks
        defense_strength = (defensive_actions / max(1, minutes/90)) * (avg_age / 25)
    
    # Normalize defense strength to reasonable range
    defense_strength = max(0.5, min(1.5, defense_strength))
    
    # Calculate form based on recent results or player form
    player_form = team_players['Gls'].sum() / max(1, team_players.shape[0]) if 'Gls' in team_players.columns else 0.5
    form = 0.5 + (player_form / 2)  # Scale to 0.5-1.0 range
    
    # Save team stats
    team_stats[team] = {
        'attack': attack_strength,
        'defense': defense_strength,
        'experience': avg_age / 30,  # Normalize to 0-1 range
        'form': form  # Calculated form
    }

# Generate synthetic match data
print("Generating synthetic match data...")
matches = []
teams = list(team_stats.keys())

# Create a simple head-to-head history dictionary
h2h_history = {}
for team1 in teams:
    for team2 in teams:
        if team1 != team2:
            # Initialize with random history if not already set
            key = f"{team1}_vs_{team2}"
            h2h_history[key] = {
                'wins': 0,
                'draws': 0,
                'losses': 0
            }

# Generate 150 synthetic matches
for _ in range(150):
    home_team = np.random.choice(teams)
    away_team = np.random.choice([t for t in teams if t != home_team])
    
    # Calculate match features
    home_attack = team_stats[home_team]['attack']
    home_defense = team_stats[home_team]['defense']
    home_experience = team_stats[home_team]['experience']
    home_form = team_stats[home_team]['form']
    
    away_attack = team_stats[away_team]['attack']
    away_defense = team_stats[away_team]['defense']
    away_experience = team_stats[away_team]['experience']
    away_form = team_stats[away_team]['form']
    
    # Home advantage factor
    home_advantage = np.random.uniform(1.1, 1.3)
    
    # Calculate match outcome probabilities
    home_strength = (home_attack * 0.4 + home_defense * 0.3 + home_experience * 0.2 + home_form * 0.1) * home_advantage
    away_strength = away_attack * 0.4 + away_defense * 0.3 + away_experience * 0.2 + away_form * 0.1
    
    # Add head-to-head history as a factor
    h2h_key = f"{home_team}_vs_{away_team}"
    if h2h_key in h2h_history:
        h2h_record = h2h_history[h2h_key]
        total_matches = h2h_record['wins'] + h2h_record['draws'] + h2h_record['losses']
        if total_matches > 0:
            h2h_win_ratio = h2h_record['wins'] / total_matches
            h2h_factor = 0.1  # Weight for h2h history
            home_strength += h2h_win_ratio * h2h_factor
    
    strength_diff = home_strength - away_strength
    
    # Determine match outcome (0 = away win, 1 = draw, 2 = home win)
    if strength_diff > 0.2:
        outcome = 2  # Home win
    elif strength_diff < -0.1:
        outcome = 0  # Away win
    else:
        outcome = 1  # Draw
        
    # Add some randomness to make the model more realistic
    if np.random.random() < 0.15:  # 15% random upsets
        outcome = np.random.choice([0, 1, 2])
    
    # Update head-to-head history
    if outcome == 2:  # Home win
        h2h_history[h2h_key]['wins'] += 1
    elif outcome == 1:  # Draw
        h2h_history[h2h_key]['draws'] += 1
    else:  # Away win
        h2h_history[h2h_key]['losses'] += 1
    
    # Create additional features
    attack_vs_defense_home = home_attack / max(0.1, away_defense)
    attack_vs_defense_away = away_attack / max(0.1, home_defense)
    form_difference = home_form - away_form
    
    # Add match to dataset
    matches.append({
        'home_team': home_team,
        'away_team': away_team,
        'home_attack': home_attack,
        'home_defense': home_defense,
        'home_experience': home_experience,
        'home_form': home_form,
        'away_attack': away_attack,
        'away_defense': away_defense,
        'away_experience': away_experience,
        'away_form': away_form,
        'attack_vs_defense_home': attack_vs_defense_home,
        'attack_vs_defense_away': attack_vs_defense_away,
        'form_difference': form_difference,
        'home_advantage': home_advantage,
        'outcome': outcome
    })

# Create a dataframe from matches
match_df = pd.DataFrame(matches)

# Prepare features and target
X = match_df.drop(['outcome', 'home_team', 'away_team'], axis=1)
y = match_df['outcome']

# Split data
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different models
print("Comparing different machine learning models...")
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Find the best model
best_model = None
best_accuracy = 0
model_accuracies = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    model_accuracies[name] = accuracy
    print(f"{name} accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"Best model: {list(models.keys())[list(models.values()).index(best_model)]} with accuracy: {best_accuracy:.4f}")

# Perform hyperparameter tuning on the best model
print("Performing hyperparameter tuning...")
best_model_name = list(models.keys())[list(models.values()).index(best_model)]

if best_model_name == 'RandomForest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    base_model = RandomForestClassifier(random_state=42)
elif best_model_name == 'GradientBoosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    base_model = GradientBoostingClassifier(random_state=42)
else:  # LogisticRegression
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000]
    }
    base_model = LogisticRegression(random_state=42, multi_class='multinomial')

# Run grid search with cross-validation
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train model with best parameters
print(f"Training final model with best parameters...")
if best_model_name == 'RandomForest':
    model = RandomForestClassifier(random_state=42, **best_params)
elif best_model_name == 'GradientBoosting':
    model = GradientBoostingClassifier(random_state=42, **best_params)
else:  # LogisticRegression
    model = LogisticRegression(random_state=42, multi_class='multinomial', **best_params)

# Train the final model
model.fit(X_train_scaled, y_train)

# Evaluate model
accuracy = model.score(X_test_scaled, y_test)
print(f"Final model trained with accuracy: {accuracy:.4f}")

# Feature importance analysis
if hasattr(model, 'feature_importances_'):
    feature_cols = [col for col in match_df.columns if col not in ['outcome', 'home_team', 'away_team']]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# Create a wrapper class to include preprocessing
class MatchPredictionModel:
    def __init__(self, model, scaler, team_stats):
        self.model = model
        self.scaler = scaler
        self.team_stats = team_stats
        self.classes_ = np.array(['Away Win', 'Draw', 'Home Win'])
    
    def predict(self, X):
        # X should contain home_team and away_team
        features = self._extract_features(X)
        features_scaled = self.scaler.transform(features)
        return self.classes_[self.model.predict(features_scaled)]
    
    def predict_proba(self, X):
        features = self._extract_features(X)
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)
    
    def _extract_features(self, X):
        # Convert input to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Extract home and away team names
        home_team = X['home_team'].iloc[0]
        away_team = X['away_team'].iloc[0]
        
        # Get team stats
        home_stats = self.team_stats.get(home_team, {
            'attack': 0, 'defense': 0, 'experience': 0, 'form': 0
        })
        
        away_stats = self.team_stats.get(away_team, {
            'attack': 0, 'defense': 0, 'experience': 0, 'form': 0
        })
        
        # Home advantage factor (default value)
        home_advantage = 1.2
        
        # Calculate additional features
        attack_vs_defense_home = home_stats['attack'] / max(0.1, away_stats['defense'])
        attack_vs_defense_away = away_stats['attack'] / max(0.1, home_stats['defense'])
        form_difference = home_stats['form'] - away_stats['form']
        
        # Create feature array
        features = np.array([[
            home_stats['attack'],
            home_stats['defense'],
            home_stats['experience'],
            home_stats['form'],
            away_stats['attack'],
            away_stats['defense'],
            away_stats['experience'],
            away_stats['form'],
            attack_vs_defense_home,
            attack_vs_defense_away,
            form_difference,
            home_advantage
        ]])
        
        return features

# Create prediction model wrapper
prediction_model = MatchPredictionModel(model, scaler, team_stats)

# Save model
model_path = os.path.join(MODELS_DIR, 'match_prediction_model.pkl')
print(f"Saving model to {model_path}")
with open(model_path, 'wb') as f:
    pickle.dump(prediction_model, f)

print("Model saved successfully!")
