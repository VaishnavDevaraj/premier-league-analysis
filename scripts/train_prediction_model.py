# Train match prediction model

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    
    # Calculate attack and defense strength
    attack_strength = (goals + xg) / 2
    defense_strength = np.random.uniform(0.5, 1.5) * (avg_age / 25)  # Simulate defense strength
    
    # Save team stats
    team_stats[team] = {
        'attack': attack_strength,
        'defense': defense_strength,
        'experience': avg_age / 30,  # Normalize to 0-1 range
        'form': np.random.uniform(0.5, 1.0)  # Random form between 0.5-1.0
    }

# Generate synthetic match data
print("Generating synthetic match data...")
matches = []
teams = list(team_stats.keys())

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

# Train model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
accuracy = model.score(X_test_scaled, y_test)
print(f"Model trained with accuracy: {accuracy}")

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
        
        # Create feature array
        features = np.array([[
            home_stats['attack'],
            home_stats['defense'],
            home_stats['experience'],
            home_stats['form'],
            away_stats['attack'],
            away_stats['defense'],
            away_stats['experience'],
            away_stats['form']
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
