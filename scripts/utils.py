"""
Utility functions for Premier League data analysis.
This module contains helper functions for data preprocessing, feature engineering,
and other common operations used throughout the project.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Union, Optional, Tuple

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values, correcting data types,
    and removing duplicates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The raw dataframe to clean
        
    Returns:
    --------
    pd.DataFrame
        The cleaned dataframe
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values (implementation depends on actual data)
    # Example:
    # numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    # categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # for col in numerical_cols:
    #     df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # for col in categorical_cols:
    #     df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features based on existing data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
        
    Returns:
    --------
    pd.DataFrame
        The dataframe with new engineered features
    """
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # Example feature engineering (implementation depends on actual data)
    # Example:
    # # Create goal difference
    # if 'GoalsFor' in df_features.columns and 'GoalsAgainst' in df_features.columns:
    #     df_features['GoalDifference'] = df_features['GoalsFor'] - df_features['GoalsAgainst']
    
    # # Create win ratio
    # if 'Wins' in df_features.columns and 'Matches' in df_features.columns:
    #     df_features['WinRatio'] = df_features['Wins'] / df_features['Matches']
    
    # # Create points per game
    # if 'Points' in df_features.columns and 'Matches' in df_features.columns:
    #     df_features['PointsPerGame'] = df_features['Points'] / df_features['Matches']
    
    return df_features

def calculate_form(results: List[str], window: int = 5) -> np.ndarray:
    """
    Calculate form based on recent match results.
    
    Parameters:
    -----------
    results : List[str]
        List of match results (e.g., ['W', 'D', 'L', 'W', 'W'])
    window : int, optional (default=5)
        Number of matches to consider for form calculation
        
    Returns:
    --------
    np.ndarray
        Array of form values
    """
    # Map results to points: win=3, draw=1, loss=0
    result_map = {'W': 3, 'D': 1, 'L': 0}
    points = [result_map.get(result, 0) for result in results]
    
    # Calculate rolling sum of points (form)
    form_series = pd.Series(points).rolling(window=window, min_periods=1).sum()
    
    return form_series.values

def create_match_features(df: pd.DataFrame, team_col: str, opponent_col: str, 
                         date_col: str) -> pd.DataFrame:
    """
    Create features for match prediction based on historical performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe with match data
    team_col : str
        Name of the column containing team names
    opponent_col : str
        Name of the column containing opponent team names
    date_col : str
        Name of the column containing match dates
        
    Returns:
    --------
    pd.DataFrame
        The dataframe with new match prediction features
    """
    # Create a copy to avoid modifying the original
    df_matches = df.copy()
    
    # Convert date column to datetime if needed
    if date_col in df_matches.columns:
        df_matches[date_col] = pd.to_datetime(df_matches[date_col])
    
    # Sort by date
    df_matches = df_matches.sort_values(date_col)
    
    # Example feature creation (implementation depends on actual data)
    # Example:
    # # Create team-specific features
    # teams = df_matches[team_col].unique()
    # 
    # # Initialize new feature columns
    # df_matches['recent_form'] = 0
    # df_matches['avg_goals_scored_last_5'] = 0
    # df_matches['avg_goals_conceded_last_5'] = 0
    # 
    # # Calculate features for each team
    # for team in teams:
    #     team_matches = df_matches[df_matches[team_col] == team].copy()
    #     
    #     # Calculate form based on last 5 matches
    #     team_matches['recent_form'] = calculate_form(team_matches['result'].tolist(), window=5)
    #     
    #     # Calculate average goals scored/conceded in last 5 matches
    #     team_matches['avg_goals_scored_last_5'] = team_matches['goals_scored'].rolling(window=5, min_periods=1).mean()
    #     team_matches['avg_goals_conceded_last_5'] = team_matches['goals_conceded'].rolling(window=5, min_periods=1).mean()
    #     
    #     # Update the main dataframe
    #     df_matches.loc[team_matches.index, 'recent_form'] = team_matches['recent_form']
    #     df_matches.loc[team_matches.index, 'avg_goals_scored_last_5'] = team_matches['avg_goals_scored_last_5']
    #     df_matches.loc[team_matches.index, 'avg_goals_conceded_last_5'] = team_matches['avg_goals_conceded_last_5']
    
    return df_matches

def calculate_league_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the league table based on match results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe with match data
        
    Returns:
    --------
    pd.DataFrame
        League table as a dataframe
    """
    # Example implementation (depends on actual data structure)
    # Example:
    # # Initialize empty table
    # teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    # table = pd.DataFrame({
    #     'Team': teams,
    #     'Played': 0,
    #     'Won': 0,
    #     'Drawn': 0,
    #     'Lost': 0,
    #     'GF': 0,  # Goals For
    #     'GA': 0,  # Goals Against
    #     'GD': 0,  # Goal Difference
    #     'Points': 0
    # })
    # 
    # # Iterate through matches and update table
    # for _, match in df.iterrows():
    #     home_team = match['home_team']
    #     away_team = match['away_team']
    #     home_goals = match['home_goals']
    #     away_goals = match['away_goals']
    #     
    #     # Update matches played
    #     table.loc[table['Team'] == home_team, 'Played'] += 1
    #     table.loc[table['Team'] == away_team, 'Played'] += 1
    #     
    #     # Update goals
    #     table.loc[table['Team'] == home_team, 'GF'] += home_goals
    #     table.loc[table['Team'] == home_team, 'GA'] += away_goals
    #     table.loc[table['Team'] == away_team, 'GF'] += away_goals
    #     table.loc[table['Team'] == away_team, 'GA'] += home_goals
    #     
    #     # Update results and points
    #     if home_goals > away_goals:  # Home win
    #         table.loc[table['Team'] == home_team, 'Won'] += 1
    #         table.loc[table['Team'] == home_team, 'Points'] += 3
    #         table.loc[table['Team'] == away_team, 'Lost'] += 1
    #     elif home_goals < away_goals:  # Away win
    #         table.loc[table['Team'] == away_team, 'Won'] += 1
    #         table.loc[table['Team'] == away_team, 'Points'] += 3
    #         table.loc[table['Team'] == home_team, 'Lost'] += 1
    #     else:  # Draw
    #         table.loc[table['Team'] == home_team, 'Drawn'] += 1
    #         table.loc[table['Team'] == home_team, 'Points'] += 1
    #         table.loc[table['Team'] == away_team, 'Drawn'] += 1
    #         table.loc[table['Team'] == away_team, 'Points'] += 1
    # 
    # # Calculate goal difference
    # table['GD'] = table['GF'] - table['GA']
    # 
    # # Sort table by points, then goal difference, then goals for
    # table = table.sort_values(['Points', 'GD', 'GF'], ascending=[False, False, False])
    # 
    # # Reset index and add position column
    # table = table.reset_index(drop=True)
    # table.index = table.index + 1  # Start from position 1
    # table = table.rename_axis('Position').reset_index()
    
    # Placeholder return until actual implementation
    return pd.DataFrame()

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate various metrics to evaluate model performance.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    y_proba : np.ndarray, optional
        Predicted probabilities for each class
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, roc_auc_score
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate ROC AUC if probabilities are provided
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception:
            # ROC AUC might fail for certain cases
            metrics['roc_auc'] = None
    
    return metrics

def save_model(model, filename: str, models_dir: str = "../models") -> str:
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : object
        The trained model to save
    filename : str
        Base filename for the model
    models_dir : str, optional
        Directory to save the model in
        
    Returns:
    --------
    str
        Full path to the saved model
    """
    import pickle
    import os
    from datetime import datetime
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.pkl"
    full_path = os.path.join(models_dir, full_filename)
    
    # Save the model
    with open(full_path, 'wb') as f:
        pickle.dump(model, f)
    
    return full_path

def load_model(filename: str, models_dir: str = "../models"):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filename : str
        Filename of the model to load
    models_dir : str, optional
        Directory where the model is saved
        
    Returns:
    --------
    object
        The loaded model
    """
    import pickle
    import os
    
    full_path = os.path.join(models_dir, filename)
    
    with open(full_path, 'rb') as f:
        model = pickle.load(f)
    
    return model
