"""
Premier League 2024-25 Analysis - Demo Script

This script demonstrates how to run the key components of the Premier League analysis project.
It shows how to load the data, perform basic analysis, and create visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = [12, 8]

# Make sure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
os.chdir(project_dir)

print("Premier League 2024-25 Analysis Demo")
print("=" * 40)

# Load the dataset
print("\n1. Loading the dataset...")
try:
    df = pd.read_csv('data/fbref_PL_2024-25.csv')
    print(f"Dataset loaded successfully with shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Display basic information
print("\n2. Basic dataset information:")
print(f"Number of players: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print("\nFirst few rows:")
print(df.head(3))

# Display column information
print("\n3. Column data types:")
print(df.dtypes)

# Display missing values
print("\n4. Missing values check:")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(missing_cols)
else:
    print("No missing values found!")

# Clean the data
print("\n5. Cleaning the data...")
df_clean = df.copy()

# Convert numerical columns
numeric_cols = ['Age', 'Born', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A', 
                'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG']

for col in numeric_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Feature engineering
print("\n6. Creating new features...")
# Create goals per 90 minutes
df_clean['goals_per_90'] = df_clean['Gls'] / df_clean['90s']
# Create assists per 90 minutes
df_clean['assists_per_90'] = df_clean['Ast'] / df_clean['90s']
# Create goal contributions per 90 minutes
df_clean['goal_contributions_per_90'] = df_clean['G+A'] / df_clean['90s']

# Handle infinity and NaN values
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# Visualization 1: Top Goal Scorers
print("\n7. Creating visualizations...")
print("   - Top goal scorers chart")
plt.figure(figsize=(12, 8))
top_scorers = df_clean.sort_values('Gls', ascending=False).head(10)
sns.barplot(x='Gls', y='Player', data=top_scorers)
plt.title('Top 10 Goal Scorers')
plt.xlabel('Goals')
plt.ylabel('Player')
plt.tight_layout()
plt.savefig('visualizations/top_goal_scorers.png')
plt.close()

# Visualization 2: Position Distribution
print("   - Position distribution chart")
plt.figure(figsize=(10, 6))
pos_counts = df_clean['Pos'].value_counts()
sns.barplot(x=pos_counts.index, y=pos_counts.values)
plt.title('Player Position Distribution')
plt.xlabel('Position')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/position_distribution.png')
plt.close()

# Visualization 3: Age Distribution
print("   - Age distribution chart")
plt.figure(figsize=(12, 6))
sns.histplot(df_clean['Age'].dropna(), bins=20, kde=True)
plt.title('Age Distribution of Premier League Players')
plt.xlabel('Age')
plt.ylabel('Number of Players')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/age_distribution.png')
plt.close()

# Visualization 4: Goals vs xG
print("   - Goals vs Expected Goals (xG) chart")
plt.figure(figsize=(12, 8))
plt.scatter(df_clean['xG'], df_clean['Gls'], alpha=0.5)
plt.plot([0, df_clean['xG'].max()], [0, df_clean['xG'].max()], 'r--')  # Diagonal line
plt.title('Goals vs Expected Goals (xG)')
plt.xlabel('Expected Goals (xG)')
plt.ylabel('Actual Goals')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/goals_vs_xg.png')
plt.close()

# Save the cleaned dataset
print("\n8. Saving the cleaned dataset...")
df_clean.to_csv('data/pl_2024_25_cleaned.csv', index=False)
print("   - Saved to data/pl_2024_25_cleaned.csv")

print("\n9. Demo analysis complete!")
print("   - Visualizations saved to the 'visualizations' directory")
print("   - Next steps: Run the Streamlit dashboard with:")
print("     streamlit run dashboard/app.py")
print("\n" + "=" * 40)
