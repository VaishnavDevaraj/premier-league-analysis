import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path
import os

# Set page configuration
st.set_page_config(
    page_title="Premier League 2024-25 Analysis Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
DATA_PATH = os.path.join(project_dir, "data")
MODELS_PATH = os.path.join(project_dir, "models")
VISUALIZATIONS_PATH = os.path.join(project_dir, "visualizations")

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try to load cleaned data first
        cleaned_data_path = os.path.join(DATA_PATH, "pl_2024_25_cleaned.csv")
        if os.path.exists(cleaned_data_path):
            st.sidebar.success("Loaded cleaned dataset!")
            df = pd.read_csv(cleaned_data_path)
        else:
            # If not found, load original data
            original_data_path = os.path.join(DATA_PATH, "fbref_PL_2024-25.csv")
            if os.path.exists(original_data_path):
                st.sidebar.info("Loaded original dataset (not cleaned)")
                df = pd.read_csv(original_data_path)
            else:
                st.error(f"Dataset not found! Checked paths:\n{cleaned_data_path}\n{original_data_path}")
                return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
    return df

# MatchPredictionModel class definition (must match the one in train_prediction_model.py)
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

# Function to load model
def load_model():
    try:
        # Check if models directory exists and contains any model files
        if os.path.exists(MODELS_PATH):
            model_files = [f for f in os.listdir(MODELS_PATH) if f.endswith('.pkl')]
            if not model_files:
                return None
                
            # Get the latest model file
            latest_model = os.path.join(MODELS_PATH, model_files[0])
            for model_file in model_files:
                model_path = os.path.join(MODELS_PATH, model_file)
                if os.path.getctime(model_path) > os.path.getctime(latest_model):
                    latest_model = model_path
            
            # Load the model
            with open(latest_model, 'rb') as file:
                model = pickle.load(file)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load data
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Team Analysis", "Player Statistics", "Match Predictions", "League Table"]
)

# Main content
st.title("Premier League 2024-25 Analysis Dashboard")

if page == "Overview":
    st.header("Premier League 2024-25 Overview")
    
    # Display dataset info
    st.subheader("Dataset Information")
    st.write(f"Number of records: {df.shape[0]}")
    st.write(f"Number of features: {df.shape[1]}")
    
    # Show data description
    with st.expander("View Dataset Description"):
        st.write("""
        This dataset contains player statistics from the Premier League 2024-25 season, sourced from FBRef.
        
        **Key Columns:**
        - **Player**: Player name
        - **Squad**: Team name
        - **Pos**: Position
        - **Age**: Player age
        - **Gls**: Goals scored
        - **Ast**: Assists
        - **xG**: Expected goals
        - **xAG**: Expected assisted goals
        - **90s**: Number of 90-minute periods played
        """)
    
    # Show a sample of the data
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # Display basic stats
    st.subheader("Key Statistics")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Note: These are placeholder metrics. Replace with actual calculations based on your data
    with col1:
        st.metric(label="Total Matches", value="X")
    with col2:
        st.metric(label="Average Goals Per Match", value="Y")
    with col3:
        st.metric(label="Total Goals Scored", value="Z")
    
    # Plot overall league statistics
    st.subheader("League Overview")
    
    # Create tabs for different league overview visualizations
    league_tabs = st.tabs(["Team Statistics", "Position Distribution", "Age Distribution", "Goals Analysis"])
    
    try:
        with league_tabs[0]:
            st.subheader("Team Statistics")
            
            if 'Squad' in df.columns:
                # Group by team and calculate aggregated statistics
                team_stats = df.groupby('Squad').agg({
                    'Player': 'count',
                    'Gls': 'sum',
                    'Ast': 'sum',
                    'xG': 'sum',
                    'xAG': 'sum',
                    'CrdY': 'sum',
                    'CrdR': 'sum'
                }).reset_index()
                
                team_stats.rename(columns={'Player': 'Players'}, inplace=True)
                
                # Display team stats table
                st.dataframe(team_stats.sort_values('Gls', ascending=False))
                
                # Create a bar chart of goals by team
                fig = px.bar(
                    team_stats.sort_values('Gls', ascending=False), 
                    x='Squad', 
                    y='Gls',
                    color='Gls',
                    labels={'Squad': 'Team', 'Gls': 'Goals Scored'},
                    title='Goals Scored by Team',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Team statistics visualization requires 'Squad' column which is not available in the dataset")
                
        with league_tabs[1]:
            st.subheader("Position Distribution")
            
            if 'Pos' in df.columns:
                # Calculate position distribution
                pos_counts = df['Pos'].value_counts().reset_index()
                pos_counts.columns = ['Position', 'Count']
                
                # Create a pie chart for position distribution
                fig = px.pie(
                    pos_counts, 
                    values='Count', 
                    names='Position',
                    title='Player Position Distribution',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a bar chart showing position distribution by team
                if 'Squad' in df.columns:
                    pos_by_team = df.groupby(['Squad', 'Pos']).size().reset_index(name='Count')
                    
                    fig = px.bar(
                        pos_by_team,
                        x='Squad',
                        y='Count',
                        color='Pos',
                        title='Position Distribution by Team',
                        labels={'Squad': 'Team', 'Count': 'Number of Players', 'Pos': 'Position'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Position distribution visualization requires 'Pos' column which is not available in the dataset")
        
        with league_tabs[2]:
            st.subheader("Age Distribution")
            
            if 'Age' in df.columns:
                # Create a histogram for age distribution
                fig = px.histogram(
                    df,
                    x='Age',
                    nbins=20,
                    title='Age Distribution of Premier League Players',
                    labels={'Age': 'Age', 'count': 'Number of Players'},
                    color_discrete_sequence=['#3366cc']
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
                
                # Age statistics
                age_stats = {
                    'Average Age': f"{df['Age'].mean():.1f}",
                    'Youngest Player': f"{df['Age'].min():.0f}",
                    'Oldest Player': f"{df['Age'].max():.0f}",
                    'Most Common Age': f"{df['Age'].mode()[0]:.0f}"
                }
                
                # Display age statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Average Age", age_stats['Average Age'])
                col2.metric("Youngest Player", age_stats['Youngest Player'])
                col3.metric("Oldest Player", age_stats['Oldest Player'])
                col4.metric("Most Common Age", age_stats['Most Common Age'])
                
                # Age distribution by position
                if 'Pos' in df.columns:
                    fig = px.box(
                        df,
                        x='Pos',
                        y='Age',
                        title='Age Distribution by Position',
                        labels={'Pos': 'Position', 'Age': 'Age'},
                        color='Pos'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age distribution visualization requires 'Age' column which is not available in the dataset")
                
        with league_tabs[3]:
            st.subheader("Goals Analysis")
            
            if 'Gls' in df.columns:
                # Top goal scorers
                top_scorers = df.sort_values('Gls', ascending=False).head(10)
                
                fig = px.bar(
                    top_scorers,
                    x='Gls',
                    y='Player',
                    orientation='h',
                    title='Top 10 Goal Scorers',
                    labels={'Gls': 'Goals', 'Player': 'Player'},
                    color='Gls',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Goals vs Expected Goals
                if 'xG' in df.columns:
                    # Scatter plot of goals vs expected goals
                    fig = px.scatter(
                        df,
                        x='xG',
                        y='Gls',
                        title='Goals vs Expected Goals (xG)',
                        labels={'xG': 'Expected Goals (xG)', 'Gls': 'Actual Goals'},
                        hover_name='Player',
                        color='Gls',
                        size='90s' if '90s' in df.columns else None,
                        color_continuous_scale='Viridis'
                    )
                    
                    # Add diagonal line (y=x)
                    max_val = max(df['xG'].max(), df['Gls'].max())
                    fig.add_trace(
                        go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Expected = Actual'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation of the chart
                    st.info("""
                    **Interpreting the Goals vs xG Chart:**
                    - **Points above the line:** Players who scored more goals than expected (overperforming)
                    - **Points below the line:** Players who scored fewer goals than expected (underperforming)
                    - **Points on the line:** Players who scored exactly as many goals as expected
                    """)
            else:
                st.info("Goals analysis visualization requires 'Gls' column which is not available in the dataset")
    
    except Exception as e:
        st.error(f"Error in league overview visualization: {e}")
        
    # Update Key Statistics with actual calculations based on available data
    try:
        # Reset the columns to update the metrics
        st.subheader("Key Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Total goals
            if 'Gls' in df.columns:
                total_goals = df['Gls'].sum()
                st.metric(label="Total Goals Scored", value=f"{total_goals:.0f}")
            else:
                st.metric(label="Total Goals Scored", value="N/A")
                
        with col2:
            # Total assists
            if 'Ast' in df.columns:
                total_assists = df['Ast'].sum()
                st.metric(label="Total Assists", value=f"{total_assists:.0f}")
            else:
                st.metric(label="Total Assists", value="N/A")
                
        with col3:
            # Number of teams
            if 'Squad' in df.columns:
                num_teams = df['Squad'].nunique()
                st.metric(label="Number of Teams", value=f"{num_teams}")
            else:
                st.metric(label="Number of Teams", value="N/A")
                
        # Add another row of metrics
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # Number of players
            num_players = df.shape[0]
            st.metric(label="Number of Players", value=f"{num_players}")
            
        with col5:
            # Yellow cards
            if 'CrdY' in df.columns:
                total_yellow_cards = df['CrdY'].sum()
                st.metric(label="Yellow Cards", value=f"{total_yellow_cards:.0f}")
            else:
                st.metric(label="Yellow Cards", value="N/A")
                
        with col6:
            # Red cards
            if 'CrdR' in df.columns:
                total_red_cards = df['CrdR'].sum()
                st.metric(label="Red Cards", value=f"{total_red_cards:.0f}")
            else:
                st.metric(label="Red Cards", value="N/A")
    except Exception as e:
        st.error(f"Error updating key statistics: {e}")

elif page == "Team Analysis":
    st.header("Team Analysis")
    
    # Team selection
    try:
        if 'Squad' in df.columns:
            teams = sorted(df['Squad'].unique())
            selected_team = st.selectbox("Select Team", teams)
            
            # Filter data for selected team
            team_data = df[df['Squad'] == selected_team]
            
            # Display team stats
            st.subheader(f"{selected_team} Statistics")
            
            # Show team metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            # Calculate and display actual team metrics
            with metrics_col1:
                if 'Gls' in team_data.columns:
                    team_goals = team_data['Gls'].sum()
                    st.metric(label="Total Goals", value=f"{team_goals:.0f}")
                else:
                    st.metric(label="Total Goals", value="N/A")
            
            with metrics_col2:
                if 'Ast' in team_data.columns:
                    team_assists = team_data['Ast'].sum()
                    st.metric(label="Total Assists", value=f"{team_assists:.0f}")
                else:
                    st.metric(label="Total Assists", value="N/A")
            
            with metrics_col3:
                squad_size = len(team_data)
                st.metric(label="Squad Size", value=squad_size)
            
            # Additional metrics row
            metrics_col4, metrics_col5, metrics_col6 = st.columns(3)
            
            with metrics_col4:
                if 'xG' in team_data.columns:
                    team_xg = team_data['xG'].sum()
                    st.metric(label="Expected Goals", value=f"{team_xg:.1f}")
                else:
                    st.metric(label="Expected Goals", value="N/A")
            
            with metrics_col5:
                if 'CrdY' in team_data.columns:
                    yellow_cards = team_data['CrdY'].sum()
                    st.metric(label="Yellow Cards", value=f"{yellow_cards:.0f}")
                else:
                    st.metric(label="Yellow Cards", value="N/A")
            
            with metrics_col6:
                if 'CrdR' in team_data.columns:
                    red_cards = team_data['CrdR'].sum()
                    st.metric(label="Red Cards", value=f"{red_cards:.0f}")
                else:
                    st.metric(label="Red Cards", value="N/A")
            
            # Team visualizations
            st.subheader("Team Performance Analysis")
            
            # Create tabs for different team visualizations
            team_tabs = st.tabs(["Player Stats", "Position Distribution", "Age Profile", "Performance Analysis"])
            
            with team_tabs[0]:
                st.subheader("Player Statistics")
                
                # Display player stats table
                st.dataframe(team_data[['Player', 'Pos', 'Age', 'Gls', 'Ast', 'xG', 'xAG', 'Min', '90s']].sort_values('Gls', ascending=False))
                
                # Top goal scorers in the team
                if 'Gls' in team_data.columns and 'Player' in team_data.columns:
                    top_team_scorers = team_data.sort_values('Gls', ascending=False).head(10)
                    
                    fig = px.bar(
                        top_team_scorers,
                        x='Player',
                        y='Gls',
                        title=f'Top Goal Scorers for {selected_team}',
                        color='Gls',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with team_tabs[1]:
                st.subheader("Position Distribution")
                
                if 'Pos' in team_data.columns:
                    # Calculate position distribution
                    pos_counts = team_data['Pos'].value_counts().reset_index()
                    pos_counts.columns = ['Position', 'Count']
                    
                    # Create a pie chart for position distribution
                    fig = px.pie(
                        pos_counts,
                        values='Count',
                        names='Position',
                        title=f'Position Distribution for {selected_team}',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Position data not available for this team")
            
            with team_tabs[2]:
                st.subheader("Age Profile")
                
                if 'Age' in team_data.columns:
                    # Age statistics
                    avg_age = team_data['Age'].mean()
                    youngest = team_data['Age'].min()
                    oldest = team_data['Age'].max()
                    
                    # Display age statistics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average Age", f"{avg_age:.1f}")
                    col2.metric("Youngest Player", f"{youngest:.0f}")
                    col3.metric("Oldest Player", f"{oldest:.0f}")
                    
                    # Create histogram for age distribution
                    fig = px.histogram(
                        team_data,
                        x='Age',
                        nbins=15,
                        title=f'Age Distribution for {selected_team}',
                        color_discrete_sequence=['#3366cc']
                    )
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Age by position
                    if 'Pos' in team_data.columns:
                        fig = px.box(
                            team_data,
                            x='Pos',
                            y='Age',
                            title=f'Age by Position for {selected_team}',
                            color='Pos'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Age data not available for this team")
            
            with team_tabs[3]:
                st.subheader("Performance Analysis")
                
                if 'Gls' in team_data.columns and 'xG' in team_data.columns:
                    # Create scatter plot of goals vs expected goals for team players
                    fig = px.scatter(
                        team_data,
                        x='xG',
                        y='Gls',
                        title=f'Goals vs Expected Goals for {selected_team} Players',
                        labels={'xG': 'Expected Goals (xG)', 'Gls': 'Actual Goals'},
                        hover_name='Player',
                        color='Gls',
                        size='Min' if 'Min' in team_data.columns else None,
                        color_continuous_scale='Viridis'
                    )
                    
                    # Add diagonal line (y=x)
                    max_val = max(team_data['xG'].max(), team_data['Gls'].max())
                    fig.add_trace(
                        go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Expected = Actual'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Team performance vs xG
                    team_goals = team_data['Gls'].sum()
                    team_xg = team_data['xG'].sum()
                    diff = team_goals - team_xg
                    
                    # Display team performance vs expectation
                    st.subheader("Team Performance vs Expectation")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Goals", f"{team_goals:.0f}")
                    col2.metric("Expected Goals", f"{team_xg:.1f}")
                    
                    if diff > 0:
                        col3.metric("Performance", f"+{diff:.1f}", f"Overperforming by {diff:.1f}")
                    elif diff < 0:
                        col3.metric("Performance", f"{diff:.1f}", f"Underperforming by {abs(diff):.1f}")
                    else:
                        col3.metric("Performance", "0", "As expected")
                else:
                    st.info("Goals and/or Expected Goals data not available for this team")
        else:
            st.error("Team data not available in the dataset. The 'Squad' column is missing.")
            
    except Exception as e:
        st.error(f"Error in team analysis: {e}")

elif page == "Player Statistics":
    st.header("Player Statistics")
    
    # Player selection
    try:
        players = sorted(df['Player'].unique())
        selected_player = st.selectbox("Select Player", players)
        
        # Filter data for selected player
        player_data = df[df['Player'] == selected_player]
        
        # Display player stats
        st.subheader(f"{selected_player} Statistics")
        
        # Show player metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        # Display actual metrics from the dataset
        with metrics_col1:
            if 'Gls' in player_data.columns:
                st.metric(label="Goals", value=player_data['Gls'].values[0])
            else:
                st.metric(label="Goals", value="N/A")
                
        with metrics_col2:
            if 'Ast' in player_data.columns:
                st.metric(label="Assists", value=player_data['Ast'].values[0])
            else:
                st.metric(label="Assists", value="N/A")
                
        with metrics_col3:
            if 'Min' in player_data.columns:
                st.metric(label="Minutes Played", value=player_data['Min'].values[0])
            else:
                st.metric(label="Minutes Played", value="N/A")
        
        # Additional metrics row
        metrics_col4, metrics_col5, metrics_col6 = st.columns(3)
        
        with metrics_col4:
            if 'xG' in player_data.columns:
                st.metric(label="Expected Goals (xG)", value=round(player_data['xG'].values[0], 2))
            else:
                st.metric(label="Expected Goals (xG)", value="N/A")
                
        with metrics_col5:
            if 'xAG' in player_data.columns:
                st.metric(label="Expected Assists (xAG)", value=round(player_data['xAG'].values[0], 2))
            else:
                st.metric(label="Expected Assists (xAG)", value="N/A")
                
        with metrics_col6:
            if '90s' in player_data.columns:
                st.metric(label="90s Played", value=round(player_data['90s'].values[0], 1))
            else:
                st.metric(label="90s Played", value="N/A")
        
        # Player performance visualization
        st.subheader("Player Performance Analysis")
        
        # Create tabs for different visualizations
        performance_tabs = st.tabs(["Performance Radar", "Goals vs xG", "Player Comparison"])
        
        with performance_tabs[0]:
            # Performance Radar Chart
            st.subheader("Performance Radar")
            
            # Define the metrics to include in the radar chart
            radar_metrics = ['Gls', 'Ast', 'xG', 'xAG', 'CrdY', 'CrdR']
            radar_display_names = ['Goals', 'Assists', 'xG', 'xAG', 'Yellow Cards', 'Red Cards']
            
            # Get values for radar chart
            radar_values = []
            for metric in radar_metrics:
                if metric in player_data.columns:
                    radar_values.append(float(player_data[metric].values[0]))
                else:
                    radar_values.append(0)
            
            # Create radar chart with plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=radar_display_names,
                fill='toself',
                name=selected_player
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with performance_tabs[1]:
            # Goals vs Expected Goals
            st.subheader("Goals vs Expected Goals")
            
            if 'Gls' in player_data.columns and 'xG' in player_data.columns:
                # Create bar chart comparing goals to xG
                comparison_data = {
                    'Metric': ['Goals', 'Expected Goals'],
                    'Value': [player_data['Gls'].values[0], player_data['xG'].values[0]]
                }
                
                fig = px.bar(
                    comparison_data, 
                    x='Metric', 
                    y='Value',
                    title=f"{selected_player}: Goals vs Expected Goals",
                    color='Metric',
                    color_discrete_map={'Goals': '#1f77b4', 'Expected Goals': '#ff7f0e'}
                )
                
                # Add a horizontal line at y=0
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=1.5,
                    y0=0,
                    y1=0,
                    line=dict(color="black", width=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate over/underperformance
                goals = player_data['Gls'].values[0]
                xg = player_data['xG'].values[0]
                diff = goals - xg
                
                if diff > 0:
                    st.success(f"**Overperforming xG by {diff:.2f} goals**")
                elif diff < 0:
                    # Format the message with the actual difference value
                    st.error(f"**Underperforming xG by {abs(diff):.2f} goals**")
                else:
                    st.info("**Performing exactly as expected**")
            else:
                st.info("Goals and/or Expected Goals data not available")
                
        with performance_tabs[2]:
            # Player Comparison
            st.subheader("Compare with Another Player")
            
            # Select another player to compare with
            other_players = [p for p in sorted(df['Player'].unique()) if p != selected_player]
            comparison_player = st.selectbox("Select player to compare with", other_players)
            
            if comparison_player:
                # Get data for comparison player
                comparison_data = df[df['Player'] == comparison_player]
                
                # Define metrics to compare
                comparison_metrics = ['Gls', 'Ast', 'xG', 'xAG', 'Min', '90s']
                display_names = ['Goals', 'Assists', 'Expected Goals', 'Expected Assists', 'Minutes', '90s Played']
                
                # Create data for comparison
                compare_data = {'Metric': [], 'Value': [], 'Player': []}
                
                for i, metric in enumerate(comparison_metrics):
                    if metric in player_data.columns and metric in comparison_data.columns:
                        # First player
                        compare_data['Metric'].append(display_names[i])
                        compare_data['Value'].append(float(player_data[metric].values[0]))
                        compare_data['Player'].append(selected_player)
                        
                        # Comparison player
                        compare_data['Metric'].append(display_names[i])
                        compare_data['Value'].append(float(comparison_data[metric].values[0]))
                        compare_data['Player'].append(comparison_player)
                
                # Create grouped bar chart
                fig = px.bar(
                    compare_data, 
                    x='Metric', 
                    y='Value',
                    color='Player',
                    barmode='group',
                    title=f"Player Comparison: {selected_player} vs {comparison_player}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a player to compare with")
        
    except Exception as e:
        st.info(f"Player statistics will be implemented based on actual data structure: {e}")

elif page == "Match Predictions":
    st.header("Match Predictions")
    
    # Create team comparison based on actual data
    st.subheader("Team Performance Comparison")
    
    try:
        # Get team stats from actual player data
        if 'Squad' in df.columns:
            # Calculate team-level statistics
            team_stats = df.groupby('Squad').agg({
                'Gls': 'sum',
                'Ast': 'sum',
                'xG': 'sum',
                'xAG': 'sum',
                'CrdY': 'sum',
                'CrdR': 'sum',
                'MP': 'sum',
                '90s': 'sum',
                'Min': 'sum'
            }).reset_index()
            
            # Calculate additional metrics
            team_stats['Goals_per_90'] = team_stats['Gls'] / team_stats['90s']
            team_stats['Assists_per_90'] = team_stats['Ast'] / team_stats['90s']
            team_stats['Cards_per_90'] = (team_stats['CrdY'] + team_stats['CrdR']) / team_stats['90s']
            
            # Create team comparison tool
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Team 1")
                team1 = st.selectbox("Select Team 1", sorted(team_stats['Squad']), key='team1')
                
                # Show team 1 stats
                if team1:
                    team1_data = team_stats[team_stats['Squad'] == team1].iloc[0]
                    
                    # Display key metrics
                    st.metric("Total Goals", f"{team1_data['Gls']:.0f}")
                    st.metric("Total Assists", f"{team1_data['Ast']:.0f}")
                    st.metric("Expected Goals (xG)", f"{team1_data['xG']:.1f}")
                    st.metric("Expected Assists (xA)", f"{team1_data['xAG']:.1f}")
            
            with col2:
                st.subheader("Team 2")
                # Filter out team1 from options
                team2_options = sorted([team for team in team_stats['Squad'] if team != team1])
                team2 = st.selectbox("Select Team 2", team2_options, key='team2')
                
                # Show team 2 stats
                if team2:
                    team2_data = team_stats[team_stats['Squad'] == team2].iloc[0]
                    
                    # Display key metrics
                    st.metric("Total Goals", f"{team2_data['Gls']:.0f}")
                    st.metric("Total Assists", f"{team2_data['Ast']:.0f}")
                    st.metric("Expected Goals (xG)", f"{team2_data['xG']:.1f}")
                    st.metric("Expected Assists (xA)", f"{team2_data['xAG']:.1f}")
            
            # Team comparison
            if team1 and team2:
                st.subheader("Head-to-Head Comparison")
                
                # Create comparison dataframe
                comparison_data = []
                
                # Get team 1 data
                team1_row = team_stats[team_stats['Squad'] == team1].iloc[0]
                team2_row = team_stats[team_stats['Squad'] == team2].iloc[0]
                
                # Calculate normalized values for radar chart
                metrics = {
                    'Goals': (team1_row['Gls'], team2_row['Gls']),
                    'Assists': (team1_row['Ast'], team2_row['Ast']),
                    'xG': (team1_row['xG'], team2_row['xG']),
                    'xA': (team1_row['xAG'], team2_row['xAG']),
                    'Goals/90': (team1_row['Goals_per_90'], team2_row['Goals_per_90'])
                }
                
                # Create radar chart using actual data
                categories = list(metrics.keys())
                team1_values = [metrics[cat][0] for cat in categories]
                team2_values = [metrics[cat][1] for cat in categories]
                
                # Normalize values for better visualization
                max_values = [max(team1_values[i], team2_values[i]) for i in range(len(categories))]
                team1_values_norm = [team1_values[i]/max_values[i] * 10 if max_values[i] > 0 else 0 for i in range(len(categories))]
                team2_values_norm = [team2_values[i]/max_values[i] * 10 if max_values[i] > 0 else 0 for i in range(len(categories))]
                
                # Create radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=team1_values_norm,
                    theta=categories,
                    fill='toself',
                    name=team1
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=team2_values_norm,
                    theta=categories,
                    fill='toself',
                    name=team2
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )
                    ),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed comparison table
                st.subheader("Statistical Comparison")
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Metric': ['Total Goals', 'Total Assists', 'Expected Goals', 'Expected Assists', 
                             'Goals per 90', 'Assists per 90', 'Yellow Cards', 'Red Cards'],
                    team1: [team1_row['Gls'], team1_row['Ast'], team1_row['xG'], team1_row['xAG'],
                          team1_row['Goals_per_90'], team1_row['Assists_per_90'], 
                          team1_row['CrdY'], team1_row['CrdR']],
                    team2: [team2_row['Gls'], team2_row['Ast'], team2_row['xG'], team2_row['xAG'],
                          team2_row['Goals_per_90'], team2_row['Assists_per_90'], 
                          team2_row['CrdY'], team2_row['CrdR']],
                })
                
                # Calculate difference
                comparison_df['Difference'] = comparison_df[team1] - comparison_df[team2]
                
                # Display the comparison table
                st.dataframe(comparison_df, use_container_width=True)
                
                # Goals comparison chart
                st.subheader("Goals and Expected Goals Comparison")
                
                # Create a goals comparison chart
                goals_df = pd.DataFrame({
                    'Team': [team1, team1, team2, team2],
                    'Metric': ['Goals', 'Expected Goals', 'Goals', 'Expected Goals'],
                    'Value': [team1_row['Gls'], team1_row['xG'], team2_row['Gls'], team2_row['xG']]
                })
                
                fig_goals = px.bar(
                    goals_df,
                    x='Team',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title='Goals vs. Expected Goals',
                    labels={'Value': 'Count', 'Team': 'Team', 'Metric': 'Metric'}
                )
                
                st.plotly_chart(fig_goals, use_container_width=True)
                
                # Individual player comparison
                st.subheader("Top Players Comparison")
                
                # Get top 5 goalscorers from each team
                team1_players = df[df['Squad'] == team1].sort_values('Gls', ascending=False).head(5)
                team2_players = df[df['Squad'] == team2].sort_values('Gls', ascending=False).head(5)
                
                # Combine into one dataframe
                top_players = pd.concat([
                    team1_players[['Player', 'Squad', 'Gls', 'Ast', 'xG', 'xAG', '90s']],
                    team2_players[['Player', 'Squad', 'Gls', 'Ast', 'xG', 'xAG', '90s']]
                ])
                
                # Calculate per 90 stats
                top_players['Goals_per_90'] = top_players['Gls'] / top_players['90s']
                top_players['Assists_per_90'] = top_players['Ast'] / top_players['90s']
                
                # Create player comparison chart
                fig_players = px.scatter(
                    top_players,
                    x='xG',
                    y='Gls',
                    size='90s',
                    color='Squad',
                    hover_name='Player',
                    labels={'xG': 'Expected Goals', 'Gls': 'Goals', '90s': 'Minutes Played (90s)'},
                    title='Top Players: Goals vs. Expected Goals'
                )
                
                # Add diagonal line (y=x)
                max_val = max(top_players['xG'].max(), top_players['Gls'].max())
                fig_players.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines',
                        line=dict(color='gray', dash='dash'),
                        name='Expected = Actual'
                    )
                )
                
                st.plotly_chart(fig_players, use_container_width=True)
                
        else:
            st.error("Required columns not found in the dataset.")
    
    except Exception as e:
        st.error(f"Error in team comparison: {e}")

elif page == "League Table":
    st.header("Premier League 2024-25 Table")
    
    try:
        # Create league table from available player statistics
        if df.empty:
            st.error("Dataset is empty. Please check the data source.")
        else:
            # Group by Squad (team) and aggregate player stats
            team_stats = df.groupby('Squad').agg({
                'Player': 'count',
                'Gls': 'sum',
                'Ast': 'sum',
                'xG': 'sum',
                'xAG': 'sum',
                'MP': 'mean',  # Average matches played per player
                'Min': 'sum',  # Total minutes played
                '90s': 'sum',  # Total 90-minute periods played
                'CrdY': 'sum',  # Yellow cards
                'CrdR': 'sum'   # Red cards
            }).reset_index()
            
            # Rename columns for clarity
            team_stats = team_stats.rename(columns={
                'Player': 'Players',
                'Gls': 'Goals',
                'Ast': 'Assists',
                'xG': 'Expected_Goals',
                'xAG': 'Expected_Assists',
                'MP': 'Avg_Matches_Per_Player',
                'Min': 'Total_Minutes',
                '90s': 'Total_90s',
                'CrdY': 'Yellow_Cards',
                'CrdR': 'Red_Cards'
            })
            
            # Calculate additional metrics
            team_stats['Goal_Contributions'] = team_stats['Goals'] + team_stats['Assists']
            team_stats['Expected_Goal_Contributions'] = team_stats['Expected_Goals'] + team_stats['Expected_Assists']
            team_stats['Performance_Index'] = team_stats['Goal_Contributions'] - team_stats['Expected_Goal_Contributions']
            
            # Sort by Goals (as a proxy for team performance)
            team_stats = team_stats.sort_values('Goals', ascending=False).reset_index(drop=True)
            team_stats.index = team_stats.index + 1  # Start position from 1
            
            # Display the league table
            st.subheader("Team Performance Stats")
            
            # Main stats table
            st.dataframe(
                team_stats[['Squad', 'Players', 'Goals', 'Assists', 'Expected_Goals', 'Expected_Assists', 
                           'Goal_Contributions', 'Performance_Index', 'Yellow_Cards', 'Red_Cards']], 
                column_config={
                    "Squad": st.column_config.TextColumn("Team"),
                    "Players": st.column_config.NumberColumn("Players"),
                    "Goals": st.column_config.NumberColumn("Goals"),
                    "Assists": st.column_config.NumberColumn("Assists"),
                    "Expected_Goals": st.column_config.NumberColumn("xG", format="%.1f"),
                    "Expected_Assists": st.column_config.NumberColumn("xA", format="%.1f"),
                    "Goal_Contributions": st.column_config.NumberColumn("G+A"),
                    "Performance_Index": st.column_config.NumberColumn("G+A - xG+xA", format="%.1f"),
                    "Yellow_Cards": st.column_config.NumberColumn("Yellow Cards"),
                    "Red_Cards": st.column_config.NumberColumn("Red Cards")
                },
                use_container_width=True,
                hide_index=False
            )
            
            # Add visualizations based on real data
            st.subheader("Team Scoring Performance")
            
            # Goals visualization
            fig_goals = px.bar(
                team_stats.head(20), 
                x='Squad', 
                y='Goals',
                color='Goals',
                color_continuous_scale='Viridis',
                labels={'Goals': 'Goals Scored', 'Squad': 'Team'},
                title='Premier League Goals Scored by Team'
            )
            
            fig_goals.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_goals, use_container_width=True)
            
            # Expected vs. Actual Goals
            st.subheader("Expected vs. Actual Goals")
            
            # Create a dataframe for plotting
            goals_comparison = team_stats[['Squad', 'Goals', 'Expected_Goals']].copy()
            goals_comparison['Difference'] = goals_comparison['Goals'] - goals_comparison['Expected_Goals']
            goals_comparison = goals_comparison.sort_values('Difference', ascending=False)
            
            # Create a horizontal bar chart for goal difference from expected
            fig_xg = px.bar(
                goals_comparison,
                y='Squad',
                x='Difference',
                color='Difference',
                color_continuous_scale='RdBu',
                labels={'Difference': 'Goals - Expected Goals', 'Squad': 'Team'},
                title='Over/Under Performance vs. Expected Goals',
                orientation='h'
            )
            
            st.plotly_chart(fig_xg, use_container_width=True)
            
            # Goals vs. Assists Scatter Plot
            st.subheader("Goals vs. Assists by Team")
            
            fig_scatter = px.scatter(
                team_stats, 
                x='Goals', 
                y='Assists',
                size='Players',
                color='Goal_Contributions',
                hover_name='Squad',
                color_continuous_scale='Viridis',
                labels={'Goals': 'Goals Scored', 'Assists': 'Assists', 'Goal_Contributions': 'Goal Contributions'},
                title='Team Attacking Output'
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Top Performers Section
            st.subheader("Team Disciplinary Record")
            
            # Calculate cards per 90 minutes
            team_stats['Cards_per_90'] = (team_stats['Yellow_Cards'] + team_stats['Red_Cards']) / team_stats['Total_90s']
            
            # Create a card visualization
            fig_cards = px.bar(
                team_stats.sort_values('Cards_per_90', ascending=False),
                x='Squad',
                y=['Yellow_Cards', 'Red_Cards'],
                labels={'value': 'Number of Cards', 'Squad': 'Team', 'variable': 'Card Type'},
                title='Team Disciplinary Record',
                barmode='stack'
            )
            
            fig_cards.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cards, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating league table: {e}")

# Footer
st.markdown("---")
st.markdown("Premier League 2024-25 Analysis Dashboard | Created with Streamlit")
