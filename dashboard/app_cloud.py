import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

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

# Add error handling for cloud deployment
try:
    # Check if we're running in Streamlit Cloud
    if os.environ.get('STREAMLIT_SHARING'):
        # In Streamlit Cloud, data might be in the root of the repo
        root_data_path = os.path.join(project_dir, "fbref_PL_2024-25.csv")
        if os.path.exists(root_data_path):
            DATA_PATH = project_dir
except:
    # If there's any error, just continue with the default paths
    pass

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
                st.sidebar.info("Loaded original dataset")
                df = pd.read_csv(original_data_path)
            else:
                # Final fallback - try to find the file in the repository root
                root_data_path = os.path.join(project_dir, "fbref_PL_2024-25.csv")
                if os.path.exists(root_data_path):
                    st.sidebar.info("Loaded dataset from repository root")
                    df = pd.read_csv(root_data_path)
                else:
                    st.error(f"Dataset not found! Checked paths:\n{cleaned_data_path}\n{original_data_path}\n{root_data_path}")
                    return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.exception(e)
        return pd.DataFrame()

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

if df.empty:
    st.error("Failed to load dataset. Please check the data file location.")
    st.write("This dashboard requires the Premier League 2024-25 dataset to function.")
    
    # Display paths checked
    st.write("Paths checked:")
    st.code(os.path.join(DATA_PATH, "pl_2024_25_cleaned.csv"))
    st.code(os.path.join(DATA_PATH, "fbref_PL_2024-25.csv"))
    st.code(os.path.join(project_dir, "fbref_PL_2024-25.csv"))
    
    # Environment info for debugging
    st.write("### Environment Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Files in current directory: {os.listdir('.')}")
    st.write(f"Files in data directory: {os.listdir(DATA_PATH) if os.path.exists(DATA_PATH) else 'Data directory not found'}")
    
    # Exit early
    st.stop()

# Overview Page
if page == "Overview":
    st.header("Premier League 2024-25 Overview")
    
    # Display dataset info
    st.subheader("Dataset Information")
    st.write(f"Number of records: {df.shape[0]}")
    st.write(f"Number of features: {df.shape[1]}")
    
    # Show data description
    with st.expander("View Dataset Description"):
        st.write(df.describe())
    
    # Show data sample
    with st.expander("View Data Sample"):
        st.dataframe(df.head(10))
    
    # Team distribution
    st.subheader("Teams in Dataset")
    team_counts = df['Squad'].value_counts().reset_index()
    team_counts.columns = ['Team', 'Number of Players']
    
    fig = px.bar(
        team_counts, 
        x='Team', 
        y='Number of Players',
        color='Number of Players',
        color_continuous_scale='Viridis',
        title='Number of Players by Team'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Position distribution
    st.subheader("Player Positions")
    if 'Pos' in df.columns:
        # Clean position data
        df['Position'] = df['Pos'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
        position_counts = df['Position'].value_counts().reset_index()
        position_counts.columns = ['Position', 'Count']
        
        fig = px.pie(
            position_counts, 
            values='Count', 
            names='Position',
            title='Distribution of Player Positions',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Goals distribution
    st.subheader("Goals Distribution")
    if 'Gls' in df.columns:
        top_scorers = df.sort_values('Gls', ascending=False).head(10)
        
        fig = px.bar(
            top_scorers,
            x='Player',
            y='Gls',
            color='Squad',
            title='Top 10 Goal Scorers',
            hover_data=['Pos', 'Age']
        )
        st.plotly_chart(fig, use_container_width=True)

# Placeholder for other pages
elif page == "Team Analysis":
    st.header("Team Analysis")
    
    # Team selection
    if 'Squad' in df.columns:
        teams = sorted(df['Squad'].unique())
        team = st.selectbox("Select a team", teams)
        
        # Filter data for selected team
        team_data = df[df['Squad'] == team]
        
        # Team overview
        st.subheader(f"{team} Team Overview")
        
        # Display team metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Number of Players", len(team_data))
            if 'Gls' in team_data.columns:
                st.metric("Total Goals", team_data['Gls'].sum())
        
        with metrics_col2:
            if 'Age' in team_data.columns:
                st.metric("Average Age", f"{team_data['Age'].mean():.1f}")
            if 'Ast' in team_data.columns:
                st.metric("Total Assists", team_data['Ast'].sum())
        
        with metrics_col3:
            if 'Min' in team_data.columns:
                total_minutes = team_data['Min'].sum()
                st.metric("Total Minutes Played", f"{total_minutes:,}")
            if 'CrdY' in team_data.columns and 'CrdR' in team_data.columns:
                st.metric("Cards (Y/R)", f"{team_data['CrdY'].sum()}/{team_data['CrdR'].sum()}")
        
        # Show team players
        st.subheader(f"{team} Players")
        st.dataframe(team_data)
        
        # Player performances
        if 'Gls' in team_data.columns and 'Ast' in team_data.columns:
            st.subheader("Player Performances")
            
            fig = px.scatter(
                team_data,
                x='Gls',
                y='Ast',
                hover_name='Player',
                size='Min',
                color='Pos',
                title=f"{team} - Goals vs Assists by Player"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Required columns not found in the dataset.")

elif page == "Player Statistics":
    st.header("Player Statistics")
    
    # Player search
    player_name = st.text_input("Search for a player", "")
    
    if player_name:
        # Search for player (case insensitive)
        player_data = df[df['Player'].str.contains(player_name, case=False, na=False)]
        
        if not player_data.empty:
            # Display player list if multiple matches
            if len(player_data) > 1:
                st.write(f"Found {len(player_data)} players matching '{player_name}':")
                selected_player = st.selectbox("Select a player", player_data['Player'].tolist())
                player_data = df[df['Player'] == selected_player]
            
            # Display player info
            player_info = player_data.iloc[0]
            st.subheader(f"{player_info['Player']} ({player_info['Squad']})")
            
            # Player metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Position", player_info['Pos'])
                if 'Nation' in player_info:
                    st.metric("Nationality", player_info['Nation'])
            
            with col2:
                if 'Age' in player_info:
                    st.metric("Age", player_info['Age'])
                if 'MP' in player_info:
                    st.metric("Matches Played", player_info['MP'])
            
            with col3:
                if 'Gls' in player_info:
                    st.metric("Goals", player_info['Gls'])
                if 'Min' in player_info:
                    st.metric("Minutes Played", player_info['Min'])
            
            with col4:
                if 'Ast' in player_info:
                    st.metric("Assists", player_info['Ast'])
                if 'G+A' in player_info:
                    st.metric("Goal Contributions", player_info['G+A'])
            
            # Display detailed stats
            st.subheader("Detailed Statistics")
            
            # Convert player data to a more readable format
            display_columns = [col for col in player_data.columns if col not in ['Rk']]
            display_data = player_data[display_columns].T.reset_index()
            display_data.columns = ['Statistic', 'Value']
            
            st.dataframe(display_data, use_container_width=True)
        else:
            st.warning(f"No players found matching '{player_name}'")
    
    # Top performers section
    st.subheader("Top Performers")
    
    metric = st.selectbox(
        "Select metric", 
        ["Goals", "Assists", "Minutes Played", "Matches Played", "Yellow Cards", "Red Cards"]
    )
    
    # Map selected metric to dataframe column
    metric_map = {
        "Goals": "Gls",
        "Assists": "Ast", 
        "Minutes Played": "Min", 
        "Matches Played": "MP",
        "Yellow Cards": "CrdY",
        "Red Cards": "CrdR"
    }
    
    if metric_map[metric] in df.columns:
        # Get top 10 players for selected metric
        top_players = df.sort_values(metric_map[metric], ascending=False).head(10)
        
        fig = px.bar(
            top_players, 
            x='Player', 
            y=metric_map[metric],
            color='Squad',
            title=f'Top 10 Players by {metric}',
            hover_data=['Pos', 'Age']
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Column for {metric} not found in the dataset")

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
        else:
            st.error("Required columns not found in the dataset.")
    
    except Exception as e:
        st.error(f"Error in team comparison: {e}")
        st.exception(e)

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
    except Exception as e:
        st.error(f"Error generating league table: {e}")
        st.exception(e)
