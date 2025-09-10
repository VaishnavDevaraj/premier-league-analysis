# Premier League 2024-25 Analysis Dashboard - User Guide

## Dashboard Overview

This interactive dashboard provides comprehensive analysis and visualization of Premier League 2024-25 player and team statistics. It offers various features to explore performance metrics, team comparisons, and league standings.

## Getting Started

### Accessing the Dashboard

1. **Local Installation:** Follow instructions in `HOW_TO_RUN.md` to set up and run the dashboard locally
2. **Online Version:** Access the deployed version at [Streamlit Cloud](https://share.streamlit.io/) (link will be added once deployed)

### Navigation

The dashboard has a sidebar navigation menu with the following sections:
- Overview
- Team Analysis
- Player Statistics
- Match Predictions
- League Table

Click on any section in the sidebar to navigate to that part of the dashboard.

## Dashboard Sections

### 1. Overview

This section provides a general introduction to the Premier League dataset:

- **Dataset Information:** Shows the number of records and features in the dataset
- **Data Description:** Displays summary statistics of the data
- **Feature Distribution:** Visualizes the distribution of key metrics
- **Data Exploration:** Presents initial insights from the dataset

### 2. Team Analysis

This section allows you to analyze individual team performance:

- **Team Selection:** Choose a team from the dropdown menu
- **Team Stats:** View aggregated team metrics and performance indicators
- **Player Contribution:** See how individual players contribute to team performance
- **Positional Analysis:** Analyze team strength by player positions
- **Team Form:** Visualize performance trends and patterns

### 3. Player Statistics

This section lets you explore and compare player performances:

- **Player Selection:** Search and select players to analyze
- **Performance Metrics:** View comprehensive stats for selected players
- **Player Comparison:** Compare multiple players across various metrics
- **Position Filtering:** Filter players by position
- **Ranking:** See top performers in different categories

### 4. Match Predictions

This section provides a team comparison tool:

- **Team Selection:** Choose two teams to compare
- **Head-to-Head Analysis:** View comprehensive comparison of team metrics
- **Radar Chart:** Visualize relative strengths and weaknesses
- **Statistical Comparison:** See detailed metric comparisons
- **Key Player Analysis:** Compare top performers from both teams

### 5. League Table

This section presents the Premier League standings:

- **Complete Table:** View the full Premier League table with key metrics
- **Team Performance:** See goals scored, assists, and other team statistics
- **Expected vs. Actual Goals:** Compare expected and actual performance
- **Disciplinary Record:** View yellow and red card statistics by team
- **Visualizations:** Explore various charts and graphs of league performance

## Using Dashboard Features

### Interactive Elements

- **Dropdowns:** Select options from dropdown menus
- **Sliders:** Adjust ranges by dragging the slider
- **Checkboxes:** Toggle options on/off
- **Tabs:** Switch between different views
- **Expandable Sections:** Click to expand/collapse sections

### Data Visualization

- **Hover:** Hover over charts to see detailed information
- **Zoom:** Click and drag on some charts to zoom in
- **Pan:** On zoomed charts, click and drag to pan
- **Reset:** Double-click to reset the view
- **Download:** Some charts have download options in the top-right corner

### Filtering Data

- Use provided filters to narrow down the data
- Combine multiple filters for more specific results
- Reset filters by clicking "Reset" or refreshing the page

## Tips and Tricks

1. **Performance:** If the dashboard is slow, try reducing the amount of data displayed
2. **Export:** Some visualizations allow you to export as PNG
3. **Refresh:** If data seems stale, refresh the browser
4. **Mobile View:** The dashboard is responsive but works best on larger screens
5. **Dark Mode:** Streamlit supports dark mode via browser settings

## Troubleshooting

- **Loading Issues:** If charts don't load, try refreshing the page
- **Missing Data:** Some analyses require complete data; missing values may affect results
- **Browser Compatibility:** For best experience, use Chrome, Firefox, or Edge
- **Connection Problems:** Ensure you have a stable internet connection

## Getting Help

If you encounter issues or have questions:
- Check the project repository on GitHub
- Open an issue on the GitHub repository
- Contact the project maintainer

## Data Sources

All data is sourced from FBRef for the Premier League 2024-25 season.

Run the setup script to organize the project structure and prepare the data:
```
python setup_project.py
```

#### 2. Data Analysis Demo

Run the demo script to perform basic analysis and generate visualizations:
```
python scripts\run_demo.py
```

This script:
- Loads the Premier League dataset
- Cleans and processes the data
- Creates new features (per-90 statistics)
- Generates several visualizations:
  - Top goal scorers
  - Position distribution
  - Age distribution
  - Goals vs Expected Goals
- Saves the cleaned dataset

#### 3. Interactive Dashboard

Start the Streamlit dashboard to explore the data interactively:
```
streamlit run dashboard\app.py
```

The dashboard will open in your web browser at http://localhost:8501

### Dashboard Features

The dashboard includes several pages:

1. **Overview**: General statistics about the Premier League season
   - Dataset preview
   - Summary statistics
   - Top goal scorers
   - Age distribution

2. **Player Analysis**: In-depth analysis of individual player performance
   - Player profile
   - Performance metrics
   - Radar chart of key statistics
   - Comparison of actual vs expected goals/assists

3. **Team Analysis**: Team-level statistics and comparisons
   - Team performance summary
   - Goals by team
   - Comparison of actual vs expected goals

4. **Position Analysis**: Analysis of player performance by position
   - Position distribution
   - Performance statistics by position
   - Comparison of various metrics across positions

## Data Description

The dataset contains information about Premier League players for the 2024-25 season, including:

- Player information (name, nationality, age, position, team)
- Playing time statistics (matches played, starts, minutes)
- Goal and assist data
- Expected goals (xG) and expected assists (xA)
- Cards (yellow and red)
- Progression metrics

## Project Extensions

To extend the project, consider:

1. **Predictive Modeling**:
   - Player performance prediction
   - Team success factors
   - Identifying undervalued players

2. **Additional Visualizations**:
   - Player comparison tool
   - Team formation analysis
   - Performance trends over time

3. **Advanced Analysis**:
   - Clustering players by performance profile
   - Positional analysis
   - Age vs performance analysis

## Troubleshooting

- **Missing packages**: Run `pip install -r requirements.txt`
- **Dashboard not starting**: Ensure Streamlit is installed correctly
- **Visualization errors**: Check that matplotlib, seaborn, and plotly are installed
- **Data not found**: Verify the dataset is in the correct location (data directory)

## Contact

For questions or assistance, please open an issue on the project repository.
