# Premier League 2024-25 Analysis Project - User Guide

## Project Overview

This project provides a comprehensive analysis of Premier League 2024-25 player statistics. It includes data exploration, statistical analysis, visualizations, and a interactive dashboard.

## Project Structure

- `data/`: Contains the dataset and processed data files
- `notebooks/`: Jupyter notebooks for detailed analysis
- `scripts/`: Python scripts for data processing and analysis
- `models/`: Trained machine learning models
- `visualizations/`: Generated charts and plots
- `dashboard/`: Streamlit dashboard application
- `reports/`: Analysis reports and findings

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning the repository)

### Installation

1. Clone or download the project repository
2. Navigate to the project directory
3. Create and activate a virtual environment:

   **Windows:**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   **Mac/Linux:**
   ```
   python -m venv venv
   source venv/bin/activate
   ```

4. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Project

#### 1. Project Setup

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
