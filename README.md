# Premier League Data Analysis Dashboard

## Overview
This project provides an interactive dashboard for analyzing English Premier League 2024-25 season data, including team and player statistics, performance metrics, and team comparisons.

## Features
- **League Overview**: Top-level view of Premier League statistics and trends
- **Team Analysis**: Detailed performance metrics for each team
- **Player Statistics**: In-depth player performance analysis and comparisons
- **Match Predictions**: Team comparison tool with performance metrics
- **League Table**: Team performance metrics and statistical visualizations

## Project Structure
- **data/**: Contains raw and processed datasets
- **notebooks/**: Jupyter notebooks for exploratory data analysis and modeling
- **scripts/**: Python scripts for data processing and utility functions
- **models/**: Saved machine learning models
- **visualizations/**: Generated plots and charts
- **dashboard/**: Interactive dashboard files
  - `app.py`: Main dashboard application
  - `app_cloud.py`: Cloud-optimized version for deployment

## Tools & Technologies
- Python (Pandas, NumPy)
- Data Visualization (Matplotlib, Plotly)
- Statistical Analysis (SciPy)
- Dashboard Development (Streamlit)

## Getting Started
1. Clone this repository
   ```bash
   git clone https://github.com/VaishnavDevaraj/premier-league-analysis.git
   cd premier-league-analysis
   ```

2. Install required dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit dashboard
   ```bash
   streamlit run dashboard/app.py
   ```

## Deployment
This dashboard can be deployed using:
- **Streamlit Cloud**: See `DEPLOYMENT.md` for detailed instructions
- **Alternative Version**: For deployment issues, use `app_cloud.py` (see `CLOUD_VERSION.md`)
- **Other Platforms**: The dashboard is compatible with Heroku, AWS, Azure, or Google Cloud

## Documentation
- **HOW_TO_RUN.md**: Detailed instructions for running the project
- **USER_GUIDE.md**: Guide for using the dashboard features
- **DEPLOYMENT.md**: Instructions for deploying to Streamlit Cloud
- **CLOUD_VERSION.md**: Information about the cloud-optimized version
- **PROJECT_SUMMARY.md**: Technical overview of the project

## Data Source
Player statistics data from FBRef for the Premier League 2024-25 season.

## License
MIT

## Author
Vaishnav Devaraj
