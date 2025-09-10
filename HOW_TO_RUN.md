# How to Run the Premier League Analysis Dashboard

This guide provides step-by-step instructions for setting up and running the Premier League data analysis dashboard.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for version control)

## Project Setup

### 1. Clone or Download the Project

If using Git:
```bash
git clone https://github.com/VaishnavDevaraj/premier-league-analysis.git
cd premier-league-analysis
```

Alternatively, download and extract the project files.

### 2. Create and Activate a Virtual Environment (Recommended)

#### For Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

Launch the Streamlit dashboard application:

```bash
streamlit run dashboard/app.py
```

This will start the dashboard on your local machine and automatically open it in your default web browser (typically at http://localhost:8501).

## Dashboard Features

The dashboard includes the following sections:

### 1. Overview
- General information about the Premier League dataset
- Key statistics and metrics
- Dataset structure and information

### 2. Team Analysis
- Detailed metrics for each Premier League team
- Comparative analysis between teams
- Performance trends and insights

### 3. Player Statistics
- Individual player performance metrics
- Player comparisons and rankings
- Advanced statistics and visualizations

### 4. Match Predictions
- Team comparison tool
- Head-to-head analysis
- Performance radar charts
- Top player comparison

### 5. League Table
- Complete Premier League standings
- Team performance metrics
- Scoring analysis
- Expected vs. actual goals comparison
- Disciplinary records

## Troubleshooting

If you encounter any issues:

1. **Dependencies not installing correctly:**
   Make sure you're using a compatible Python version (3.8 or higher).
   Try installing dependencies one by one.

2. **Dashboard not starting:**
   Check if Streamlit is installed correctly: `pip show streamlit`
   Ensure all required data files are in the correct location.

3. **Data visualization errors:**
   Verify that all data files are in the `data/` directory.
   Check that the required columns exist in the dataset.

## Contact

For any issues or questions, please contact the repository owner.

## Running the Dashboard

Launch the interactive Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

This will start a local web server and open the dashboard in your default web browser.

## Additional Scripts

### Data Collection (Optional)

If you want to collect additional data:
```bash
python scripts/data_collection.py
```

**Note:** This script contains placeholder functions. You'll need to implement actual data collection logic as needed.

## Troubleshooting

If you encounter any issues:

1. **Package Installation Errors**:
   - Try updating pip: `pip install --upgrade pip`
   - Install packages one by one to identify problematic dependencies

2. **Jupyter Notebook Issues**:
   - Ensure Jupyter is installed: `pip install jupyter`
   - Try running with: `python -m jupyter notebook`

3. **Streamlit Dashboard Errors**:
   - Ensure Streamlit is installed: `pip install streamlit`
   - Check for any missing dependencies in the error message

4. **Data Loading Errors**:
   - Verify the data file path is correct
   - Check if the data file has been modified or corrupted
