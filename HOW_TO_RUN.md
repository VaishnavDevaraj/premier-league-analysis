# How to Run the Premier League Analysis Project

This guide provides step-by-step instructions for setting up and running the Premier League data analysis project.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for version control)

## Project Setup

### 1. Clone or Download the Project

If using Git:
```bash
git clone <repository-url>
cd PremierLeague_Analysis
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

## Running the Analysis

### 1. Data Exploration and Cleaning

Open and run the first Jupyter notebook:
```bash
jupyter notebook notebooks/1_Data_Exploration_Cleaning.ipynb
```

This notebook will:
- Load the raw Premier League data
- Explore the data structure
- Clean the dataset
- Perform initial visualizations
- Save the cleaned dataset for further analysis

### 2. Statistical Analysis and Visualization

Open and run the second Jupyter notebook:
```bash
jupyter notebook notebooks/2_Statistical_Analysis_Visualization.ipynb
```

This notebook will:
- Load the cleaned dataset
- Perform in-depth statistical analysis
- Create visualizations of key patterns and trends
- Apply hypothesis testing
- Save visualizations for the report

### 3. Predictive Modeling

Open and run the third Jupyter notebook:
```bash
jupyter notebook notebooks/3_Predictive_Modeling.ipynb
```

This notebook will:
- Prepare data for machine learning
- Train and evaluate various models
- Tune hyperparameters
- Analyze feature importance
- Save the best performing model

### 4. Final Report

Open and run the fourth Jupyter notebook:
```bash
jupyter notebook notebooks/4_Final_Report.ipynb
```

This notebook will:
- Summarize all findings
- Present key visualizations
- Provide business recommendations
- Document limitations and future work

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
