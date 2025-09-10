"""
Premier League Analysis Project Setup Script

This script sets up the project structure, creates necessary directories,
and organizes the data for analysis.
"""

import os
import shutil
import pandas as pd
import sys

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def main():
    # Define project root directory (where this script is located)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Print welcome message
    print("\n" + "="*60)
    print("Premier League 2024-25 Analysis Project Setup")
    print("="*60 + "\n")
    
    # Create project structure
    directories = [
        'data',
        'notebooks',
        'scripts',
        'models',
        'visualizations',
        'dashboard',
        'reports'
    ]
    
    print("Setting up project directories...")
    for directory in directories:
        create_directory(os.path.join(project_root, directory))
    
    # Look for the dataset in current directory or parent directory
    source_file = None
    possible_locations = [
        os.path.join(project_root, 'fbref_PL_2024-25.csv'),
        os.path.join(os.path.dirname(project_root), 'fbref_PL_2024-25.csv')
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            source_file = location
            break
    
    if source_file:
        # Move dataset to data directory
        dest_file = os.path.join(project_root, 'data', 'fbref_PL_2024-25.csv')
        
        if not os.path.exists(dest_file):
            shutil.copy2(source_file, dest_file)
            print(f"Copied dataset to: {dest_file}")
        else:
            print(f"Dataset already exists in data directory: {dest_file}")
        
        # Verify the dataset
        try:
            df = pd.read_csv(dest_file)
            print(f"\nDataset verification successful!")
            print(f"Number of rows: {df.shape[0]}")
            print(f"Number of columns: {df.shape[1]}")
            print(f"Columns: {', '.join(df.columns[:5])}... (and {len(df.columns)-5} more)")
        except Exception as e:
            print(f"Error verifying dataset: {e}")
    else:
        print("ERROR: Dataset 'fbref_PL_2024-25.csv' not found!")
        print("Please make sure the dataset is in the project root or parent directory.")
        sys.exit(1)
    
    # Create requirements.txt file
    requirements = [
        "# Data Analysis",
        "pandas>=1.5.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "scipy>=1.9.0",
        "statsmodels>=0.13.0",
        "",
        "# Machine Learning",
        "scikit-learn>=1.0.0",
        "xgboost>=1.6.0",
        "",
        "# Visualization",
        "plotly>=5.10.0",
        "",
        "# Dashboard",
        "streamlit>=1.20.0",
        "",
        "# Jupyter",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
        "ipykernel>=6.15.0"
    ]
    
    requirements_path = os.path.join(project_root, 'requirements.txt')
    with open(requirements_path, 'w') as f:
        f.write('\n'.join(requirements))
    print(f"\nCreated requirements.txt file")
    
    # Create README.md file
    readme = [
        "# Premier League 2024-25 Analysis Project",
        "",
        "A comprehensive data analysis project using Premier League 2024-25 player statistics.",
        "",
        "## Project Structure",
        "",
        "- `data/`: Contains the dataset and processed data files",
        "- `notebooks/`: Jupyter notebooks for analysis",
        "- `scripts/`: Python scripts for data processing and analysis",
        "- `models/`: Trained machine learning models",
        "- `visualizations/`: Generated charts and plots",
        "- `dashboard/`: Streamlit dashboard app",
        "- `reports/`: Analysis reports and findings",
        "",
        "## Setup",
        "",
        "1. Clone this repository",
        "2. Create a virtual environment: `python -m venv venv`",
        "3. Activate the environment:",
        "   - Windows: `venv\\Scripts\\activate`",
        "   - Mac/Linux: `source venv/bin/activate`",
        "4. Install requirements: `pip install -r requirements.txt`",
        "",
        "## Running the Analysis",
        "",
        "1. Run the demo script: `python scripts/run_demo.py`",
        "2. Explore the Jupyter notebooks in the `notebooks/` directory",
        "3. Launch the dashboard: `streamlit run dashboard/app.py`",
        "",
        "## Data Source",
        "",
        "The data is sourced from FBRef, a comprehensive football statistics website.",
        "",
        "## License",
        "",
        "MIT"
    ]
    
    readme_path = os.path.join(project_root, 'README.md')
    with open(readme_path, 'w') as f:
        f.write('\n'.join(readme))
    print(f"Created README.md file")
    
    # Create run instructions
    how_to_run = [
        "# How to Run the Premier League Analysis Project",
        "",
        "## Prerequisites",
        "",
        "- Python 3.8 or higher",
        "- Git (optional, for cloning the repository)",
        "",
        "## Step 1: Environment Setup",
        "",
        "### Create and activate a virtual environment",
        "",
        "#### Windows:",
        "```",
        "python -m venv venv",
        "venv\\Scripts\\activate",
        "```",
        "",
        "#### Mac/Linux:",
        "```",
        "python -m venv venv",
        "source venv/bin/activate",
        "```",
        "",
        "### Install dependencies",
        "",
        "```",
        "pip install -r requirements.txt",
        "```",
        "",
        "## Step 2: Data Analysis",
        "",
        "### Run the demo script",
        "",
        "```",
        "python scripts/run_demo.py",
        "```",
        "",
        "This will perform basic analysis and generate visualizations in the `visualizations/` directory.",
        "",
        "### Explore Jupyter notebooks",
        "",
        "Start Jupyter Notebook server:",
        "```",
        "jupyter notebook",
        "```",
        "",
        "Navigate to the `notebooks/` directory and open the notebooks in sequence:",
        "1. `01_Data_Exploration.ipynb`",
        "2. `02_Statistical_Analysis.ipynb`",
        "3. `03_Predictive_Modeling.ipynb`",
        "4. `04_Final_Report.ipynb`",
        "",
        "## Step 3: Interactive Dashboard",
        "",
        "Run the Streamlit dashboard:",
        "```",
        "streamlit run dashboard/app.py",
        "```",
        "",
        "This will start a local web server, and the dashboard will open in your default web browser.",
        "",
        "## Troubleshooting",
        "",
        "- If you encounter package import errors, ensure all dependencies are installed: `pip install -r requirements.txt`",
        "- For visualization issues, make sure matplotlib, seaborn, and plotly are correctly installed",
        "- If the dashboard doesn't load properly, check that Streamlit is installed and running on the correct port"
    ]
    
    how_to_run_path = os.path.join(project_root, 'HOW_TO_RUN.md')
    with open(how_to_run_path, 'w') as f:
        f.write('\n'.join(how_to_run))
    print(f"Created HOW_TO_RUN.md file")
    
    print("\n" + "="*60)
    print("Project setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Create a virtual environment: python -m venv venv")
    print("2. Activate the environment: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Mac/Linux)")
    print("3. Install requirements: pip install -r requirements.txt")
    print("4. Run the demo script: python scripts/run_demo.py")
    print("5. Start the dashboard: streamlit run dashboard/app.py")
    print("\nFor more detailed instructions, see HOW_TO_RUN.md")

if __name__ == "__main__":
    main()
