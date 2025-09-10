"""
Script to capture screenshots of the Premier League Analysis Dashboard.
This will help generate images for the README.md file.
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
SCREENSHOTS_DIR = os.path.join(project_dir, "visualizations", "screenshots")

# Create screenshots directory if it doesn't exist
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")

# Function to capture screenshots
def capture_dashboard_screenshots():
    print("Starting dashboard screenshot capture...")
    
    try:
        # Initialize the browser
        driver = webdriver.Chrome(options=chrome_options)
        
        # Dashboard is running at http://localhost:8501
        driver.get("http://localhost:8501")
        
        # Wait for the dashboard to load
        time.sleep(5)
        
        # Capture main page (Overview)
        print("Capturing Overview page...")
        driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "overview.png"))
        
        # Navigate to and capture Team Analysis
        print("Capturing Team Analysis page...")
        team_analysis_link = driver.find_element(By.XPATH, "//div[contains(text(), 'Team Analysis')]")
        team_analysis_link.click()
        time.sleep(3)
        driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "team_analysis.png"))
        
        # Navigate to and capture Player Statistics
        print("Capturing Player Statistics page...")
        player_stats_link = driver.find_element(By.XPATH, "//div[contains(text(), 'Player Statistics')]")
        player_stats_link.click()
        time.sleep(3)
        driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "player_statistics.png"))
        
        # Navigate to and capture Match Predictions
        print("Capturing Match Predictions page...")
        match_pred_link = driver.find_element(By.XPATH, "//div[contains(text(), 'Match Predictions')]")
        match_pred_link.click()
        time.sleep(3)
        driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "match_predictions.png"))
        
        # Navigate to and capture League Table
        print("Capturing League Table page...")
        league_table_link = driver.find_element(By.XPATH, "//div[contains(text(), 'League Table')]")
        league_table_link.click()
        time.sleep(3)
        driver.save_screenshot(os.path.join(SCREENSHOTS_DIR, "league_table.png"))
        
        print("All screenshots captured successfully!")
        
    except Exception as e:
        print(f"Error capturing screenshots: {e}")
    
    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    capture_dashboard_screenshots()
