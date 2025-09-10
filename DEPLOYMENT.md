# Streamlit Cloud Deployment Guide

This document provides instructions for deploying the Premier League Analysis Dashboard to Streamlit Cloud.

## Prerequisites

Before deploying, ensure:
1. Your GitHub repository is public
2. All required files are committed to the repository
3. Your `requirements.txt` file is up-to-date
4. You have a Streamlit Cloud account (register at [streamlit.io/cloud](https://streamlit.io/cloud))

## Deployment Steps

### 1. Login to Streamlit Cloud

Visit [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.

### 2. Deploy a New App

1. Click "New App" button
2. Fill in the repository details:
   - Repository: `VaishnavDevaraj/premier-league-analysis`
   - Branch: `master`
   - Main file path: `dashboard/app.py`
   - Advanced settings: Leave as default unless you need to set environment variables

3. Click "Deploy"

### 3. Wait for Deployment

Streamlit Cloud will pull your repository and install the required dependencies. This may take a few minutes.

### 4. Test Your Deployed App

Once deployed, you'll get a URL like `https://username-app-name.streamlit.app/`. Open this URL to ensure your dashboard is functioning correctly.

### 5. Share Your App

You can now share the URL with others, or add it to your portfolio by:
- Including it in your resume
- Adding it to your LinkedIn profile
- Featuring it in your GitHub profile README

## Updating Your App

When you make changes to your repository and push them to GitHub, Streamlit Cloud will automatically update your app.

## Troubleshooting

If your app fails to deploy:

1. **Check the logs** in the Streamlit Cloud dashboard
2. **Verify dependencies** in your requirements.txt file
3. **Test locally** before deploying
4. **Check file paths** in your code (use relative paths)
5. **Ensure data files** are available and correctly referenced

## Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [Streamlit GitHub Issues](https://github.com/streamlit/streamlit/issues)
