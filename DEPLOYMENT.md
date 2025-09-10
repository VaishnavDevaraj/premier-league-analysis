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

## Handling Deployment Issues

If you encounter dependency installation errors:

### Using requirements_cloud.txt

This project includes a `requirements_cloud.txt` file specifically for cloud deployment. If you encounter issues with the standard requirements.txt file, try:

1. In Streamlit Cloud, go to your app settings
2. Under "Advanced settings", change the "Requirements file path" to `requirements_cloud.txt`
3. Save and redeploy

### Python Version Compatibility

Streamlit Cloud may use a different Python version than your local environment. If you encounter version compatibility issues:

1. In Streamlit Cloud, go to your app settings
2. Under "Advanced settings", specify a compatible Python version (e.g., "3.9" or "3.10")
3. Save and redeploy

### Common Issues and Solutions

1. **Missing system dependencies**: 
   - If you see errors related to missing system packages like `cmake`, contact Streamlit support or consider using a different hosting platform.

2. **Package build failures**:
   - Use prebuilt wheels where possible
   - Avoid pinning to specific versions unless necessary
   - Use version ranges (e.g., `pandas>=1.5.0,<2.0.0`) to allow flexibility

3. **Memory/resource limitations**:
   - Streamlit Cloud has resource limits. If your app is resource-intensive, consider optimizing your code or upgrading your account.

## Alternative Deployment Options

If Streamlit Cloud isn't working for your needs, consider these alternatives:

1. **Heroku**: Good for small to medium-sized applications
2. **Render**: Simple deployment with good free tier
3. **AWS Elastic Beanstalk**: More complex but highly scalable
4. **Google Cloud Run**: Serverless deployment option
5. **Azure App Service**: Microsoft's PaaS offering

## Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [Streamlit GitHub Issues](https://github.com/streamlit/streamlit/issues)
