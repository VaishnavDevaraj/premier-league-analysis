# Using the Cloud-Optimized Dashboard Version

If you're experiencing deployment issues with the standard dashboard, this project includes a simplified cloud-friendly version.

## What's Different in the Cloud Version?

The file `dashboard/app_cloud.py` contains a version of the dashboard that:

1. Has simplified dependencies that are more compatible with Streamlit Cloud
2. Includes additional error handling for cloud environments
3. Is more flexible about data file locations
4. Has more verbose error reporting for troubleshooting

## How to Use It

### Local Testing

You can test the cloud version locally by running:

```bash
streamlit run dashboard/app_cloud.py
```

### Streamlit Cloud Deployment

To use the cloud version in Streamlit Cloud:

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. When deploying, set the "Main file path" to:
   ```
   dashboard/app_cloud.py
   ```
3. Use the `requirements_cloud.txt` file for dependencies

## Troubleshooting Data File Issues

The cloud version will look for the data file in multiple locations:

1. `/data/pl_2024_25_cleaned.csv` (cleaned version)
2. `/data/fbref_PL_2024-25.csv` (original version)
3. `/fbref_PL_2024-25.csv` (in repository root)

If your deployment is failing due to data file not found:

1. Make sure the CSV file is committed to your repository
2. Try moving the CSV file to the root of your repository
3. Check the Streamlit Cloud logs for specific error messages

## Getting Help

If you continue to experience issues:

1. Check the Streamlit Cloud logs
2. Try the troubleshooting steps in `DEPLOYMENT.md`
3. Consider using a different hosting provider if Streamlit Cloud is not compatible with your needs
