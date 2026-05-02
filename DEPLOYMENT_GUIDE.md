# Streamlit Deployment Guide

## Correct deployment settings

```text
Repository: anas-iazza/AeroAnalytics---Streamlit-Aviation-Data-Visualization-Aircraft-Detection
Branch: main
Main file path: aeroanalytics.py
```

## Steps

1. Push all files to GitHub.
2. Go to Streamlit Cloud.
3. Click **New app**.
4. Select the repository.
5. Set the main file path to `aeroanalytics.py`.
6. Click **Deploy**.

## If the app does not open

- Make sure you are logged in to Streamlit with the same GitHub account: `anas-iazza`.
- Reconnect GitHub from Streamlit settings if necessary.
- Do not use an old Streamlit app link after renaming the repository or the Python file.

## If PDF export fails

The dashboard can still work. The PDF export may fail on Streamlit Cloud if Kaleido or Chrome is unavailable. The export works better locally after installing the dependencies.
