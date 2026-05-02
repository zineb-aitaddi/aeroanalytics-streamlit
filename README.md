# AeroAnalytics - Streamlit Aviation Data Visualization & Aircraft Detection

## Overview

**AeroAnalytics** is a professional Streamlit web application for aviation data analysis, flight route visualization, airport/route exploration, and aircraft detection using Machine Learning.

The project combines an interactive aviation dashboard with a trained aircraft recognition model. It is designed to show how data science, visualization, and machine learning can support aeronautical analysis and decision-making.

## GitHub Description

Streamlit web application for aviation data visualization, airport and flight route analysis, and aircraft detection using Machine Learning.

## Main Features

- Interactive Streamlit dashboard
- Airport and flight route visualization
- Aviation dataset exploration
- Aircraft specifications analysis
- Operational KPI cards
- Delay, distance, fuel consumption, and CO₂ analysis
- Machine learning-based aircraft detection
- CSV export for filtered data
- PDF report generation support
- Ready for Streamlit Cloud deployment

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- PyDeck
- Pillow
- PyTorch
- TorchVision
- ReportLab
- Kaleido

## Project Structure

```text
AeroAnalytics/
│
├── aeroanalytics.py
├── aircraft_specs.csv
├── flights_aero.csv
├── logo.png
├── requirements.txt
├── README.md
└── models/
    ├── aircraft_resnet18.pth
    └── class_to_idx.json
```

## Files Description

| File / Folder | Description |
|---|---|
| `aeroanalytics.py` | Main Streamlit application file |
| `flights_aero.csv` | Aviation dataset used for routes, KPIs, and visual analytics |
| `aircraft_specs.csv` | Dataset containing aircraft specifications |
| `logo.png` | Application logo |
| `requirements.txt` | Required Python libraries for deployment |
| `models/aircraft_resnet18.pth` | Trained PyTorch aircraft recognition model |
| `models/class_to_idx.json` | Aircraft class mapping used by the ML model |

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/anas-iazza/AeroAnalytics---Streamlit-Aviation-Data-Visualization-Aircraft-Detection.git
```

### 2. Open the project folder

```bash
cd AeroAnalytics---Streamlit-Aviation-Data-Visualization-Aircraft-Detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run aeroanalytics.py
```

## Streamlit Cloud Deployment

Use these deployment settings:

```text
Repository: anas-iazza/AeroAnalytics---Streamlit-Aviation-Data-Visualization-Aircraft-Detection
Branch: main
Main file path: aeroanalytics.py
```

## Notes for Streamlit Cloud

The application is compatible with Streamlit Cloud. The PDF generation feature uses `reportlab` and `kaleido`. In some cloud environments, exporting Plotly charts to images may require Chrome support through Kaleido. If PDF export does not work in the cloud, the main dashboard and machine learning features can still run normally.

## Machine Learning Module

The aircraft detection module uses a ResNet18-based PyTorch model. The user uploads an aircraft image, and the model returns the most probable aircraft classes.

## Skills Demonstrated

- Python application development
- Streamlit dashboard design
- Aviation data analysis
- Data visualization
- Machine learning integration
- Aircraft image classification
- GitHub project organization
- Streamlit Cloud deployment

## Author

**AIT ADDI ZINEB**  
Aeronautical Engineering and Space Technologies Student

## License

This project is developed for academic and learning purposes.
