# Quick Start Guide

## Urban Traffic Congestion Prediction System

This guide will help you get started quickly with the traffic prediction system.

---

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for weather API)

---

## Installation Steps

### 1. Install Dependencies

Open a terminal/command prompt and navigate to the project directory:

```bash
cd traffic_congestion_prediction
```

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- scikit-learn (machine learning)
- requests (API calls)
- jupyter (optional, for notebooks)

---

## Running the Complete Pipeline

Execute the following scripts in order:

### Step 1: Data Collection (~1 minute)
```bash
python src/data_collection.py
```

**What it does**:
- Generates realistic traffic data (17,520 hourly records)
- Fetches weather data from Open-Meteo API
- Simulates event data
- Saves to `data/raw/traffic_data_raw.csv`

### Step 2: Data Preprocessing (~30 seconds)
```bash
python src/data_preprocessing.py
```

**What it does**:
- Cleans data and handles missing values
- Engineers 30+ features
- Creates congestion level labels
- Splits into train/val/test sets
- Saves processed data to `data/processed/`

### Step 3: Exploratory Data Analysis (~1 minute)
```bash
python src/exploratory_analysis.py
```

**What it does**:
- Generates 8 comprehensive visualizations
- Analyzes traffic patterns
- Studies weather impact
- Saves plots to `visualizations/`

### Step 4: Model Training (~2-3 minutes)
```bash
python src/model_training.py
```

**What it does**:
- Trains 5 regression models
- Trains 3 classification models
- Saves trained models to `models/`
- Generates feature importance plots

### Step 5: Model Evaluation (~30 seconds)
```bash
python src/model_evaluation.py
```

**What it does**:
- Evaluates all models on test set
- Creates performance comparison plots
- Generates confusion matrices
- Saves results to `models/` and `visualizations/`

---

## Quick Demo (All at once)

To run everything automatically, use this PowerShell script:

```powershell
# Run complete pipeline
python src/data_collection.py
python src/data_preprocessing.py
python src/exploratory_analysis.py
python src/model_training.py
python src/model_evaluation.py

Write-Host "✓ Pipeline complete! Check visualizations/ and models/ folders"
```

---

## Viewing Results

### Visualizations

Navigate to the `visualizations/` folder to see:

1. **traffic_by_hour.png** - Hourly traffic patterns with rush hours highlighted
2. **traffic_by_weekday.png** - Weekly patterns (weekday vs weekend)
3. **seasonal_patterns.png** - Monthly trends over 2 years
4. **holiday_comparison.png** - Holiday impact analysis
5. **hourly_heatmap.png** - Day × Hour traffic heatmap
6. **correlation_matrix.png** - Feature correlations
7. **weather_impact.png** - Temperature, rain, snow effects
8. **traffic_distribution.png** - Statistical distribution
9. **regression_comparison.png** - Model performance comparison
10. **predictions_vs_actual.png** - Prediction accuracy plots
11. **confusion_matrices.png** - Classification results
12. **feature_importance_*.png** - Feature importance plots

### Model Results

Check `models/` folder for:
- `regression_results.csv` - Regression model comparison
- `classification_results.csv` - Classification model comparison
- `test_regression_results.csv` - Test set regression performance
- `test_classification_results.csv` - Test set classification performance
- `*.pkl` files - Trained models (can be loaded for predictions)

---

## Expected Results

### Regression Models (Traffic Volume Prediction)

| Model | R² Score | RMSE |
|-------|----------|------|
| Random Forest | 0.70-0.75 | 600-650 |
| Gradient Boosting | 0.68-0.73 | 620-680 |
| Ridge Regression | 0.60-0.65 | 800-850 |

**Best Model**: Random Forest (R² ≈ 0.75)

### Classification Models (Congestion Level)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | 78-82% | 0.78-0.82 |
| SVM | 73-77% | 0.73-0.77 |
| Logistic Regression | 69-73% | 0.69-0.73 |

**Best Model**: Random Forest (Accuracy ≈ 82%)

---

## Key Insights You'll Discover

From your analysis, you should find:

✅ **Peak traffic occurs at 8 AM and 5 PM** (rush hours)  
✅ **Weekday traffic is 30-35% higher** than weekends  
✅ **Heavy rain reduces traffic by ~30%**  
✅ **Holidays decrease traffic by 15-25%**  
✅ **Lagged features are most predictive** (previous hour traffic)  
✅ **Temperature has moderate correlation** with traffic  

---

## Troubleshooting

### "ModuleNotFoundError"
**Solution**: Make sure you installed requirements
```bash
pip install -r requirements.txt
```

### "No module named 'requests'"
**Solution**: Install requests specifically
```bash
pip install requests
```

### Weather API fails
**Don't worry!** The script automatically falls back to synthetic weather data.

### Slow execution
**Normal.** Model training can take 2-3 minutes depending on your computer.

---

## Next Steps (For Students)

After running the pipeline, try:

1. **Modify Features**:
   - Add new engineered features in `data_preprocessing.py`
   - Try polynomial features, interactions

2. **Tune Hyperparameters**:
   - Adjust `n_estimators`, `max_depth` in models
   - Use GridSearchCV for optimization

3. **Try New Models**:
   - XGBoost, LightGBM
   - Neural Networks (LSTM for time series)

4. **Improve Visualizations**:
   - Add more plots in `exploratory_analysis.py`
   - Create interactive plots with Plotly

5. **Real-Time Predictions**:
   - Build a web app using Flask/Streamlit
   - Create a dashboard

---

## Documentation

- **README.md** - Full project overview
- **REPORT.md** - Comprehensive technical report (30 pages)
- **This file** - Quick start guide

---

## Support

For detailed explanations of methodology, results, and recommendations, please read **REPORT.md**.

For project structure and setup, see **README.md**.

---

**Happy Analyzing! 🚗📊**
