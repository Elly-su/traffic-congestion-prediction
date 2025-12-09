# Urban Traffic Congestion Prediction Using Data Science

A comprehensive machine learning solution for predicting urban traffic congestion using real-world data. This project demonstrates the complete data science workflow from data collection through model deployment, achieving 75% prediction accuracy (R² = 0.75) and 82% congestion classification accuracy.

## 🎯 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python src/data_collection.py
python src/data_preprocessing.py
python src/exploratory_analysis.py
python src/model_training.py
python src/model_evaluation.py
```

## 🎮 Interactive Dashboard

**NEW!** Launch the interactive web dashboard for real-time predictions and data exploration:

```bash
streamlit run app.py
```

**Dashboard Features:**
- 🔮 **Make Predictions**: Enter conditions and get instant traffic forecasts
- 📊 **Explore Data**: Interactive charts with filters and time series analysis
- 🎯 **Model Performance**: Compare all models with visualizations
- 💡 **Insights**: Key findings and actionable recommendations
- 📱 **Responsive Design**: Modern UI with smooth interactions

**Access at**: `http://localhost:8501` after running the command above

## 📈 Visualizations

### Traffic Patterns

![Traffic by Hour](visualizations/traffic_by_hour.png)
*Peak traffic occurs at 8 AM and 5 PM during rush hours*

![Weekly Patterns](visualizations/traffic_by_weekday.png)
*Weekday traffic is 30% higher than weekends*

### Model Performance

![Model Comparison](visualizations/regression_comparison.png)
*Random Forest achieves best performance with R² = 0.75*

![Feature Importance](visualizations/feature_importance_random_forest_regression.png)
*Top features: Rolling mean (24h), Previous hour traffic, Hour of day*

### Classification Results

![Confusion Matrix](visualizations/confusion_matrices.png)
*Classification confusion matrices showing 82% overall accuracy*

## 📊 Key Results

- **Best Regression Model**: Random Forest (R² = 0.75, RMSE = 620)
- **Best Classification Model**: Random Forest (82% accuracy)
- **Peak Traffic Impact**: Rush hours at 8 AM and 5 PM
- **Weather Effect**: -30% to -40% traffic reduction in adverse conditions

## 📂 Project Structure

```
traffic_congestion_prediction/
├── app.py                  # 🎮 Interactive Streamlit dashboard
├── dashboard_utils.py      # Dashboard helper functions
├── src/                    # Python modules
├── data/                   # Raw and processed datasets
├── models/                 # Trained ML models
├── visualizations/         # Generated plots (15+ charts)
├── Datasets/              # All datasets with documentation
├── README.md              # This file
├── REPORT_EXECUTIVE.md    # 5-page executive summary
└── REPORT.md              # 30-page technical report
```

## 🚀 Features

- Real dataset integration (UCI Traffic + Open-Meteo Weather API)
- 30+ engineered features (temporal, weather, lagged, rolling stats)
- 8 machine learning models (5 regression, 3 classification)
- Comprehensive visualizations and analysis
- Actionable recommendations for city planners
- Ethical AI framework

## 📖 Documentation

- **REPORT_EXECUTIVE.md** - 5-page executive summary
- **REPORT.md** - Comprehensive 30-page technical report
- **Datasets/DATASETS_INFO.md** - Complete dataset documentation

## 💡 Recommendations

The project identifies 4 key strategies for reducing traffic congestion by 10-20%:
1. Adaptive traffic signal optimization
2. Weather-based traffic management
3. Real-time route optimization
4. Demand-responsive public transport

## 🎓 Educational Value

Perfect for intermediate data science students learning:
- Data collection and API integration
- Feature engineering techniques
- Machine learning model comparison
- Data visualization
- Technical writing and reporting

## 📄 License

This project is created for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and extend this project for your own learning!

---

**For detailed analysis, see REPORT_EXECUTIVE.md (5 pages) or REPORT.md (30 pages)**
