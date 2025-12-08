@echo off
REM Git Commands for Traffic Congestion Prediction Project
REM Run this batch file to initialize Git and make the first commit

echo Initializing Git repository...
git init

echo Configuring Git user...
git config user.name "Elly-su"
git config user.email "ellyaston404@gmail.com"

echo Checking Git status...
git status

echo Adding all files...
git add .

echo Creating initial commit...
git commit -m "Initial commit: Urban Traffic Congestion Prediction System - Complete ML system with R^2 = 0.75 (regression), 82%% accuracy (classification) - Python codebase: 5 modules, ~2,030 lines - Comprehensive documentation and reports - 8 ML models, data pipeline, visualizations"

echo.
echo ================================
echo Git repository initialized!
echo ================================
echo.
echo Next steps:
echo 1. Create a repository on GitHub: https://github.com/new
echo    Repository name: traffic-congestion-prediction
echo.
echo 2. Link to remote:
echo    git remote add origin https://github.com/Elly-su/traffic-congestion-prediction.git
echo.
echo 3. Push to GitHub:
echo    git branch -M main
echo    git push -u origin main
echo.
pause
