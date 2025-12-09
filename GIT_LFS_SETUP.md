# Git LFS Setup for Large Model Files

## Why We Need This
GitHub has a soft limit of 50MB and hard limit of 100MB per file. Your models:
- `random_forest_reg.pkl` = 38MB (at the edge)
- `random_forest_clf.pkl` = 18MB

Files this size can fail to push properly without Git LFS.

## Setup Steps

### 1. Install Git LFS (if not installed)
```bash
# Check if already installed
git lfs version

# If not found, download from: https://git-lfs.github.com/
# Or install via package manager
```

### 2. Initialize Git LFS in Your Repo
```bash
cd C:\Users\ellio\.gemini\antigravity\scratch\traffic_congestion_prediction

# Initialize LFS
git lfs install

# Track .pkl files
git lfs track "models/*.pkl"

# Add the tracking config
git add .gitattributes

# Verify tracking
git lfs ls-files
```

### 3. Remove Old Files and Re-add with LFS
```bash
# Remove files from git history (keeps local files)
git rm --cached models/*.pkl

# Re-add with LFS tracking
git add models/*.pkl

# Commit
git commit -m "Track model files with Git LFS"

# Push with LFS
git push origin main
```

### 4. Verify Upload
After push, check: https://github.com/Elly-su/traffic-congestion-prediction/tree/main/models

Files tracked by LFS will show:
- File size listed
- "Stored with Git LFS" badge

## Alternative: Smaller Models Only

If Git LFS doesn't work, deploy only small models:
```bash
# Remove large models from git
git rm models/random_forest_*.pkl
git rm models/svm_clf.pkl

# Keep only small models
git add models/linear_regression_reg.pkl
git add models/lasso_regression_reg.pkl
git add models/ridge_regression_reg.pkl
git add models/gradient_boosting_reg.pkl
git add models/logistic_regression_clf.pkl

git commit -m "Deploy smaller models only"
git push origin main
```

Then update `app.py` to use Gradient Boosting or Linear Regression as default.

## Check Current Status
```bash
# See tracked LFS files
git lfs ls-files

# See file sizes in repo
git ls-files -s models/*.pkl
```
