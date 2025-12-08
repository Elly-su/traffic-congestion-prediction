"""
Model Training Module for Traffic Congestion Prediction
Implements multiple regression and classification models.
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class TrafficModelTrainer:
    """Class for training traffic prediction models."""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.trained_models = {}
        
    def load_processed_data(self) -> tuple:
        """Load preprocessed data."""
        print("\n" + "="*70)
        print("  LOADING PROCESSED DATA")
        print("="*70)
        
        train_df = pd.read_csv('data/processed/train_data.csv')
        val_df = pd.read_csv('data/processed/val_data.csv')
        test_df = pd.read_csv('data/processed/test_data.csv')
        
        print(f"  ✓ Train set: {len(train_df):,} records")
        print(f"  ✓ Validation set: {len(val_df):,} records")
        print(f"  ✓ Test set: {len(test_df):,} records")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame, task: str = 'regression') -> tuple:
        """
        Prepare feature matrices and target vectors.
        
        Args:
            train_df, val_df, test_df: DataFrames
            task: 'regression' or 'classification'
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("\n" + "="*70)
        print(f"  PREPARING FEATURES FOR {task.upper()}")
        print("="*70)
        
        # Columns to exclude from features
        exclude_cols = ['date_time', 'traffic_volume', 'congestion_level', 
                       'congestion_level_encoded']
        
        # Select feature columns
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        X_test = test_df[feature_cols]
        
        if task == 'regression':
            y_train = train_df['traffic_volume']
            y_val = val_df['traffic_volume']
            y_test = test_df['traffic_volume']
        else:  # classification
            y_train = train_df['congestion_level_encoded']
            y_val = val_df['congestion_level_encoded']
            y_test = test_df['congestion_level_encoded']
        
        print(f"  ✓ Feature shape: {X_train.shape}")
        print(f"  ✓ Number of features: {len(feature_cols)}")
        print(f"  ✓ Target variable: {'traffic_volume' if task == 'regression' else 'congestion_level'}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_regression_models(self, X_train, y_train, X_val, y_val) -> dict:
        """Train multiple regression models."""
        print("\n" + "="*70)
        print("  TRAINING REGRESSION MODELS")
        print("="*70)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=20, 
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        trained_models = {}
        results = []
        
        for name, model in models.items():
            print(f"\n  Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            
            results.append({
                'Model': name,
                'Train RMSE': train_rmse,
                'Val RMSE': val_rmse,
                'Train R²': train_r2,
                'Val R²': val_r2,
                'Val MAE': val_mae
            })
            
            print(f"    Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f}")
            print(f"    Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}")
            
            # Save model
            model_path = os.path.join(self.models_dir, f'{name.replace(" ", "_").lower()}_reg.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            trained_models[name] = model
        
        # Print comparison
        results_df = pd.DataFrame(results)
        print("\n" + "="*70)
        print("  REGRESSION MODELS COMPARISON")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv(os.path.join(self.models_dir, 'regression_results.csv'), index=False)
        
        return trained_models
    
    def train_classification_models(self, X_train, y_train, X_val, y_val) -> dict:
        """Train multiple classification models."""
        print("\n" + "="*70)
        print("  TRAINING CLASSIFICATION MODELS")
        print("="*70)
        
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                random_state=42
            )
        }
        
        trained_models = {}
        results = []
        
        for name, model in models.items():
            print(f"\n  Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_val, val_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_val, val_pred, average='weighted', zero_division=0)
            
            results.append({
                'Model': name,
                'Train Accuracy': train_acc,
                'Val Accuracy': val_acc,
                'Val Precision': val_precision,
                'Val Recall': val_recall,
                'Val F1-Score': val_f1
            })
            
            print(f"    Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")
            print(f"    Val F1-Score: {val_f1:.4f}")
            
            # Save model
            model_path = os.path.join(self.models_dir, f'{name.replace(" ", "_").lower()}_clf.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            trained_models[name] = model
        
        # Print comparison
        results_df = pd.DataFrame(results)
        print("\n" + "="*70)
        print("  CLASSIFICATION MODELS COMPARISON")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv(os.path.join(self.models_dir, 'classification_results.csv'), index=False)
        
        return trained_models
    
    def plot_feature_importance(self, model, feature_names: list, 
                               model_name: str, top_n: int = 20) -> None:
        """Plot feature importance for tree-based models."""
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
            plt.title(f'Top {top_n} Feature Importances - {model_name}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            
            filename = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(os.path.join('visualizations', filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ Saved feature importance plot: {filename}")


def main():
    """Main model training pipeline."""
    print("\n" + "="*70)
    print("  TRAFFIC PREDICTION MODEL TRAINING")
    print("="*70)
    
    trainer = TrafficModelTrainer()
    
    # Load data
    train_df, val_df, test_df = trainer.load_processed_data()
    
    # ===== REGRESSION TASK =====
    print("\n\n" + "="*70)
    print("  TASK 1: TRAFFIC VOLUME REGRESSION")
    print("="*70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_features(
        train_df, val_df, test_df, task='regression'
    )
    
    regression_models = trainer.train_regression_models(X_train, y_train, X_val, y_val)
    
    # Plot feature importance for Random Forest
    print("\n  Generating feature importance plots...")
    feature_names = X_train.columns.tolist()
    trainer.plot_feature_importance(
        regression_models['Random Forest'], 
        feature_names, 
        'Random Forest Regression'
    )
    
    # ===== CLASSIFICATION TASK =====
    print("\n\n" + "="*70)
    print("  TASK 2: CONGESTION LEVEL CLASSIFICATION")
    print("="*70)
    
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_features(
        train_df, val_df, test_df, task='classification'
    )
    
    classification_models = trainer.train_classification_models(X_train, y_train, X_val, y_val)
    
    # Plot feature importance for Random Forest
    trainer.plot_feature_importance(
        classification_models['Random Forest'],
        feature_names,
        'Random Forest Classification'
    )
    
    print("\n" + "="*70)
    print("  ✓ MODEL TRAINING COMPLETE")
    print(f"  ✓ All models saved to: {trainer.models_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
