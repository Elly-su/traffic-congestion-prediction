"""
Model Evaluation Module for Traffic Congestion Prediction
Comprehensive evaluation and visualization of model performance.
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report)


class ModelEvaluator:
    """Class for evaluating trained models."""
    
    def __init__(self, models_dir: str = 'models', viz_dir: str = 'visualizations'):
        self.models_dir = models_dir
        self.viz_dir = viz_dir
        os.makedirs(viz_dir, exist_ok=True)
    
    def load_data(self) -> tuple:
        """Load test data."""
        print("\n" + "="*70)
        print("  LOADING TEST DATA")
        print("="*70)
        
        test_df = pd.read_csv('data/processed/test_data.csv')
        print(f"  ✓ Test set: {len(test_df):,} records")
        
        return test_df
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def evaluate_regression_models(self, test_df: pd.DataFrame) -> None:
        """Evaluate all regression models on test set."""
        print("\n" + "="*70)
        print("  EVALUATING REGRESSION MODELS")
        print("="*70)
        
        # Prepare features
        exclude_cols = ['date_time', 'traffic_volume', 'congestion_level', 
                       'congestion_level_encoded']
        feature_cols = [col for col in test_df.columns if col not in exclude_cols]
        
        X_test = test_df[feature_cols]
        y_test = test_df['traffic_volume']
        
        # Load and evaluate each model
        regression_models = [
            'linear_regression_reg.pkl',
            'ridge_regression_reg.pkl',
            'lasso_regression_reg.pkl',
            'random_forest_reg.pkl',
            'gradient_boosting_reg.pkl'
        ]
        
        results = []
        predictions = {}
        
        for model_file in regression_models:
            model_path = os.path.join(self.models_dir, model_file)
            if not os.path.exists(model_path):
                continue
                
            model_name = model_file.replace('_reg.pkl', '').replace('_', ' ').title()
            print(f"\n  Evaluating {model_name}...")
            
            model = self.load_model(model_path)
            y_pred = model.predict(X_test)
            predictions[model_name] = y_pred
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # MAPE
            mask = y_test != 0
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            
            results.append({
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2,
                'MAPE (%)': mape
            })
            
            print(f"    RMSE: {rmse:.2f}")
            print(f"    MAE: {mae:.2f}")
            print(f"    R²: {r2:.4f}")
            print(f"    MAPE: {mape:.2f}%")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        print("\n" + "="*70)
        print("  REGRESSION MODELS - TEST SET PERFORMANCE")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv(os.path.join(self.models_dir, 'test_regression_results.csv'), index=False)
        
        # Visualizations
        self.plot_regression_comparison(results_df)
        self.plot_predictions_vs_actual(y_test, predictions)
        self.plot_residuals(y_test, predictions)
        
    def evaluate_classification_models(self, test_df: pd.DataFrame) -> None:
        """Evaluate all classification models on test set."""
        print("\n" + "="*70)
        print("  EVALUATING CLASSIFICATION MODELS")
        print("="*70)
        
        # Prepare features
        exclude_cols = ['date_time', 'traffic_volume', 'congestion_level', 
                       'congestion_level_encoded']
        feature_cols = [col for col in test_df.columns if col not in exclude_cols]
        
        X_test = test_df[feature_cols]
        y_test = test_df['congestion_level_encoded']
        
        # Load and evaluate each model
        classification_models = [
            'logistic_regression_clf.pkl',
            'random_forest_clf.pkl',
            'svm_clf.pkl'
        ]
        
        results = []
        predictions = {}
        
        for model_file in classification_models:
            model_path = os.path.join(self.models_dir, model_file)
            if not os.path.exists(model_path):
                continue
                
            model_name = model_file.replace('_clf.pkl', '').replace('_', ' ').title()
            print(f"\n  Evaluating {model_name}...")
            
            model = self.load_model(model_path)
            y_pred = model.predict(X_test)
            predictions[model_name] = y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1-Score: {f1:.4f}")
            
            # Classification report
            print(f"\n    Classification Report:")
            report = classification_report(y_test, y_pred, 
                                         target_names=['Low', 'Medium', 'High'],
                                         zero_division=0)
            print(report)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        print("\n" + "="*70)
        print("  CLASSIFICATION MODELS - TEST SET PERFORMANCE")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv(os.path.join(self.models_dir, 'test_classification_results.csv'), index=False)
        
        # Visualizations
        self.plot_classification_comparison(results_df)
        self.plot_confusion_matrices(y_test, predictions)
    
    def plot_regression_comparison(self, results_df: pd.DataFrame) -> None:
        """Plot comparison of regression models."""
        print("\n  Creating regression comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['RMSE', 'MAE', 'R² Score', 'MAPE (%)']
        colors = ['steelblue', 'coral', 'green', 'purple']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            
            data = results_df.sort_values(metric, ascending=(metric != 'R² Score'))
            
            ax.barh(data['Model'], data[metric], color=color, edgecolor='black')
            ax.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax.set_ylabel('Model', fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            
            # Add value labels
            for i, v in enumerate(data[metric]):
                ax.text(v, i, f' {v:.2f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'regression_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: regression_comparison.png")
    
    def plot_predictions_vs_actual(self, y_test, predictions: dict) -> None:
        """Plot predicted vs actual values for regression models."""
        print("\n  Creating predictions vs actual plots...")
        
        n_models = len(predictions)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (model_name, y_pred) in enumerate(predictions.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.3, s=10)
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Perfect Prediction')
            
            # R² score
            r2 = r2_score(y_test, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Actual Traffic Volume', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted Traffic Volume', fontsize=10, fontweight='bold')
            ax.set_title(model_name, fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(predictions), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'predictions_vs_actual.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: predictions_vs_actual.png")
    
    def plot_residuals(self, y_test, predictions: dict) -> None:
        """Plot residuals for regression models."""
        print("\n  Creating residual plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (model_name, y_pred) in enumerate(predictions.items()):
            if idx >= len(axes):
                break
                
            residuals = y_test - y_pred
            ax = axes[idx]
            
            ax.scatter(y_pred, residuals, alpha=0.3, s=10)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            
            ax.set_xlabel('Predicted Values', fontsize=10, fontweight='bold')
            ax.set_ylabel('Residuals', fontsize=10, fontweight='bold')
            ax.set_title(f'{model_name} - Residual Plot', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(predictions), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'residual_plots.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: residual_plots.png")
    
    def plot_classification_comparison(self, results_df: pd.DataFrame) -> None:
        """Plot comparison of classification models."""
        print("\n  Creating classification comparison plot...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(results_df))
        width = 0.2
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['steelblue', 'coral', 'green', 'purple']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - 1.5)
            ax.bar(x + offset, results_df[metric], width, 
                  label=metric, color=color, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Classification Models Performance Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'classification_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: classification_comparison.png")
    
    def plot_confusion_matrices(self, y_test, predictions: dict) -> None:
        """Plot confusion matrices for classification models."""
        print("\n  Creating confusion matrices...")
        
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        class_labels = ['Low', 'Medium', 'High']
        
        for idx, (model_name, y_pred) in enumerate(predictions.items()):
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_labels, yticklabels=class_labels,
                       ax=axes[idx], cbar=True)
            
            axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{model_name}\nConfusion Matrix', 
                              fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'confusion_matrices.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: confusion_matrices.png")


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*70)
    print("  MODEL EVALUATION ON TEST SET")
    print("="*70)
    
    evaluator = ModelEvaluator()
    
    # Load test data
    test_df = evaluator.load_data()
    
    # Evaluate regression models
    evaluator.evaluate_regression_models(test_df)
    
    # Evaluate classification models
    evaluator.evaluate_classification_models(test_df)
    
    print("\n" + "="*70)
    print("  ✓ EVALUATION COMPLETE")
    print(f"  ✓ Results saved to: {evaluator.models_dir}/")
    print(f"  ✓ Visualizations saved to: {evaluator.viz_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
