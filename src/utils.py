"""
Utility functions for the Traffic Congestion Prediction project.
Provides helper functions for data loading, model persistence, and visualization.
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Any, Dict, Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        data = pd.read_csv(filepath)
        print(f"✓ Successfully loaded data from {filepath}")
        print(f"  Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"✗ Error loading data from {filepath}: {e}")
        raise


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk using pickle.
    
    Args:
        model: Trained model object
        filepath: Path where to save the model
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved to {filepath}")
    except Exception as e:
        print(f"✗ Error saving model to {filepath}: {e}")
        raise


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"✗ Error loading model from {filepath}: {e}")
        raise


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: list, title: str = "Confusion Matrix",
                         save_path: str = None) -> None:
    """
    Plot and optionally save a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     task_type: str = 'regression') -> Dict[str, float]:
    """
    Calculate evaluation metrics based on task type.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: 'regression' or 'classification'
        
    Returns:
        Dictionary containing metric names and values
    """
    from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                                 r2_score, accuracy_score, precision_score,
                                 recall_score, f1_score)
    
    metrics = {}
    
    if task_type == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(mse)
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        
        # MAPE (handling zero values)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['MAPE'] = mape
            
    elif task_type == 'classification':
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['F1-Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        model_name: Name of the model (optional)
    """
    if model_name:
        print(f"\n{'='*60}")
        print(f"  {model_name} - Performance Metrics")
        print(f"{'='*60}")
    
    for metric_name, value in metrics.items():
        print(f"  {metric_name:15s}: {value:8.4f}")
    print()


def create_output_dir(directory: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def set_plotting_style() -> None:
    """Set consistent plotting style for all visualizations."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
