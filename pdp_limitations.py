"""
pdp_limitations.py - A module demonstrating why Partial Dependence Plots can be misleading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def PDP_scatter(features, X, model):
    """
    Create a 2D Partial Dependence Plot with overlaid scatter points to show actual data distribution.
    
    Parameters:
        features (list): List of two feature indices to plot
        X (pd.DataFrame): Input dataset
        model: Fitted sklearn model
        
    Returns:
        None (displays plot)
    """
    disp = PartialDependenceDisplay.from_estimator(
        model, 
        X, 
        [features],
        kind='average',
        grid_resolution=100
    )
    
    # Access the contour plot
    contour = disp.axes_[0, 0].collections[0]
    
    # Overlay data points from the dataset
    disp.axes_[0, 0].scatter(
        X.iloc[:, features[0]], 
        X.iloc[:, features[1]], 
        color='black', 
        alpha=0.7, 
        marker='x', 
        label="Data Points"
    )
    
    # Final adjustments
    disp.axes_[0, 0].legend(loc="upper right")
    plt.title("2D Partial Dependence Plot with Data Distribution")
    plt.tight_layout()
    plt.show()

def conditional_expectation_plot(data, feature, predicted_values, bins=10):
    """
    Calculate and plot Conditional Expectation values using binning to show actual relationships.
    
    Parameters:
        data (pd.DataFrame): Dataset containing features
        feature (str): Feature name to analyze
        predicted_values (np.ndarray): Model predictions
        bins (int): Number of bins for discretizing the feature
        
    Returns:
        pd.DataFrame: DataFrame with bin centers and conditional means
    """
    data = data.copy()
    data['predicted'] = predicted_values
    data['bin'] = pd.cut(data[feature], bins=bins, include_lowest=True)
    
    bin_summary = data.groupby('bin').agg({
        feature: 'mean',
        'predicted': 'mean'
    }).reset_index()
    
    return bin_summary[[feature, 'predicted']]

def generate_correlated_data(n_samples, correlation=0.8, random_state=0):
    """
    Generate synthetic correlated variables to demonstrate PDP limitations with dependencies.
    
    Parameters:
        n_samples (int): Number of samples to generate
        correlation (float): Desired correlation between variables
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Two numpy arrays (x1, x2) with specified correlation
    """
    np.random.seed(random_state)
    
    # Generate uncorrelated standard normal variables
    z1 = np.random.normal(0, 1, n_samples)
    z2 = np.random.normal(0, 1, n_samples)
    
    # Apply Cholesky decomposition for correlation
    cov_matrix = np.array([[1.0, correlation], [correlation, 1.0]])
    L = np.linalg.cholesky(cov_matrix)
    
    x1, x2 = L @ np.array([z1, z2])
    return x1, x2

def plot_predicted_surface(X, model, feature_names=None):
    """
    Create a surface plot of model predictions to compare with PDP.
    
    Parameters:
        X (pd.DataFrame): Input dataset with features
        model: Fitted sklearn model
        feature_names (tuple): Optional names for x1, x2 features
        
    Returns:
        None (displays plot)
    """
    if feature_names is None:
        feature_names = ('x1', 'x2')
        
    # Create grid
    x1_grid = np.linspace(X[feature_names[0]].min(), X[feature_names[0]].max(), 50)
    x2_grid = np.linspace(X[feature_names[1]].min(), X[feature_names[1]].max(), 50)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
    
    # Prepare prediction data
    X_surface = pd.DataFrame({
        feature_names[0]: x1_mesh.ravel(),
        feature_names[1]: x2_mesh.ravel()
    })
    
    # Get predictions
    yhat_surface = model.predict(X_surface).reshape(x1_mesh.shape)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot prediction surface
    im = plt.imshow(
        yhat_surface,
        extent=[x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    
    # Add data points
    plt.scatter(
        X[feature_names[0]], 
        X[feature_names[1]],
        c='black',
        marker='x',
        alpha=0.5,
        label='Data Points'
    )
    
    # Formatting
    plt.colorbar(im, label='Model Predictions')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Model Predictions Surface vs PDP')
    plt.legend()
    plt.tight_layout()
    plt.show()

def demonstrate_pdp_problems(X, model, feature_pairs):
    """
    Demonstrate common PDP limitations by comparing PDP with actual predictions.
    
    Parameters:
        X (pd.DataFrame): Input dataset
        model: Fitted sklearn model
        feature_pairs (list): List of feature pairs to analyze
        
    Returns:
        None (displays multiple plots)
    """
    for pair in feature_pairs:
        plt.figure(figsize=(15, 5))
        
        # Plot 1: PDP
        plt.subplot(1, 2, 1)
        PDP_scatter(pair, X, model)
        plt.title("Partial Dependence Plot")
        
        # Plot 2: Actual Surface
        plt.subplot(1, 2, 2)
        plot_predicted_surface(X, model, feature_names=[f'x{pair[0]}', f'x{pair[1]}'])
        plt.title("Actual Prediction Surface")
        
        plt.tight_layout()
        plt.show()
def custom_pdp_plot(model, X, feature_index, feature_name, grid_range, num_points=50):
    """
    Custom function to generate PDP plots with manually controlled grid ranges.

    Parameters:
    - model: Trained model
    - X: Input DataFrame
    - feature_index: Index of the feature to plot
    - feature_name: Name of the feature to plot
    - grid_range: Tuple specifying the range of values for the feature grid (min, max)
    - num_points: Number of points to use in the grid (default=50)
    """
    # Create a grid of feature values within the specified range
    feature_grid = np.linspace(grid_range[0], grid_range[1], num_points)

    # Copy the dataset to manipulate the feature
    X_copy = X.copy()
    pdp_values = []

    # Calculate mean predictions for each value in the feature grid
    for value in feature_grid:
        X_copy.iloc[:, feature_index] = value
        predictions = model.predict(X_copy)
        pdp_values.append(predictions.mean())  # Average across all instances

    # Plot the PDP
    plt.figure(figsize=(10, 6))
    plt.plot(feature_grid, pdp_values, color='red', lw=2, label="PDP")
    plt.title(f"Custom PDP Plot for Feature: {feature_name}")
    plt.xlabel(f"{feature_name} (Grid Range: {grid_range[0]} to {grid_range[1]})")
    plt.ylabel("Average Predicted Target")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
